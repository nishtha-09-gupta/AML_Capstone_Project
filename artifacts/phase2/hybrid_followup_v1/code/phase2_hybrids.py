"""Follow-up hybrids: continuous residual correction and cross-fitted calibration.

Original Phase 2 classes and artifacts are unchanged. All fit calls accept only
caller-supplied training rows; outer validation, C and T are never required.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from advanced_modeling import make_preprocessor
from phase2_baselines import DiscreteHeartModel
from phase2_models import HeartMLP, cross_fit_bn, log_odds


class ResidualLogitNetwork(nn.Module):
    """Start exactly at the BN; learn only a shrunken continuous correction."""
    def __init__(self, dimensions, hidden, dropout, strength):
        super().__init__()
        layers=[]
        for width in hidden:
            layers.extend([nn.Linear(dimensions,width),nn.ReLU(),nn.Dropout(dropout)])
            dimensions=width
        self.features=nn.Sequential(*layers)
        self.output=nn.Linear(dimensions,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.strength=strength

    def forward(self, matrix):
        return matrix[:,-1:] + self.strength*self.output(self.features(matrix[:,:-1]))


class BNResidualMLP(HeartMLP):
    def __init__(self, strength=.5, random_state=42, max_epochs=150):
        super().__init__(hybrid='bn',random_state=random_state,max_epochs=max_epochs)
        self.strength=strength

    def _prepare_fit(self,x,y):
        preprocessor=make_preprocessor()
        features=preprocessor.fit_transform(x)
        p,folds=cross_fit_bn(x,y)
        bundle=dict(preprocessor=preprocessor,fit_indices=x.index.tolist(),
                    bn=DiscreteHeartModel().fit(x,y),cross_fit_folds=folds)
        return np.column_stack([features,log_odds(p)]).astype(np.float32),bundle

    def _transform(self,x,bundle):
        return np.column_stack([bundle['preprocessor'].transform(x),
            log_odds(bundle['bn'].predict_proba(x)[:,1])]).astype(np.float32)

    def _network(self,dimensions):
        return ResidualLogitNetwork(dimensions-1,self.hidden,self.dropout,self.strength)

    def fit(self,x,y):
        if not np.isfinite(self.strength) or self.strength<=0:
            raise ValueError('Residual strength must be finite and positive')
        super().fit(x,y)
        self.input_features_[-1]='bn_log_odds_offset'
        return self


def base_pair(x,y,seed=42,max_epochs=150):
    """Cross-fit both base learners afresh within exactly this training subset."""
    y=pd.Series(np.asarray(y),index=x.index)
    oof=np.empty((len(x),2))
    audits=[]
    for train,valid in StratifiedKFold(3,shuffle=True,random_state=1729).split(x,y):
        bn=DiscreteHeartModel().fit(x.iloc[train],y.iloc[train])
        mlp=HeartMLP(random_state=seed,max_epochs=max_epochs).fit(x.iloc[train],y.iloc[train])
        oof[valid,0]=log_odds(bn.predict_proba(x.iloc[valid])[:,1]).ravel()
        oof[valid,1]=log_odds(mlp.predict_proba(x.iloc[valid])[:,1]).ravel()
        audits.append(dict(train_indices=x.index[train].tolist(),
                           validation_indices=x.index[valid].tolist(),
                           mlp_audit=mlp.audit_))
    bases=[DiscreteHeartModel().fit(x,y),
           HeartMLP(random_state=seed,max_epochs=max_epochs).fit(x,y)]
    return oof,bases,audits


class CalibratedHybrid(ClassifierMixin,BaseEstimator):
    """Logistic stacking; single-base modes isolate calibration from fusion.

    component='both' learns a BN+MLP stack. 'bn' and 'mlp' are its single-base
    calibration controls, fitted using exactly the same out-of-fold predictions.
    The training package may be shared only for identical x/y/seed settings.
    """
    def __init__(self,component='both',C=1.,random_state=42,max_epochs=150):
        self.component=component
        self.C=C
        self.random_state=random_state
        self.max_epochs=max_epochs

    def fit(self,x,y,training_package=None):
        if self.component not in {'bn','mlp','both'}:
            raise ValueError('Unknown component')
        if (not x.index.is_unique or len(x)!=len(y) or set(np.unique(y))!={0,1}
                or (isinstance(y,pd.Series) and not y.index.equals(x.index))):
            raise ValueError('Aligned binary rows with unique indices required')
        if training_package is None:
            training_package=base_pair(x,y,self.random_state,self.max_epochs)
        oof,self.bases_,self.cross_fit_audit_=training_package
        self.columns_={'bn':[0],'mlp':[1],'both':[0,1]}[self.component]
        self.meta_=LogisticRegression(C=self.C,max_iter=1000,solver='lbfgs').fit(oof[:,self.columns_],y)
        self.classes_=np.array([0,1])
        self.fit_indices_=x.index.tolist()
        return self

    def predict_proba(self,x):
        check_is_fitted(self,'meta_')
        matrix=np.column_stack([log_odds(model.predict_proba(x)[:,1]).ravel() for model in self.bases_])
        return self.meta_.predict_proba(matrix[:,self.columns_])

    def predict(self,x):
        return (self.predict_proba(x)[:,1]>=.5).astype(int)
