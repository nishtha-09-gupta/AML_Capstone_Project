"""Finite exploratory nested-CV follow-up; S only, no C/T evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import time
from datetime import datetime,timezone
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold,train_test_split
from threadpoolctl import threadpool_limits
from data_loader import load_dataset
from advanced_modeling import clean_features
from experiment import write_json
from phase2_models import HeartMLP
from phase2_hybrids import BNResidualMLP,CalibratedHybrid,base_pair
from phase2_baselines import DiscreteHeartModel
from phase2_train import metrics,describe,ROOT
from train import prepare_partitions,positive_probability

FAMILIES=['BN','Tuned MLP','Original BN feature MLP','BN residual MLP',
          'Calibrated BN','Calibrated MLP','Calibrated BN-MLP']
GRIDS={'BN':[None],'Tuned MLP':[.0001,.001,.01],
       'Original BN feature MLP':[None],'BN residual MLP':[.25,.5,1.],
       'Calibrated BN':[.1,1.,10.],'Calibrated MLP':[.1,1.,10.],
       'Calibrated BN-MLP':[.1,1.,10.]}


class TrainingCache:
    """Reuse only identical fitted subsets, never an outer OOF matrix.

    Each outer experiment owns a new cache. The row IDs AND feature/target
    contents identify a subset; package reuse changes only meta C/component.
    """
    def __init__(self,seed,smoke=False):
        self.seed=seed;self.epochs=3 if smoke else 150
        self.models={};self.packages={}

    def fit(self,name,value,x,y):
        content=pd.util.hash_pandas_object(x,index=True).values.tobytes()+np.asarray(y).tobytes()
        key=hashlib.sha256(content).hexdigest()
        model_key=(key,name,value)
        if model_key in self.models:return self.models[model_key]
        if name.startswith('Calibrated'):
            if key not in self.packages:self.packages[key]=base_pair(x,y,self.seed,self.epochs)
            component={'Calibrated BN':'bn','Calibrated MLP':'mlp','Calibrated BN-MLP':'both'}[name]
            model=CalibratedHybrid(component,C=value,random_state=self.seed,max_epochs=self.epochs)
            model.fit(x,y,self.packages[key])
        elif name=='BN':model=DiscreteHeartModel().fit(x,y)
        elif name=='BN residual MLP':
            model=BNResidualMLP(strength=value,random_state=self.seed,max_epochs=self.epochs).fit(x,y)
        else:
            model=HeartMLP(hybrid='bn' if name=='Original BN feature MLP' else 'none',
                weight_decay=.001 if value is None else value,
                random_state=self.seed,max_epochs=self.epochs).fit(x,y)
        self.models[model_key]=model
        return model


def detail(model):
    if isinstance(model,CalibratedHybrid):
        return dict(fit_indices=model.fit_indices_,cross_fit_audit=model.cross_fit_audit_,
            coefficients=model.meta_.coef_,intercept=model.meta_.intercept_,
            component=model.component,C=model.C,base_mlp=describe(model.bases_[1]),
            base_bn=model.bases_[0].explanation())
    result=describe(model)
    if isinstance(model,BNResidualMLP):
        result['residual_strength']=model.strength
        result['offset']='Raw cross-fitted BN log-odds, not a standardized feature'
    return result


def select(name,x,y,ids,cache,smoke=False):
    cv=list(StratifiedKFold(2 if smoke else 3,shuffle=True,random_state=145).split(x,y))
    searches=[]
    for value in GRIDS[name]:
        fold_scores=[]
        for train,valid in cv:
            model=cache.fit(name,value,x.iloc[train],y.iloc[train])
            p=positive_probability(model,x.iloc[valid])
            fold_scores.append(metrics(y.iloc[valid],p))
        searches.append(dict(parameter=value,mean_brier=float(np.mean([r['brier'] for r in fold_scores])),
            mean_auc=float(np.mean([r['roc_auc'] for r in fold_scores])),fold_scores=fold_scores))
    best=min(range(len(searches)),key=lambda i:(searches[i]['mean_brier'],-searches[i]['mean_auc'],i))
    return GRIDS[name][best],dict(candidates=searches,selected_index=best,
        folds=[dict(train_ids=ids.iloc[a].tolist(),validation_ids=ids.iloc[b].tolist()) for a,b in cv])


def run(output,smoke=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    logging.getLogger('pgmpy').setLevel(logging.ERROR)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    d,t=prepare_partitions(load_dataset())
    s,c=train_test_split(d,test_size=.25,stratify=d.num>0,random_state=43)
    x,y=clean_features(s),(s.num>0).astype(int)
    seeds=[42] if smoke else [42,52,62]
    folds=list(StratifiedKFold(2 if smoke else 5,shuffle=True,random_state=144).split(x,y))
    write_json(output/'config.json',dict(smoke=smoke,seeds=seeds,grids=GRIDS,
        primary='mean outer Brier; lower is better',selection_ids=s.id.tolist(),
        forbidden_ids=c.id.tolist()+t.id.tolist(),outer_seed=144,inner_seed=145,
        outer_folds=[dict(train_ids=s.id.iloc[a].tolist(),validation_ids=s.id.iloc[b].tolist()) for a,b in folds],
        source_sha256=hashlib.sha256((ROOT/'heart_disease_uci.csv').read_bytes()).hexdigest(),
        exploratory_after_prior_results=True))
    shutil.copy2(ROOT/'docs/phase2/hybrid_improvement_protocol.md',output/'protocol.md')
    code=output/'code';code.mkdir()
    for name in ['phase2_improve.py','phase2_hybrids.py','phase2_models.py','phase2_baselines.py','advanced_modeling.py']:
        shutil.copy2(ROOT/name,code/name)
    write_json(output/'environment.json',{p:importlib.metadata.version(p) for p in
        ['numpy','pandas','scikit-learn','pgmpy','torch','joblib']})
    models_dir=output/'models';models_dir.mkdir()
    manifest=dict(status='running',started=datetime.now(timezone.utc).isoformat())
    write_json(output/'manifest.json',manifest)
    predictions=[];scores=[];searches=[];audits=[];hashes={}
    started=time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            for seed in seeds:
                for fold,(fit,valid) in enumerate(folds):
                    cache=TrainingCache(seed,smoke)
                    for name in FAMILIES:
                        value,search=select(name,x.iloc[fit],y.iloc[fit],s.id.iloc[fit],cache,smoke)
                        model=cache.fit(name,value,x.iloc[fit],y.iloc[fit])
                        p=positive_probability(model,x.iloc[valid])
                        key=f'{name.replace(" ","_")}__{seed}__{fold}'
                        file=models_dir/(key+'.joblib');joblib.dump(model,file)
                        hashes[key]=hashlib.sha256(file.read_bytes()).hexdigest()
                        searches.append(dict(model=name,seed=seed,fold=fold,**search))
                        scores.append(dict(model=name,seed=seed,fold=fold,parameter=value,
                                           **metrics(y.iloc[valid],p)))
                        predictions.extend(dict(model=name,seed=seed,fold=fold,source_id=int(i),
                            target=int(a),probability=float(b)) for i,a,b in zip(s.id.iloc[valid],y.iloc[valid],p))
                        audits.append(dict(model=name,seed=seed,fold=fold,details=detail(model)))
                    print(f'Nested outer fold {fold+1}/{len(folds)}, seed {seed}: all families finished; elapsed {time.perf_counter()-started:.1f}s',flush=True)
                    write_json(output/'scores.json',scores);write_json(output/'searches.json',searches)
                    write_json(output/'training_details.json',audits)
                    pd.DataFrame(predictions).to_csv(output/'predictions.csv',index=False)
                    write_json(output/'model_sha256.json',hashes)
            manifest.update(status='complete',seconds=time.perf_counter()-started,
                            finished=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status='failed',error=repr(error));raise
    finally:write_json(output/'manifest.json',manifest)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.output,args.smoke)
