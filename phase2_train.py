"""Run the predeclared small Phase 2 study; never train on import.

Usage: python phase2_train.py --output artifacts/phase2/run_name
Use --smoke only to exercise the workflow, never for reported conclusions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import shutil
import subprocess
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from advanced_modeling import clean_features, make_preprocessor
from experiment import write_json
from phase2_baselines import DiscreteHeartModel
from phase2_models import HeartMLP
from train import prepare_partitions, choose_threshold, scores, positive_probability

ROOT = Path(__file__).resolve().parent
PRIMARY = ['BN', 'Naive Bayes', 'MLP', 'BN + MLP']
NEURAL = ['MLP', 'BN + MLP', 'MLP no dropout', 'MLP no weight decay', 'NB + MLP']
MODELS = PRIMARY + ['Logistic regression', 'Random forest'] + NEURAL[2:]
SEEDS = [42, 52, 62]


def make_model(name, seed=42, smoke=False):
    if name in ['BN', 'Naive Bayes']:
        return DiscreteHeartModel(structured=name == 'BN')
    if name == 'Logistic regression':
        return make_pipeline(make_preprocessor(), LogisticRegression(C=1, max_iter=1000, random_state=42))
    if name == 'Random forest':
        return make_pipeline(make_preprocessor(), RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=1))
    options = dict(hybrid={'BN + MLP': 'bn', 'NB + MLP': 'nb'}.get(name, 'none'),
                   dropout=0 if name == 'MLP no dropout' else .2,
                   weight_decay=0 if name == 'MLP no weight decay' else .001,
                   max_epochs=3 if smoke else 150, random_state=seed)
    return HeartMLP(**options)


def metrics(y, p, threshold=.5):
    result = scores(y, p, threshold)
    precision, recall, _ = precision_recall_curve(y, p)
    result['pr_auc_trapezoid'] = float(auc(recall, precision))
    result['sensitivity'] = result['recall']
    bins = np.minimum((np.asarray(p) * 5).astype(int), 4)
    result['reliability'] = [dict(lower=i/5, upper=(i+1)/5, n=int((bins == i).sum()),
        predicted=float(np.mean(p[bins == i])) if (bins == i).any() else None,
        observed=float(np.mean(np.asarray(y)[bins == i])) if (bins == i).any() else None) for i in range(5)]
    result['ece_5_bins'] = sum(b['n'] * abs(b['predicted'] - b['observed'])
                             for b in result['reliability'] if b['n']) / len(y)
    return result


def describe(model):
    if isinstance(model, HeartMLP):
        result = dict(audit=model.audit_, best_epoch=model.best_epoch_,
                      parameters=model.parameter_count_, inputs=model.input_features_,
                      early_stopping_history=model.history_, refit_history=model.refit_history_)
        if model.hybrid != 'none':
            result['inference_bn'] = model.bundle_['bn'].explanation()
        return result
    if isinstance(model, DiscreteHeartModel):
        return model.explanation()
    return {'parameters': {k: v for k, v in model.get_params().items()
                           if isinstance(v, (str, int, float, bool, type(None)))
                           and not (isinstance(v, float) and not np.isfinite(v))}}


def cross_validate(data, folder, names, seeds, smoke=False):
    folder.mkdir(parents=True)
    x, y = clean_features(data), (data.num > 0).astype(int)
    folds = list(StratifiedKFold(2 if smoke else 5, shuffle=True, random_state=44).split(x, y))
    write_json(folder / 'folds.json', [{'train_ids': data.id.iloc[a].tolist(),
        'validation_ids': data.id.iloc[b].tolist()} for a, b in folds])
    predictions, summaries, audits = [], [], []
    for name in names:
        for seed in seeds if name in NEURAL else [42]:
            for fold, (fit, valid) in enumerate(folds):
                started = time.perf_counter()
                model = make_model(name, seed, smoke).fit(x.iloc[fit], y.iloc[fit])
                p = positive_probability(model, x.iloc[valid])
                summary = dict(model=name, seed=seed, fold=fold,
                               seconds=time.perf_counter()-started, **metrics(y.iloc[valid], p))
                summaries.append(summary)
                predictions.extend(dict(model=name, seed=seed, fold=fold, source_id=int(i),
                    target=int(t), probability=float(v)) for i,t,v in zip(data.id.iloc[valid],y.iloc[valid],p))
                audits.append(dict(model=name, seed=seed, fold=fold, details=describe(model)))
            print(f'CV {folder.name}: {name}, seed {seed} finished', flush=True)
    write_json(folder / 'scores.json', summaries)
    write_json(folder / 'training_details.json', audits)
    pd.DataFrame(predictions).to_csv(folder / 'predictions.csv', index=False)
    return summaries


def frozen_evaluation(selection, cutoff, evaluation, folder, names, seeds, smoke=False):
    """Fit on S, choose C cutoffs, persist frozen state, then score evaluation."""
    folder.mkdir(parents=True)
    groups = [set(d.id) for d in [selection, cutoff, evaluation]]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('S, C and evaluation must be disjoint')
    write_json(folder/'partitions.json', dict(selection=selection.id.tolist(),
        cutoff=cutoff.id.tolist(), evaluation=evaluation.id.tolist()))
    models, decisions, cutoff_rows, details = {}, {}, [], {}
    xs, ys = clean_features(selection), (selection.num > 0).astype(int)
    xc, yc = clean_features(cutoff), (cutoff.num > 0).astype(int)
    for name in names:
        for seed in seeds if name in NEURAL else [42]:
            key = f'{name}__{seed}'
            model = make_model(name, seed, smoke).fit(xs, ys)
            p = positive_probability(model, xc)
            decision = choose_threshold(yc, p)
            file = folder / (key.replace(' ', '_') + '.joblib')
            joblib.dump(model, file)
            models[key] = model
            decisions[key] = dict(model=name, seed=seed, threshold=decision['threshold'],
                threshold_search=decision, file=file.name, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            cutoff_rows.extend(dict(key=key, source_id=int(i), target=int(t), probability=float(v))
                               for i,t,v in zip(cutoff.id, yc, p))
            details[key] = describe(model)
    write_json(folder/'freeze.json', dict(frozen_at=datetime.now(timezone.utc).isoformat(), models=decisions))
    write_json(folder/'training_details.json', details)
    pd.DataFrame(cutoff_rows).to_csv(folder/'cutoff_predictions.csv', index=False)
    # Evaluation labels first used here, after every model and threshold is frozen.
    xt, yt = clean_features(evaluation), (evaluation.num > 0).astype(int)
    predictions, results = [], {}
    for key, model in models.items():
        p = positive_probability(model, xt)
        cutoff_value = decisions[key]['threshold']
        results[key] = dict(default=metrics(yt, p), tuned=metrics(yt, p, cutoff_value),
                            threshold=cutoff_value)
        predictions.extend(dict(key=key, source_id=int(i), target=int(t), probability=float(v))
                           for i,t,v in zip(evaluation.id, yt, p))
    prior = np.full(len(yt), float(ys.mean()))
    results['constant_reference'] = dict(default=metrics(yt, prior),
        always_positive=metrics(yt, prior, 0), prior_from_selection=float(ys.mean()))
    write_json(folder/'metrics.json', results)
    pd.DataFrame(predictions).to_csv(folder/'evaluation_predictions.csv', index=False)
    print(f'Frozen evaluation {folder.name} finished', flush=True)


def run(output, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    logging.getLogger('pgmpy').setLevel(logging.ERROR)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds = [42] if smoke else SEEDS
    config = dict(smoke=smoke, model_order=MODELS, primary_order=PRIMARY, seeds=seeds,
        folds=2 if smoke else 5, selection_metric='mean fold F1 at 0.5 across all declared seeds',
        models={name: make_model(name, smoke=smoke).get_params(deep=False) for name in NEURAL},
        source_sha256=hashlib.sha256((ROOT/'heart_disease_uci.csv').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((ROOT/'docs/phase2/protocol.md').read_bytes()).hexdigest())
    write_json(output/'config.json', config)
    shutil.copy2(ROOT/'docs/phase2/protocol.md', output/'protocol.md')
    code = output/'code'
    code.mkdir()
    for path in [*ROOT.glob('phase2_*.py'), ROOT/'advanced_modeling.py', ROOT/'bayesian_network.py',
                 ROOT/'train.py', ROOT/'experiment.py', ROOT/'data_loader.py']:
        shutil.copy2(path, code/path.name)
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn',
                  'torch','pgmpy','matplotlib','scipy','joblib','threadpoolctl','pytest']},
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    write_json(output/'environment.json', environment)
    (output/'requirements-lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in environment['packages'].items())+'\n')
    (output/'working_tree.patch').write_text(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT,text=True))
    raw = pd.read_csv(ROOT/'heart_disease_uci.csv')
    development, retrospective = prepare_partitions(raw)
    selection, cutoff = train_test_split(development, test_size=.25, stratify=(development.num > 0), random_state=43)
    write_json(output/'row_index_to_source_id.json', {str(i): int(v) for i,v in raw.id.items()})
    manifest = dict(status='running', retrospective=True, smoke=smoke,
                    started=datetime.now(timezone.utc).isoformat())
    write_json(output/'manifest.json', manifest)
    try:
        with threadpool_limits(limits=1):
            cv = cross_validate(selection, output/'selection_cv', MODELS, seeds, smoke)
            means = {name: {metric: float(np.mean([r[metric] for r in cv if r['model']==name]))
                for metric in ['f1','roc_auc']} for name in PRIMARY}
            winner = min(PRIMARY, key=lambda name: (-means[name]['f1'], -means[name]['roc_auc'], PRIMARY.index(name)))
            write_json(output/'selection.json', dict(selected_family=winner, scores=means,
                       primary_saved_seed=42, note='No seed or ablation chosen using evaluation results'))
            frozen_evaluation(selection, cutoff, retrospective, output/'pooled', MODELS, seeds, smoke)
            for site in sorted(development.dataset.unique()):
                held = development[development.dataset == site]
                other = development[development.dataset != site]
                local_s, local_c = train_test_split(other, test_size=.25,
                    stratify=(other.num > 0), random_state=43)
                frozen_evaluation(local_s, local_c, held, output/('hospital_'+site.replace(' ','_')),
                                  PRIMARY, [42], smoke)
            cleveland = development[development.dataset == 'Cleveland']
            cross_validate(cleveland, output/'cleveland_cv', PRIMARY, [42], smoke)
        manifest.update(status='complete', finished=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(output/'manifest.json', manifest)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.output, args.smoke)
