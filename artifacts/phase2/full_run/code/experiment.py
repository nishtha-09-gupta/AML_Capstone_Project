"""Save one explicitly authorized exploratory run; reuse train.py's validation.

No experiment runs on import. All writes are under one exclusively created run
folder. The source protocol and historical artifacts are never changed.
"""

import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split
from threadpoolctl import threadpool_info, threadpool_limits

from train import (
    PROTOCOL, ROOT, develop, evaluate_frozen, make_candidates,
    positive_probability, prepare_partitions, scores, stratified_cv,
)
from advanced_modeling import clean_features, deduplicate_records


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n")


def partition_plan(data, seeds, protocol):
    """Record the exact S/C and nested fold memberships before fitting."""
    selection, cutoff = train_test_split(data, test_size=0.25, stratify=(data['num'] > 0).astype(int), random_state=seeds['split'])
    y = (selection['num'] > 0).astype(int)

    def folds(frame, labels, count, seed):
        return [{"train_source_ids": frame.iloc[a]['id'].tolist(), "validation_source_ids": frame.iloc[b]['id'].tolist()}
                for a, b in stratified_cv(labels, count, seed).split(frame, labels)]

    outer = folds(selection, y, protocol['selection']['outer_cv']['n_splits'], seeds['outer'])
    for fold in outer:
        positions = {value: i for i, value in enumerate(selection['id'])}
        train = selection.iloc[[positions[value] for value in fold['train_source_ids']]]
        fold['inner'] = folds(train, (train['num'] > 0).astype(int), 3, seeds['inner'])
    return {"seeds": seeds, "selection_source_ids": selection['id'].tolist(), "cutoff_source_ids": cutoff['id'].tolist(),
            "outer": outer, "final_search": folds(selection, y, 3, seeds['final_search'])}


def bootstrap_intervals(y, p, threshold, seed=47, repetitions=2000):
    """Paired row bootstrap, vectorized on fixed predictions; never refit/retune."""
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    n = len(y)
    draws = np.random.default_rng(seed).integers(0, n, size=(repetitions, n))
    counts = np.zeros((repetitions, n), dtype=int)
    np.add.at(counts, (np.repeat(np.arange(repetitions), n), draws.ravel()), 1)
    positive, negative = counts @ y, counts @ (1 - y)
    predicted = p >= threshold
    tp, fp = counts @ (y * predicted), counts @ ((1 - y) * predicted)
    fn, tn = positive - tp, negative - fp

    def divide(a, b, missing=False):
        return np.divide(a, b, out=np.full(repetitions, np.nan if missing else 0.0), where=b != 0)

    order = np.argsort(p)
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    pos = np.add.reduceat(counts[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(counts[:, order] * (1 - y[order]), starts, axis=1)
    auc = divide((pos * (np.cumsum(neg, axis=1) - neg + 0.5 * neg)).sum(axis=1), positive * negative, True)
    descending_pos, descending_total = pos[:, ::-1], (pos + neg)[:, ::-1]
    precision_at_group = np.divide(np.cumsum(descending_pos, axis=1), np.cumsum(descending_total, axis=1),
                                   out=np.zeros_like(descending_pos, dtype=float), where=np.cumsum(descending_total, axis=1) != 0)
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    values = {
        'accuracy': (tp + tn) / n, 'precision': divide(tp, tp + fp), 'recall': divide(tp, positive),
        'specificity': divide(tn, negative, True), 'f1': divide(2 * tp, 2 * tp + fp + fn),
        'roc_auc': auc, 'average_precision': divide((precision_at_group * descending_pos).sum(axis=1), positive),
        'brier': counts @ ((p - y) ** 2) / n,
        'log_loss': counts @ (-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))) / n,
    }
    return {name: {'valid_resamples': int(np.isfinite(v).sum()), 'invalid_resamples': int((~np.isfinite(v)).sum()),
                   'percentile_95': np.percentile(v[np.isfinite(v)], [2.5, 97.5]).tolist() if np.isfinite(v).sum() >= 1000 else None}
            for name, v in values.items()}


def save_development(folder, result, plan):
    assert result['selection_source_ids'] == plan['selection_source_ids']
    assert result['cutoff_source_ids'] == plan['cutoff_source_ids']
    write_json(folder / 'searches.json', result['searches'])
    pd.DataFrame(result['oof']).to_csv(folder / 'nested_oof_predictions.csv', index=False)
    cutoff_rows = []
    for name, probabilities in result['cutoff_probabilities'].items():
        cutoff_rows.extend({'model': name, 'source_id': row, 'target': target, 'probability': p}
                           for row, target, p in zip(result['cutoff_source_ids'], result['cutoff_targets'], probabilities))
    pd.DataFrame(cutoff_rows).to_csv(folder / 'cutoff_predictions.csv', index=False)
    write_json(folder / 'thresholds.json', result['thresholds'])
    write_json(folder / 'comparison.json', result['comparison'])
    model_dir = folder / 'models'
    model_dir.mkdir()
    hashes = {}
    parameters = {}
    for name, model in result['models'].items():
        file = model_dir / (name.replace(' ', '_') + '.joblib')
        joblib.dump(model, file)
        hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        parameters[name] = {key: model.get_params()[key] for key in next(s['results']['params'][0] for s in result['searches'] if s['model'] == name)}
        if name == 'Bayesian network':
            write_json(folder / 'bn_state.json', {'graph': model.graph_, 'states': model.states_, 'bins': model.bin_edges_,
                                                 'training_fill_values': model.fill_values_, 'cpts': model.cpds_, 'alpha': model.alpha})
    # Persist model identities and decisions before evaluation is allowed to start.
    write_json(folder / 'freeze.json', {'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
               'selected_model': result['selected_model'], 'best_parameters': parameters,
               'model_sha256': hashes, 'thresholds': {k: v['threshold'] for k, v in result['thresholds'].items()}})


def save_evaluation(folder, data, result, evaluation, selection):
    rows, metrics = [], {}
    y = np.asarray(evaluation['target'])
    for name, family in evaluation['families'].items():
        p = np.asarray(family['probability'])
        threshold = result['thresholds'][name]['threshold']
        rows.extend({'model': name, 'source_id': row, 'target': int(target), 'probability': float(prob),
                     'default_prediction': int(prob >= 0.5), 'threshold': threshold, 'tuned_prediction': int(prob >= threshold)}
                    for row, target, prob in zip(evaluation['source_ids'], y, p))
        bins = np.minimum((p * 5).astype(int), 4)
        reliability = [{'lower': k / 5, 'upper': (k + 1) / 5, 'n': int((bins == k).sum()),
                        'mean_probability': float(p[bins == k].mean()) if (bins == k).any() else None,
                        'positive_fraction': float(y[bins == k].mean()) if (bins == k).any() else None} for k in range(5)]
        subgroups = {}
        for variable, labels in {'sex': data['sex'], 'age': pd.cut(data['age'], [0, 40, 50, 60, 70, 120], right=False,
                                                                  labels=['under40', '40-49', '50-59', '60-69', '70plus'])}.items():
            subgroups[variable] = {}
            for group in labels.dropna().unique():
                mask = (labels == group).to_numpy()
                subgroups[variable][str(group)] = {'n': int(mask.sum()), 'positive': int(y[mask].sum()),
                    'tuned_metrics': scores(y[mask], p[mask], threshold) if len(np.unique(y[mask])) == 2 else None}
        metrics[name] = {'threshold': threshold, 'default': family['default'], 'tuned': family['tuned'],
                         'default_intervals': bootstrap_intervals(y, p, 0.5), 'tuned_intervals': bootstrap_intervals(y, p, threshold),
                         'reliability_bins': reliability, 'descriptive_subgroups': subgroups,
                         'search_refit_seconds': sum(s['elapsed_seconds'] for s in result['searches'] if s['model'] == name)}
    prior = float((selection['num'] > 0).mean())
    write_json(folder / 'metrics.json', {'label': evaluation['label'], 'n': len(y), 'positive': int(y.sum()),
               'selected_model': result['selected_model'], 'families': metrics,
               'constant_reference': {'probability_from_S': prior, 'always_positive_metrics': scores(y, np.full(len(y), prior), 0.0)},
               'interval_method': '2000 paired row-bootstrap resamples, seed 47; fixed models/cutoffs; percentile 95%; conditional uncertainty only'})
    pd.DataFrame(rows).to_csv(folder / 'evaluation_predictions.csv', index=False)
    return metrics


def run_experiment(*, allow_provisional=False):
    if not allow_provisional:
        raise ValueError('Exploratory provisional BN execution requires explicit opt-in')
    started = time.perf_counter()
    run = ROOT / 'artifacts/runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(), 'failures': [],
                'run_directory': str(run), 'mode': 'user-authorized exploratory provisional BN',
                'limitations': ['BN graph is provisional, not faculty approved and not causal evidence',
                               'BN alpha=1 only; unequal search budgets', 'Historical holdout evaluation is retrospective',
                               'Faculty questions and merged-dataset provenance remain unresolved',
                               'Bootstrap intervals condition on fitted models and do not resolve patient linkage or selection uncertainty'],
                'initial_environment_issue': 'See environment.json for the runtime and thread settings used in this run'}
    def progress(message):
        line = datetime.now(timezone.utc).isoformat() + ' ' + message
        with (run / 'execution.log').open('a') as log:
            log.write(line + '\n')
        print(line, flush=True)
    progress('RUN_DIRECTORY=' + str(run))
    write_json(run / 'manifest.json', manifest)
    captured = []
    try:
        protocol = copy.deepcopy(PROTOCOL)
        protocol['execution_authorized'] = True
        protocol['status'] = 'exploratory_stage4_user_authorized_not_faculty_approved'
        protocol['models']['grids']['Bayesian network'] = {'alpha': [1.0]}
        protocol['models']['bn_grid_status'] = 'Existing documented provisional alpha=1 default; single-setting exploratory run authorized before results'
        protocol['evaluation']['subgroups'] = 'Descriptive sex groups and fixed age bins [0,40,50,60,70,120), right=False; no subgroup optimization'
        write_json(run / 'effective_protocol.json', protocol)
        shutil.copy2(ROOT / 'config/bn_graph.json', run / 'bn_graph.json')
        code = run / 'code'
        code.mkdir()
        source_files = ['train.py', 'experiment.py', 'advanced_modeling.py', 'modeling.py', 'data_loader.py', 'bayesian_network.py', 'requirements.txt',
                        'config/phase1_protocol.json', 'config/bn_graph.json', 'tests/test_modeling.py',
                        'tests/test_validation.py', 'tests/test_bayesian_network.py']
        for name in source_files:
            target = code / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        manifest['code_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files}
        manifest['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        manifest['git_status'] = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
        (run / 'working_tree.patch').write_text(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True))
        (run / 'requirements-freeze.txt').write_text(subprocess.check_output([sys.executable, '-B', '-m', 'pip', '--isolated', 'freeze'], text=True))
        write_json(run / 'environment.json', {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                   'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib', 'scipy', 'xgboost', 'threadpoolctl']},
                   'DYLD_LIBRARY_PATH': os.environ.get('DYLD_LIBRARY_PATH'), 'worker_limit': 1})
        source = ROOT / protocol['data_contract']['source']
        manifest['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
        assert manifest['source_sha256'] == protocol['data_contract']['expected_sha256']
        raw = pd.read_csv(source)
        development, retrospective = prepare_partitions(raw)
        assert (len(raw), len(development), len(retrospective)) == (920, 734, 184)
        assert (int((development['num'] > 0).sum()), int((retrospective['num'] > 0).sum())) == (406, 102)
        retained = deduplicate_records(raw)
        groups = raw.groupby(list(raw.columns.difference(['id'])), dropna=False, sort=False)['id'].agg(list)
        write_json(run / 'duplicate_lineage.json', {'source_rows': len(raw), 'retained_rows': len(retained),
                   'removed_source_ids': raw.loc[~raw.index.isin(retained.index), 'id'].tolist(),
                   'duplicate_groups_keep_first': [ids for ids in groups if len(ids) > 1]})
        membership = retained[['id', 'dataset', 'num']].copy()
        membership['partition'] = np.where(membership['id'].isin(retrospective['id']), 'T_retrospective', 'D_development')
        membership.to_csv(run / 'historical_split.csv', index_label='source_index')
        seeds = {'split': 43, 'outer': 44, 'inner': 45, 'final_search': 46}
        plans = {'pooled': partition_plan(development, seeds, protocol)}
        for site in protocol['hospital_evaluation']['site_order']:
            other = development.loc[development['dataset'] != site]
            plans[site] = partition_plan(other, protocol['hospital_evaluation']['seed_map'][site], protocol)
        for name, plan in plans.items():
            folder = run / ('pooled' if name == 'pooled' else 'hospital_' + name.replace(' ', '_'))
            folder.mkdir()
            write_json(folder / 'partitions.json', plan)
        candidates = make_candidates(allow_provisional=True, protocol=protocol)
        configurations = {name: len(ParameterGrid(grid)) for name, (_, grid) in candidates.items()}
        manifest['configuration_counts'] = configurations
        manifest['planned_fits_per_experiment'] = sum(18 * n + 6 for n in configurations.values())
        manifest['planned_experiments'] = ['pooled'] + protocol['hospital_evaluation']['site_order']
        write_json(run / 'manifest.json', manifest)
        progress(f"Five families; {manifest['planned_fits_per_experiment']} fits per experiment; all partitions saved before fitting")
        with warnings.catch_warnings(record=True) as captured, threadpool_limits(limits=1):
            write_json(run / 'threadpools.json', threadpool_info())
            pooled = develop(development, candidates, forbidden_ids=retrospective['id'], protocol=protocol,
                             progress=lambda message: progress('pooled: ' + message))
            save_development(run / 'pooled', pooled, plans['pooled'])
            summary = {'pooled_selected_model': pooled['selected_model'], 'hospitals': {}}
            # Each site independently selects its family/parameters/cutoff; no pooled winner is imported.
            for site in protocol['hospital_evaluation']['site_order']:
                held = development.loc[development['dataset'] == site]
                other = development.loc[development['dataset'] != site]
                folder = run / ('hospital_' + site.replace(' ', '_'))
                local = develop(other, candidates, forbidden_ids=retrospective['id'].tolist() + held['id'].tolist(),
                                seeds=protocol['hospital_evaluation']['seed_map'][site], protocol=protocol,
                                progress=lambda message, site=site: progress(site + ': ' + message))
                save_development(folder, local, plans[site])
                evaluation = evaluate_frozen(held, local, label='D-only held-out hospital: ' + site)
                selection = other.loc[other['id'].isin(local['selection_source_ids'])]
                metrics = save_evaluation(folder, held, local, evaluation, selection)
                summary['hospitals'][site] = {'n': len(held), 'selected_model': local['selected_model'],
                                             'families': {name: {'threshold': m['threshold'], 'tuned': m['tuned']} for name, m in metrics.items()}}
                progress(site + ': independent evaluation and outputs complete')
            # This is the first T scoring call; all study settings/models/cutoffs are already frozen.
            evaluation = evaluate_frozen(retrospective, pooled)
            selection = development.loc[development['id'].isin(pooled['selection_source_ids'])]
            metrics = save_evaluation(run / 'pooled', retrospective, pooled, evaluation, selection)
            summary['retrospective'] = {'n': len(retrospective), 'label': evaluation['label'],
                                       'families': {name: {'threshold': m['threshold'], 'tuned': m['tuned'],
                                                           'search_refit_seconds': m['search_refit_seconds']} for name, m in metrics.items()}}
            write_json(run / 'summary.json', summary)
        manifest['status'] = 'complete'
        progress('All five families completed pooled and four independent hospital experiments')
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['failures'].append({'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        progress('FAILED: ' + type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter() - started
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        manifest['warnings'] = sorted(set(str(w.message) for w in captured))
        write_json(run / 'manifest.json', manifest)
    return run
