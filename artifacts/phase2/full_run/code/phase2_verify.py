"""Reconstruct metrics, cutoffs, model predictions and split isolation for a run."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from advanced_modeling import clean_features
from experiment import write_json
from phase2_train import ROOT, PRIMARY, metrics
from phase2_baselines import DiscreteHeartModel
from train import choose_threshold, prepare_partitions, positive_probability


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare(actual, expected):
    for key in ['accuracy','precision','recall','specificity','f1','roc_auc',
                'average_precision','pr_auc_trapezoid','brier','log_loss','ece_5_bins']:
        require(np.isclose(actual[key], expected[key], atol=1e-10, rtol=0), f'Metric mismatch: {key}')
    require(actual['confusion_matrix'] == expected['confusion_matrix'], 'Confusion matrix mismatch')


def check_audit(detail, expected_ids, mapping):
    if 'audit' not in detail:
        return
    audit = detail['audit']
    source_ids = {mapping[str(i)] for i in audit['fit_indices']}
    require(source_ids == set(expected_ids), 'Neural fit used unexpected rows')
    train, valid = set(audit['early_train_indices']), set(audit['early_validation_indices'])
    require(not train & valid and train | valid == set(audit['fit_indices']), 'Early-stop leakage')
    for key, allowed in [('early_bn_folds', train), ('final_bn_folds', train | valid)]:
        covered = []
        for fold in audit[key]:
            a,b = set(fold['train_indices']), set(fold['validation_indices'])
            require(not a & b and a | b == allowed, 'BN cross-fit leakage')
            covered.extend(b)
        if audit[key]:
            require(len(covered)==len(set(covered)) and set(covered)==allowed, 'Invalid BN fold coverage')


def verify(run):
    run = Path(run)
    require(read(run/'manifest.json')['status']=='complete', 'Run did not finish')
    config = read(run/'config.json')
    require(hashlib.sha256((ROOT/'heart_disease_uci.csv').read_bytes()).hexdigest()
            == config['source_sha256'], 'Source dataset changed')
    raw = pd.read_csv(ROOT/'heart_disease_uci.csv')
    by_id = raw.set_index('id', drop=False)
    development, retrospective = prepare_partitions(raw)
    mapping = read(run/'row_index_to_source_id.json')
    checks, count = [], 0
    for folder in [run/'selection_cv', run/'cleveland_cv']:
        folds = read(folder/'folds.json')
        predicted = pd.read_csv(folder/'predictions.csv')
        summaries = read(folder/'scores.json')
        audits = read(folder/'training_details.json')
        allowed = set(read(run/'pooled/partitions.json')['selection']) if folder.name=='selection_cv' else set(
            development.loc[development.dataset=='Cleveland','id'])
        for fold in folds:
            a,b = set(fold['train_ids']),set(fold['validation_ids'])
            require(not a & b and a | b==allowed, 'CV partition leakage')
        for (_, _), rows in predicted.groupby(['model','seed']):
            require(rows.source_id.is_unique and set(rows.source_id)==allowed, 'OOF coverage mismatch')
        for record, audit in zip(summaries, audits):
            rows = predicted[(predicted.model==record['model']) & (predicted.seed==record['seed'])
                             & (predicted.fold==record['fold'])]
            require(set(rows.source_id)==set(folds[record['fold']]['validation_ids']), 'Wrong CV fold')
            compare(metrics(rows.target,rows.probability.to_numpy()),record)
            check_audit(audit['details'],folds[record['fold']]['train_ids'],mapping)
        checks.append(folder.name+': common folds, OOF scores and training isolation')
    selection = read(run/'selection.json')
    cv_scores = read(run/'selection_cv/scores.json')
    means = {n: {k: np.mean([r[k] for r in cv_scores if r['model']==n]) for k in ['f1','roc_auc']} for n in PRIMARY}
    winner = min(PRIMARY,key=lambda n:(-means[n]['f1'],-means[n]['roc_auc'],PRIMARY.index(n)))
    require(winner==selection['selected_family'], 'Family selection mismatch')
    for folder in [run/'pooled',*sorted(run.glob('hospital_*'))]:
        partitions = read(folder/'partitions.json')
        s,c,t = [set(partitions[k]) for k in ['selection','cutoff','evaluation']]
        require(not (s & c or s & t or c & t), 'Evaluation partition leakage')
        if folder.name=='pooled':
            require(t==set(retrospective.id) and s|c==set(development.id), 'Historical split mismatch')
        else:
            require((s|c|t)==set(development.id), 'Hospital experiment must use D only')
            site = folder.name.removeprefix('hospital_').replace('_',' ')
            require(set(by_id.loc[list(t)].dataset)=={site}, 'Wrong hospital holdout')
            require(site not in set(by_id.loc[list(s|c)].dataset), 'Hospital leaked into fitting')
        predicted = pd.read_csv(folder/'evaluation_predictions.csv')
        cutoff = pd.read_csv(folder/'cutoff_predictions.csv')
        freeze = read(folder/'freeze.json')['models']
        saved = read(folder/'metrics.json')
        details = read(folder/'training_details.json')
        for key, spec in freeze.items():
            file = folder/spec['file']
            require(hashlib.sha256(file.read_bytes()).hexdigest()==spec['sha256'], 'Model hash mismatch')
            model = joblib.load(file)
            rows = predicted[predicted.key==key]
            cr = cutoff[cutoff.key==key]
            require(rows.source_id.is_unique and set(rows.source_id)==t,'Wrong evaluation rows')
            require(cr.source_id.is_unique and set(cr.source_id)==c,'Wrong cutoff rows')
            for subset in [rows,cr]:
                frame = by_id.loc[subset.source_id].copy()
                require(np.array_equal((frame.num>0).astype(int),subset.target),'Target mismatch')
                p = positive_probability(model,clean_features(frame))
                require(np.allclose(p,subset.probability,atol=1e-7,rtol=0),'Saved-model prediction mismatch')
            decision = choose_threshold(cr.target,cr.probability.to_numpy())
            require(decision['threshold']==spec['threshold'],'Threshold mismatch')
            compare(metrics(rows.target,rows.probability.to_numpy()),saved[key]['default'])
            compare(metrics(rows.target,rows.probability.to_numpy(),spec['threshold']),saved[key]['tuned'])
            check_audit(details[key],s,mapping)
            if isinstance(model, DiscreteHeartModel):
                query=clean_features(by_id.loc[rows.source_id.iloc[:5]])
                require(np.allclose(model.predict_proba(query),model.predict_pgmpy(query),atol=1e-12),'pgmpy mismatch')
            count += 1
        checks.append(folder.name+': hashes, round trips, cutoffs, metrics and isolation')
    result=dict(status='PASS',saved_models_checked=count,checks=checks,smoke=config['smoke'])
    write_json(run/'verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    print(json.dumps(verify(parser.parse_args().run),indent=2))
