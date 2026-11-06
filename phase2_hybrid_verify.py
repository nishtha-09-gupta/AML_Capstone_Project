"""Verify follow-up nested CV, calibration isolation and every saved model."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from data_loader import load_dataset
from advanced_modeling import clean_features
from experiment import write_json
from phase2_train import ROOT,metrics
from phase2_verify import read,require,compare,check_audit
from phase2_improve import FAMILIES,GRIDS
from phase2_hybrids import CalibratedHybrid
from sklearn.model_selection import train_test_split
from train import prepare_partitions,positive_probability


def verify_detail(detail,training_ids,mapping):
    if 'cross_fit_audit' not in detail:
        check_audit(detail,training_ids,mapping);return
    allowed={int(i) for i,j in mapping.items() if j in training_ids}
    require(set(detail['fit_indices'])==allowed,'Stack training row mismatch')
    covered=[]
    for fold in detail['cross_fit_audit']:
        a,b=set(fold['train_indices']),set(fold['validation_indices'])
        require(not a&b and a|b==allowed,'Meta cross-fitting leakage')
        check_audit({'audit':fold['mlp_audit']},{mapping[str(i)] for i in a},mapping)
        covered.extend(b)
    require(len(covered)==len(set(covered)) and set(covered)==allowed,'Meta coverage mismatch')
    check_audit(detail['base_mlp'],training_ids,mapping)


def verify(run):
    run=Path(run);config=read(run/'config.json')
    require(read(run/'manifest.json')['status']=='complete','Study incomplete')
    raw=load_dataset();by_id=raw.set_index('id',drop=False)
    require(hashlib.sha256((ROOT/'heart_disease_uci.csv').read_bytes()).hexdigest()==config['source_sha256'],
            'Source dataset hash mismatch')
    d,t=prepare_partitions(raw)
    s,c=train_test_split(d,test_size=.25,stratify=d.num>0,random_state=43)
    allowed=set(s.id);forbidden=set(c.id)|set(t.id)
    mapping={str(i):int(v) for i,v in raw.id.items()}
    require(set(config['selection_ids'])==allowed and set(config['forbidden_ids'])==forbidden,'Wrong S/C/T scope')
    rows=pd.read_csv(run/'predictions.csv');scores=read(run/'scores.json')
    audits=read(run/'training_details.json');searches=read(run/'searches.json')
    hashes=read(run/'model_sha256.json')
    expected={(n,seed,f) for n in FAMILIES for seed in config['seeds'] for f in range(len(config['outer_folds']))}
    identity=lambda r:(r['model'],r['seed'],r['fold'])
    require({identity(r) for r in scores}==expected and len(scores)==len(expected),'Missing/duplicate scores')
    require({identity(r) for r in audits}==expected and len(audits)==len(expected),'Missing/duplicate audits')
    require({identity(r) for r in searches}==expected and len(searches)==len(expected),'Missing/duplicate searches')
    audits={identity(r):r for r in audits};searches={identity(r):r for r in searches}
    count=0
    for score in scores:
        name,seed,fold=identity(score)
        plan=config['outer_folds'][fold];a,b=set(plan['train_ids']),set(plan['validation_ids'])
        require(not a&b and a|b==allowed and not(a|b)&forbidden,'Outer split leakage')
        predicted=rows[(rows.model==name)&(rows.seed==seed)&(rows.fold==fold)]
        require(predicted.source_id.is_unique and set(predicted.source_id)==b,'Wrong outer rows')
        frame=by_id.loc[predicted.source_id]
        require(np.array_equal((frame.num>0).astype(int),predicted.target),'Wrong outer targets')
        compare(metrics(predicted.target,predicted.probability.to_numpy()),score)
        search=searches[(name,seed,fold)]
        for inner in search['folds']:
            ia,ib=set(inner['train_ids']),set(inner['validation_ids'])
            require(not ia&ib and ia|ib==a,'Inner CV leakage')
        candidates=search['candidates']
        require([r['parameter'] for r in candidates]==GRIDS[name],'Search grid mismatch')
        for candidate in candidates:
            require(np.isclose(candidate['mean_brier'],np.mean([v['brier'] for v in candidate['fold_scores']]),atol=1e-12),
                    'Inner Brier mean mismatch')
            require(np.isclose(candidate['mean_auc'],np.mean([v['roc_auc'] for v in candidate['fold_scores']]),atol=1e-12),
                    'Inner AUC mean mismatch')
        best=min(range(len(candidates)),key=lambda i:(candidates[i]['mean_brier'],-candidates[i]['mean_auc'],i))
        require(best==search['selected_index'] and candidates[best]['parameter']==score['parameter'],'Inner selection mismatch')
        verify_detail(audits[(name,seed,fold)]['details'],a,mapping)
        key=f'{name.replace(" ","_")}__{seed}__{fold}'
        file=run/'models'/(key+'.joblib')
        require(hashlib.sha256(file.read_bytes()).hexdigest()==hashes[key],'Follow-up model hash mismatch')
        model=joblib.load(file);p=positive_probability(model,clean_features(frame))
        require(np.allclose(p,predicted.probability,atol=1e-7,rtol=0),'Follow-up probability roundtrip mismatch')
        count+=1
    for _,part in rows.groupby(['model','seed']):
        require(part.source_id.is_unique and set(part.source_id)==allowed,'Incomplete outer OOF predictions')
    result=dict(status='PASS',models_checked=count,smoke=config['smoke'],
        source_rows=len(allowed),forbidden_rows_excluded=len(forbidden),
        checks=['Every model and seed has complete common outer-fold coverage',
                'Inner search stays within outer training rows',
                'Meta cross-fitting and early stopping exclude validation rows',
                'C and T never enter training or scoring',
                'Model hashes, round trips, targets, metrics and grid selections reconstruct'],
        scope='Exploratory follow-up after original results; not untouched confirmation')
    write_json(run/'verification.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run')
    print(json.dumps(verify(parser.parse_args().run),indent=2))
