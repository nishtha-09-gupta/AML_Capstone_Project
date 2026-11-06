"""Fit follow-up inference models on S after the finite nested-CV study."""
import argparse
import hashlib
import json
import logging
from datetime import datetime,timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from data_loader import load_dataset
from advanced_modeling import clean_features,validate_features
from experiment import write_json
from phase2_improve import TrainingCache,select,detail
from phase2_train import ROOT
from phase2_verify import read,require
from phase2_hybrid_verify import verify_detail
from train import prepare_partitions,positive_probability


def freeze(run):
    logging.getLogger('pgmpy').setLevel(logging.ERROR)
    run=Path(run);finding=read(run/'finding.json');config=read(run/'config.json')
    require(read(run/'verification.json')['status']=='PASS' and not config['smoke'],'Verify full follow-up first')
    output=run/'final_models';output.mkdir(exist_ok=False)
    raw=load_dataset();d,t=prepare_partitions(raw)
    s,c=train_test_split(d,test_size=.25,stratify=d.num>0,random_state=43)
    x,y=clean_features(s),(s.num>0).astype(int)
    require(set(s.id)==set(config['selection_ids']),'Unexpected training IDs')
    mapping={str(i):int(v) for i,v in raw.id.items()}
    names=list(dict.fromkeys([finding['best_hybrid'],finding['strongest_single_base_control'],
                             'Tuned MLP','Calibrated MLP']))
    decisions={};searches=[];audit={};training_rows=[]
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=1):
        for seed in config['seeds']:
            cache=TrainingCache(seed)
            for name in names:
                value,search=select(name,x,y,s.id,cache)
                model=cache.fit(name,value,x,y)
                key=f'{name.replace(" ","_")}__{seed}';file=output/(key+'.joblib')
                joblib.dump(model,file)
                probability=positive_probability(model,x)
                restored=positive_probability(joblib.load(file),x)
                require(np.allclose(probability,restored,atol=1e-7,rtol=0),'Final model roundtrip mismatch')
                audit[key]=detail(model);verify_detail(audit[key],set(s.id),mapping)
                decisions[key]=dict(model=name,seed=seed,parameter=value,file=file.name,
                    sha256=hashlib.sha256(file.read_bytes()).hexdigest(),decision_cutoff=.5)
                searches.append(dict(model=name,seed=seed,**search))
                training_rows.extend(dict(key=key,source_id=int(i),probability=float(v)) for i,v in zip(s.id,probability))
            print(f'Final S-only fits verified for seed {seed}',flush=True)
    write_json(output/'freeze.json',dict(frozen_at=datetime.now(timezone.utc).isoformat(),
        selection_ids=s.id.tolist(),forbidden_ids=c.id.tolist()+t.id.tolist(),models=decisions,
        evaluation='No C/T scoring; recorded predictions are training roundtrip evidence, not performance estimates'))
    write_json(output/'searches.json',searches);write_json(output/'training_details.json',audit)
    pd.DataFrame(training_rows).to_csv(output/'training_roundtrip.csv',index=False)
    return dict(status='PASS',final_models_checked=len(decisions),training_rows=len(s),C_T_evaluated=False)


def predict(run,rows,key='Calibrated_BN-MLP__42'):
    run=Path(run);require(read(run/'verification.json')['status']=='PASS','Study must pass verification')
    errors=validate_features(rows)
    if errors:raise ValueError('; '.join(errors))
    record=read(run/'final_models/freeze.json')['models']
    require(key in record,'Choose a model listed in final_models/freeze.json')
    spec=record[key];file=run/'final_models'/spec['file']
    require(hashlib.sha256(file.read_bytes()).hexdigest()==spec['sha256'],'Model integrity failure')
    p=positive_probability(joblib.load(file),clean_features(rows))
    return pd.DataFrame(dict(probability=p,cutoff=.5,predicted_presence=(p>=.5).astype(int)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run')
    parser.add_argument('--input-csv');parser.add_argument('--output');parser.add_argument('--model',default='Calibrated_BN-MLP__42')
    args=parser.parse_args()
    if args.input_csv:
        if not args.output:parser.error('--output is required with --input-csv')
        if Path(args.output).exists():raise FileExistsError('Choose a new output path')
        predict(args.run,pd.read_csv(args.input_csv),args.model).to_csv(args.output,index=False)
    else:print(json.dumps(freeze(args.run),indent=2))
