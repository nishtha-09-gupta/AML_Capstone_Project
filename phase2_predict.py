"""Batch inference from a verified run, with no fitting or threshold changes."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from advanced_modeling import clean_features, validate_features


def predict(run, rows, key='BN + MLP__42'):
    run=Path(run)
    verification=json.loads((run/'verification.json').read_text())
    if verification['status']!='PASS' or verification['smoke']:
        raise ValueError('Use a verified full experiment, not a smoke run')
    errors=validate_features(rows)
    if errors:
        raise ValueError('; '.join(errors))
    freeze=json.loads((run/'pooled/freeze.json').read_text())['models']
    if key not in freeze:
        raise ValueError('Unknown saved model')
    spec=freeze[key]
    path=run/'pooled'/spec['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
        raise ValueError('Model failed integrity check')
    model=joblib.load(path)
    p=model.predict_proba(clean_features(rows))[:,1]
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probabilities')
    return pd.DataFrame(dict(probability=p,cutoff=spec['threshold'],
                             predicted_presence=(p>=spec['threshold']).astype(int)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run')
    parser.add_argument('input_csv')
    parser.add_argument('--model',default='BN + MLP__42')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    path=Path(args.output)
    if path.exists():
        raise FileExistsError('Choose a new output path')
    predict(args.run,pd.read_csv(args.input_csv),args.model).to_csv(path,index=False)
