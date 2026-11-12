"""Rebuild all Phase 2 tables, charts and paired uncertainty from saved evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, f1_score
from experiment import write_json
from phase2_train import PRIMARY, NEURAL, metrics


def read(path):
    return json.loads(Path(path).read_text())


def summarize(run):
    run=Path(run)
    if read(run/'verification.json')['status']!='PASS' or read(run/'config.json')['smoke']:
        raise ValueError('Summaries require a verified full run')
    plots=run/'figures'
    plots.mkdir(exist_ok=True)
    cv=pd.DataFrame(read(run/'selection_cv/scores.json'))
    columns=['f1','roc_auc','pr_auc_trapezoid','average_precision','recall','specificity','brier','log_loss']
    means=cv.groupby('model')[columns].agg(['mean','std'])
    means.to_csv(run/'cv_summary.csv')
    seed_summary=cv.groupby(['model','seed'])[columns].mean()
    seed_summary.to_csv(run/'cv_by_seed.csv')
    saved=read(run/'pooled/metrics.json')
    eval_rows=[]
    for key,v in saved.items():
        if 'tuned' not in v: continue
        name,seed=key.rsplit('__',1)
        for mode in ['default','tuned']:
            eval_rows.append(dict(model=name,seed=int(seed),operating_point=mode,
                cutoff=.5 if mode=='default' else v['threshold'],
                **{c:v[mode][c] for c in columns},accuracy=v[mode]['accuracy'],ece=v[mode]['ece_5_bins']))
    evaluation=pd.DataFrame(eval_rows)
    evaluation.to_csv(run/'evaluation_summary.csv',index=False)
    hospital=[]
    for folder in sorted(p for p in run.glob('hospital_*') if p.is_dir()):
        for key,v in read(folder/'metrics.json').items():
            if 'tuned' in v:
                hospital.append(dict(hospital=folder.name.removeprefix('hospital_').replace('_',' '),
                    model=key.rsplit('__',1)[0], **{c:v['tuned'][c] for c in columns}))
    pd.DataFrame(hospital).to_csv(run/'hospital_summary.csv',index=False)
    pd.DataFrame(read(run/'cleveland_cv/scores.json')).groupby('model')[columns].mean().to_csv(run/'cleveland_summary.csv')
    predictions=pd.read_csv(run/'pooled/evaluation_predictions.csv')
    paired={}
    # Identical resampled rows for both fixed models; threshold stays C-selected.
    for seed in [42,52,62]:
        a=predictions[predictions.key==f'MLP__{seed}'].set_index('source_id')
        b=predictions[predictions.key==f'BN + MLP__{seed}'].set_index('source_id').loc[a.index]
        y=a.target.to_numpy(); pa=a.probability.to_numpy(); pb=b.probability.to_numpy()
        ta=saved[f'MLP__{seed}']['threshold']; tb=saved[f'BN + MLP__{seed}']['threshold']
        rng=np.random.default_rng(47); draws=rng.integers(0,len(y),(2000,len(y)))
        values={k:[] for k in ['f1','roc_auc','brier']}
        for idx in draws:
            yi=y[idx]; p=pa[idx]; q=pb[idx]
            values['f1'].append(f1_score(yi,q>=tb,zero_division=0)-f1_score(yi,p>=ta,zero_division=0))
            values['brier'].append(np.mean((q-yi)**2)-np.mean((p-yi)**2))
            if len(np.unique(yi))==2:
                values['roc_auc'].append(roc_auc_score(yi,q)-roc_auc_score(yi,p))
        paired[str(seed)]={k:dict(mean=float(np.mean(v)),percentile_95=np.percentile(v,[2.5,97.5]).tolist(),
                                 valid_resamples=len(v)) for k,v in values.items()}
    write_json(run/'paired_hybrid_minus_mlp.json',paired)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in PRIMARY:
        rows=predictions[predictions.key==name+'__42']
        fpr,tpr,_=roc_curve(rows.target,rows.probability)
        axes[0].plot(fpr,tpr,label=name)
        precision,recall,_=precision_recall_curve(rows.target,rows.probability)
        axes[1].plot(recall,precision,label=name)
    axes[0].plot([0,1],[0,1],'k--',lw=.7)
    axes[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='Retrospective ROC, seed 42')
    axes[1].set(xlabel='Recall',ylabel='Precision',title='Retrospective precision–recall, seed 42')
    for ax in axes: ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'discrimination.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in PRIMARY:
        reliability=saved[name+'__42']['default']['reliability']
        nonempty=[b for b in reliability if b['n']]
        axes[0].plot([b['predicted'] for b in nonempty],[b['observed'] for b in nonempty],'o-',label=name)
    axes[0].plot([0,1],[0,1],'k--',lw=.7)
    axes[0].set(xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='Five-bin reliability (small bins)')
    axes[0].legend(fontsize=8)
    names=['MLP','BN + MLP','MLP no dropout','MLP no weight decay','NB + MLP']
    v=seed_summary.loc[names]['f1']
    values=[v.loc[name].to_numpy() for name in names]
    axes[1].barh(names,[np.mean(a) for a in values],xerr=[np.std(a,ddof=1) for a in values],color='#227f87')
    axes[1].set(xlabel='Mean S-fold F1 at 0.5',title='Ablations: mean ± SD across three seeds',xlim=(.75,.9))
    fig.tight_layout();fig.savefig(plots/'calibration_ablations.png',dpi=180);plt.close(fig)
    details=read(run/'pooled/training_details.json')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,name in zip(axes,['MLP','BN + MLP']):
        history=details[name+'__42']['early_stopping_history']
        ax.plot([r['epoch'] for r in history],[r['train_bce'] for r in history],label='Optimization subset')
        ax.plot([r['epoch'] for r in history],[r['validation_bce'] for r in history],label='Early-stop subset')
        ax.axvline(details[name+'__42']['best_epoch'],color='gray',linestyle='--')
        ax.set(xlabel='Epoch',ylabel='Binary cross entropy',title=name+' epoch selection');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(plots/'learning_curves.png',dpi=180);plt.close(fig)
    selected=read(run/'selection.json')['selected_family']
    lines=['# Phase 2 results',f'\nSelected primary family by S-only CV: **{selected}**.',
        '\nFull run; all evaluation results are retrospective. Primary saved seed is 42.',
        '\n## Cross-validation (fixed configurations; common five folds)',
        '\n| Model | Mean F1 | Mean ROC-AUC | Mean PR-AUC | Mean Brier |',
        '|---|---:|---:|---:|---:|']
    for name,row in cv.groupby('model')[columns].mean().iterrows():
        lines.append(f"| {name} | {row.f1:.4f} | {row.roc_auc:.4f} | {row.pr_auc_trapezoid:.4f} | {row.brier:.4f} |")
    lines+=['\n## Retrospective evaluation: seed 42, C-selected cutoffs',
            '\n| Model | Cutoff | F1 | ROC-AUC | PR-AUC | Sensitivity | Specificity | Brier |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in evaluation[(evaluation.seed==42)&(evaluation.operating_point=='tuned')].iterrows():
        lines.append(f'| {r.model} | {r.cutoff:.2f} | {r.f1:.4f} | {r.roc_auc:.4f} | {r.pr_auc_trapezoid:.4f} | {r.recall:.4f} | {r.specificity:.4f} | {r.brier:.4f} |')
    lines+=['\n## Hybrid finding',
        '\nBN + MLP does not outperform the MLP on mean selection-CV F1 or seed-42 retrospective F1/AUC/Brier. '
        'This is a functioning, technically motivated feature-stacking hybrid with a negative empirical result; '
        'the evidence does not establish an additional-marks performance claim. No evaluation-driven retuning was performed.',
        '\nPaired hybrid-minus-MLP intervals (seed 42; fixed models and cutoffs):']
    for k,v in paired['42'].items(): lines.append(f"- {k}: 95% interval [{v['percentile_95'][0]:.4f}, {v['percentile_95'][1]:.4f}].")
    lines+=['\nThese intervals condition on row-level predictions, omit training/selection and patient-cluster uncertainty, and are not formal superiority tests.',
        '\n## Secondary analyses','\nSee hospital_summary.csv and cleveland_summary.csv. Hospital models are independently fitted on other D hospitals; '
        'Cleveland CV uses the Cleveland slice of D only. All use fixed settings and seed 42.',
        '\n## Interpretation limits','\nThe outcome is recorded disease presence, not future risk. The supplied merged CSV lacks verified patient linkage. '
        'The graph is provisional. Five-bin calibration is descriptive, not proof of clinical calibration. '
        'Phase 1 tuned-model scores and Phase 2 fixed-setting control scores have unequal search budgets. '
        'Neither hospital transfer nor prospective clinical utility is established.']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run')
    summarize(parser.parse_args().run)
