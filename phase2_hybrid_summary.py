"""Summarize the finite S-only hybrid study without evaluating historical C/T."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import write_json
from phase2_verify import read,require


def summarize(run):
    run=Path(run)
    verified=read(run/'verification.json')
    require(verified['status']=='PASS' and not verified['smoke'],'Use a verified full follow-up')
    scores=pd.DataFrame(read(run/'scores.json'))
    columns=['brier','roc_auc','pr_auc_trapezoid','log_loss','f1','recall','specificity','ece_5_bins']
    means=scores.groupby('model')[columns].mean().sort_values('brier')
    means.to_csv(run/'summary.csv')
    by_seed=scores.groupby(['model','seed'])[columns].mean()
    by_seed.to_csv(run/'by_seed.csv')
    predictions=pd.read_csv(run/'predictions.csv')
    hypotheses=['Calibrated BN-MLP','BN residual MLP']
    controls=['BN','Tuned MLP','Calibrated BN','Calibrated MLP','Original BN feature MLP']
    paired={}
    rng=np.random.default_rng(147)
    ids=sorted(predictions.source_id.unique())
    draws=rng.integers(0,len(ids),size=(2000,len(ids)))
    for hybrid in hypotheses:
        for control in controls:
            loss_deltas=[];seed_deltas={};fold_deltas=[]
            for seed in sorted(predictions.seed.unique()):
                a=predictions[(predictions.model==hybrid)&(predictions.seed==seed)].set_index('source_id').loc[ids]
                b=predictions[(predictions.model==control)&(predictions.seed==seed)].set_index('source_id').loc[ids]
                delta=(a.probability.to_numpy()-a.target.to_numpy())**2-(b.probability.to_numpy()-b.target.to_numpy())**2
                loss_deltas.append(delta);seed_deltas[str(seed)]=float(delta.mean())
                for fold in sorted(a.fold.unique()):fold_deltas.append(float(delta[a.fold.to_numpy()==fold].mean()))
            # Resample source rows, retaining all seed predictions together.
            per_row=np.mean(loss_deltas,axis=0)
            boot=per_row[draws].mean(axis=1)
            paired[hybrid+' minus '+control]=dict(mean_brier_difference=float(per_row.mean()),
                percentile_95=np.percentile(boot,[2.5,97.5]).tolist(),seed_differences=seed_deltas,
                winning_seed_fold_pairs=int(np.sum(np.asarray(fold_deltas)<0)),total_seed_fold_pairs=len(fold_deltas),
                method='2000 paired source-row bootstrap resamples retaining all seeds, fixed OOF predictions',
                limitations='Conditional exploratory intervals; overlapping CV training, training variability and patient clusters not represented; not confirmatory tests')
    write_json(run/'paired_comparisons.json',paired)
    best_hybrid=min(hypotheses,key=lambda n:means.loc[n,'brier'])
    strongest_control=min([n for n in controls if n!='Original BN feature MLP'],key=lambda n:means.loc[n,'brier'])
    delta=float(means.loc[best_hybrid,'brier']-means.loc[strongest_control,'brier'])
    result=dict(best_hybrid=best_hybrid,strongest_single_base_control=strongest_control,
        hybrid_brier=float(means.loc[best_hybrid,'brier']),control_brier=float(means.loc[strongest_control,'brier']),
        brier_difference=delta,relative_brier_reduction_percent=float(-100*delta/means.loc[strongest_control,'brier']),
        lower_brier_than_every_control=bool(delta<0),
        every_seed_improves_on_strongest_control=all(v<0 for v in paired[best_hybrid+' minus '+strongest_control]['seed_differences'].values()),
        evidence_scope='Exploratory nested CV on S only, not new external or untouched validation',
        bonus_marks_guaranteed=False)
    write_json(run/'finding.json',result)
    figures=run/'figures';figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9,4.6))
    order=list(means.index)
    ax.barh(order,means.loc[order,'brier'],color=['#227f87' if n in hypotheses else '#8a9ca6' for n in order])
    ax.set(xlabel='Nested outer Brier score (lower is better)',title='S-only exploratory comparison; all three seeds')
    ax.invert_yaxis();ax.set_xlim(0,max(means.brier)*1.12)
    for i,n in enumerate(order):ax.text(means.loc[n,'brier']+.001,i,f"{means.loc[n,'brier']:.4f}",va='center',fontsize=9)
    fig.tight_layout();fig.savefig(figures/'nested_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name in [best_hybrid,strongest_control,'Tuned MLP']:
        vals=by_seed.loc[name,'brier']
        axes[0].plot(vals.index,vals.values,'o-',label=name)
    axes[0].set(xlabel='Neural training seed',ylabel='Mean nested Brier',title='Seed consistency');axes[0].legend(fontsize=8)
    coefficients=[]
    for detail in read(run/'training_details.json'):
        if detail['model']=='Calibrated BN-MLP':
            coefficients.append(dict(seed=detail['seed'],fold=detail['fold'],
                bn_coefficient=detail['details']['coefficients'][0][0],
                mlp_coefficient=detail['details']['coefficients'][0][1],
                intercept=detail['details']['intercept'][0],C=detail['details']['C']))
    coef=pd.DataFrame(coefficients);coef.to_csv(run/'stack_coefficients.csv',index=False)
    axes[1].boxplot([coef.bn_coefficient,coef.mlp_coefficient],tick_labels=['BN logit','MLP logit'])
    axes[1].set(ylabel='Learned logistic coefficient',title='Inspecting fusion across outer fits')
    fig.tight_layout();fig.savefig(figures/'consistency_coefficients.png',dpi=180);plt.close(fig)
    lines=['# Hybrid improvement study: exploratory follow-up',
        '\nOriginal Phase 2 results are preserved. This follow-up uses S only (550 records); C and T (368 records) are excluded from fitting and scoring.',
        '\n## Mechanism and comparison',
        '\nBN residual MLP starts at BN log-odds and learns continuous corrections. Calibrated BN–MLP learns a penalized logistic combination of fresh, cross-fitted BN and MLP logits. Calibrated BN and calibrated MLP controls distinguish fusion value from calibration alone.',
        '\nFive outer stratified folds, three inner tuning folds, and all three seeds are retained. Inner Brier chooses each configuration. The primary outcome is Brier; this is a documented follow-up criterion, not a replacement for the original F1-first experiment.',
        '\n## Nested outer results','\n| Model | Brier ↓ | ROC-AUC | PR-AUC | F1 at 0.5 | Sensitivity | Specificity |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in means.iterrows():lines.append(f'| {name} | {r.brier:.5f} | {r.roc_auc:.5f} | {r.pr_auc_trapezoid:.5f} | {r.f1:.5f} | {r.recall:.5f} | {r.specificity:.5f} |')
    lines+=['\n## What the evidence establishes',
        f"\nBest evaluated hybrid: **{best_hybrid}**. Strongest single-base control by Brier: **{strongest_control}**. Relative Brier reduction: **{result['relative_brier_reduction_percent']:.2f}%** (negative means the hybrid was worse).",
        f"\nLower Brier than every control: **{result['lower_brier_than_every_control']}**. Improvement versus the strongest control in every seed: **{result['every_seed_improves_on_strongest_control']}**.",
        '\n## Paired differences (hybrid minus control; negative Brier difference favors hybrid)',
        '\n| Comparison | Brier difference | Conditional 95% interval | Winning seed/fold pairs |',
        '|---|---:|---|---:|']
    for name,v in paired.items():lines.append(f"| {name} | {v['mean_brier_difference']:.5f} | [{v['percentile_95'][0]:.5f}, {v['percentile_95'][1]:.5f}] | {v['winning_seed_fold_pairs']}/{v['total_seed_fold_pairs']} |")
    lines+=['\n## How to present this',
        '\nDescribe exactly which metric improves and against which controls, including seed consistency and intervals. Do not claim uniform metric superiority, statistical confirmation, clinical benefit, algorithmic novelty or guaranteed marks. Report unfavorable residual-model and original-hybrid results too.',
        '\nThis is standard feature fusion and residual learning applied to the assigned BN/MLP gap. The implementation rationale, nested selection, calibration-only controls, saved coefficients and ablations make the contribution inspectable.',
        '\n## Limits',
        '\nThe follow-up was motivated after prior results were inspected. The selection records are reused for exploratory research; outer folds limit within-study tuning leakage but do not create a new untouched dataset. Conditional bootstrap intervals omit overlapping-CV training variation, model-selection variation and patient clustering. Earlier graph/provenance/patient-linkage limitations remain. Additional marks remain a faculty judgment.']
    (run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run')
    print(json.dumps(summarize(parser.parse_args().run),indent=2))
