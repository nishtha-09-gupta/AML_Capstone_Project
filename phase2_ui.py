"""Present verified Phase 2 evidence using the Streamlit app's existing runtime."""

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from advanced_modeling import CATEGORY_OPTIONS, FEATURES
from heartlens_ui import COLORS, LABELS, chart, hero, yes_no
from study import ROOT, read_json


PHASE2 = ROOT / "artifacts/phase2"
METRICS = {
    "Brier score": ("brier", "Mean squared error of the probabilities. Lower is better."),
    "F1": ("f1", "Balances precision and recall at the study cutoff. Higher is better."),
    "ROC-AUC": ("roc_auc", "Measures how well scores rank positive above negative records. Higher is better."),
    "Sensitivity": ("recall", "Share of recorded positives classified as positive. Higher is better."),
    "Specificity": ("specificity", "Share of recorded negatives classified as negative. Higher is better."),
}
EVALUATIONS = ("Calibration follow-up", "Original selection CV", "Original retrospective test", "Hospital holdouts")
NUMERIC_FORMAT = {
    "age": "Years", "trestbps": "mm Hg", "chol": "mg/dL",
    "thalch": "Beats per minute", "oldpeak": "ST depression relative to rest",
}


@lru_cache(maxsize=2)
def evidence_index(run):
    directory = PHASE2 / run
    if read_json(directory / "verification.json")["status"] != "PASS":
        raise ValueError("Phase 2 verification did not pass")
    return read_json(directory / "evidence_sha256.json")


@lru_cache(maxsize=None)
def evidence_file(run, relative):
    data = (PHASE2 / run / relative).read_bytes()
    if sha256(data).hexdigest() != evidence_index(run)[relative]:
        raise ValueError("Phase 2 evidence differs from its recorded digest")
    return data


def metrics_table(evaluation):
    if evaluation == EVALUATIONS[0]:
        return pd.read_csv(BytesIO(evidence_file("hybrid_followup_v1", "summary.csv")))
    if evaluation == EVALUATIONS[1]:
        by_seed = pd.read_csv(BytesIO(evidence_file("full_run", "cv_by_seed.csv")))
        return by_seed.groupby("model", as_index=False).mean(numeric_only=True)
    if evaluation == EVALUATIONS[2]:
        results = pd.read_csv(BytesIO(evidence_file("full_run", "evaluation_summary.csv")))
        return results.loc[results["seed"].eq(42) & results["operating_point"].eq("tuned")]
    results = pd.read_csv(BytesIO(evidence_file("full_run", "hospital_summary.csv")))
    hospital = st.selectbox("Held-out hospital", sorted(results["hospital"].unique()), key="phase2_hospital")
    return results.loc[results["hospital"].eq(hospital)]


def comparisons():
    evaluation_control, metric_control = st.columns(2)
    with evaluation_control:
        evaluation = st.selectbox("Evaluation", EVALUATIONS, key="phase2_evaluation")
    with metric_control:
        metric = st.selectbox("Rank models by", list(METRICS), key="phase2_metric")
    if evaluation == EVALUATIONS[0]:
        run = "hybrid_followup_v1"
        finding = json.loads(evidence_file(run, "finding.json"))
        paired = json.loads(evidence_file(run, "paired_comparisons.json"))
        low, high = paired["Calibrated BN-MLP minus Calibrated BN"]["percentile_95"]
        st.write(f"The calibrated BN–MLP stack had the lowest mean Brier score ({finding['hybrid_brier']:.5f}). "
                 f"It reduced Brier by {finding['relative_brier_reduction_percent']:.2f}% relative to calibrated BN. "
                 f"The conditional 95% interval for the difference spanned {low:.5f} to {high:.5f}.")
        st.caption("Exploratory nested cross-validation on 550 development records, with five outer folds, three inner folds and three seeds. F1, sensitivity and specificity use a 0.50 cutoff.")
    elif evaluation == EVALUATIONS[1]:
        st.write("The standalone MLP was selected from BN, Naive Bayes, MLP and BN + MLP. Its mean F1 was 0.8296, compared with 0.8283 for the original hybrid.")
        st.caption("Five common development folds; means across available seeds. Classification metrics use a 0.50 cutoff. The other models are references and ablations.")
    elif evaluation == EVALUATIONS[2]:
        st.write("On the retrospective test set, the original MLP had higher F1 and ROC-AUC and lower Brier score than the original BN + MLP.")
        st.caption("184 retrospective records, seed 42. Cutoffs were chosen on a separate development partition and frozen before scoring.")
    else:
        st.caption("Each model was fitted using development records from the other hospitals. The held-out hospital was used for evaluation.")
    results = metrics_table(evaluation)
    field, description = METRICS[metric]
    st.caption(description)
    results = results.sort_values(field, ascending=field == "brier", kind="stable")
    ranked = results.iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ranked[field], y=ranked["model"], orientation="h",
        marker_color=[COLORS[0] if model == results.iloc[0]["model"] else "#C8A68B"
                      for model in ranked["model"]],
        text=ranked[field], texttemplate="%{x:.4f}", textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_xaxes(range=[0, 1.35 * ranked[field].max()], fixedrange=True, tickfont=dict(color="#3B261E"))
    fig.update_yaxes(fixedrange=True, automargin=True, showgrid=False, tickfont=dict(size=11, color="#3B261E"))
    chart(fig, 75 + 31 * len(ranked), key="phase2_rankings")
    fields = {column: label for label, (column, _) in METRICS.items()}
    if "cutoff" in results:
        fields["cutoff"] = "Cutoff"
    table = results[["model", *fields]].rename(columns={"model": "Model", **fields})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={label: st.column_config.NumberColumn(format="%.4f") for label in fields.values()})


def input_variables():
    st.subheader("The 11 model inputs")
    st.write("Both the Bayesian network and the neural models use these variables to classify recorded disease presence.")
    rows = []
    for feature in FEATURES:
        if feature in NUMERIC_FORMAT:
            values = NUMERIC_FORMAT[feature]
        elif feature in {"fbs", "exang"}:
            values = "Yes, no"
        else:
            values = ", ".join(CATEGORY_OPTIONS[feature])
        rows.append({"Variable": LABELS[feature], "Dataset field": feature, "Units or values": values})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("How the models prepare the inputs"):
        st.markdown(
            "- **MLP:** fills missing values, scales numeric measurements and one-hot encodes categories. "
            "It uses two hidden layers with 32 and 16 units, ReLU, dropout and weight decay.\n"
            "- **Bayesian network (BN):** bins continuous values and marginalizes over missing evidence. "
            "Its graph includes the dependence between ST depression and ST slope.\n"
            "- **Original BN + MLP:** adds the Bayesian network score as a feature. "
            "Cross-fitting generates this score without using the same row's label.\n"
            "- **Calibrated BN–MLP:** combines the two models' log-odds using logistic regression "
            "trained on predictions made on held-out folds."
        )
    st.caption("Record ID, hospital, vessel count (ca), and thalassemia (thal) are excluded as predictors. The target is the presence of recorded disease (num > 0).")


def example_predictions():
    examples = read_json(ROOT / "assets/phase2_examples.json")
    st.subheader("Compare scores for a fictional profile")
    st.write(f"Choose a profile to replay scores from four frozen follow-up models using seed {examples['seed']}.")
    name = st.selectbox("Fictional profile", list(examples["profiles"]), key="phase2_profile")
    profile = examples["profiles"][name]
    cutoff = examples["cutoff"]
    st.dataframe(pd.DataFrame([
        {"Model": model, "Model score": score, "Decision": "Above cutoff" if score >= cutoff else "Below cutoff"}
        for model, score in profile["scores"].items()
    ]), hide_index=True, width="stretch",
        column_config={"Model score": st.column_config.NumberColumn(format="percent")})
    st.caption(f"All four models use a {cutoff:.0%} demonstration cutoff. Scores are educational examples.")
    with st.expander("See the profile inputs"):
        inputs = [{"Variable": LABELS[key], "Value": yes_no(value) if isinstance(value, bool) or value is None else str(value)}
                  for key, value in profile["inputs"].items()]
        st.dataframe(pd.DataFrame(inputs), hide_index=True, width="stretch")
    with st.expander("See how BN–MLP combines its scores"):
        explanation = profile["explanation"]
        components = {"Intercept": "intercept", "Bayesian network": "bn_log_odds_contribution",
                      "Neural network": "mlp_log_odds_contribution", "Combined log-odds": "combined_log_odds"}
        st.dataframe(pd.DataFrame([{"Component": label, "Log-odds": explanation[key]}
                                   for label, key in components.items()]), hide_index=True, width="stretch",
                     column_config={"Log-odds": st.column_config.NumberColumn(format="%+.4f")})
        st.caption("Positive contributions raise the combined score and negative contributions lower it. The contributions describe the model calculation.")


def phase2():
    hero("Phase 2 · Deep learning",
         "Explore how a neural network and a Bayesian network classify the same heart disease records.")
    try:
        with st.container(key="phase2-study"):
            models, variables, examples = st.tabs(["Compare models", "Input variables", "Example predictions"])
            with models:
                comparisons()
            with variables:
                input_variables()
            with examples:
                example_predictions()
    except (OSError, ValueError, KeyError):
        st.error("The saved Phase 2 evidence could not be loaded. Please report this to the project team.")
