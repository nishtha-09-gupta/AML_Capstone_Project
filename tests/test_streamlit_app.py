"""Exercise live inference and the visitor paths without fitting a model."""

from unittest.mock import patch

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

import study
from heartlens_ui import PRESETS


def test_live_inference_changes_without_fitting():
    models = [study.frozen_model(name) for name in study.MODEL_NAMES]
    with patch.object(type(models[0]), "fit", side_effect=AssertionError("Inference must not fit")):
        first = study.predict_profile(PRESETS["Profile A"], compare=True)
        second = study.predict_profile(PRESETS["Profile B"], compare=True)
    assert len(first) == 5
    assert first.Score.between(0, 1).all()
    assert first.iloc[0].Score < .38 < second.iloc[0].Score
    assert all(first["Study cutoff"] == second["Study cutoff"])
    missing = study.predict_profile(PRESETS["Missing values"], compare=True)
    assert np.isfinite(missing.Score).all()


def test_invalid_input_and_saved_threshold_metrics():
    with pytest.raises(ValueError, match="outside"):
        study.predict_profile({**PRESETS["Profile A"], "age": 101})
    with pytest.raises(ValueError, match="known model"):
        study.threshold_scores("untrusted", .5)
    for name in study.MODEL_NAMES:
        saved = study.evidence()["metrics"]["families"][name]
        actual = study.threshold_scores(name, saved["threshold"])
        for key in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
            assert actual[key] == pytest.approx(saved["tuned"][key], abs=1e-12)
    assert len(study.cohort()) == 918
    assert study.cohort().chol.isna().sum() == 201


def test_app_submission_controls_and_evidence_views():
    app = AppTest.from_file(str(study.ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert not app.dataframe
    app.button[0].click().run()
    assert not app.exception
    initial = app.dataframe[0].value
    app.selectbox(key="preset").select("Profile B").run()
    assert not app.exception
    assert app.number_input(key="input_age").value == 65
    assert not app.dataframe
    app.button[0].click().run()
    assert not app.exception
    assert not initial.equals(app.dataframe[0].value)
    app.selectbox(key="preset").select("Missing values").run()
    assert not app.exception
    assert all(field.value is None for field in app.number_input if field.key != "input_age")
    assert not app.dataframe
    app.button[0].click().run()
    assert not app.exception
    assert app.dataframe
    app.switch_page("views/models.py").run()
    assert not app.exception
    assert [m for m in app.metric if m.label == "Recall"][-1].value == "96.1%"
    app.slider(key="cutoff_Random forest").set_value(1.0).run()
    assert not app.exception
    assert [m for m in app.metric if m.label == "Recall"][-1].value == "0.0%"
    assert [link.label for link in app.get("page_link")] == ["Prediction", "Model studio", "Phase 2"]


def test_phase2_explores_verified_results_and_frozen_examples():
    app = AppTest.from_file(str(study.ROOT / "app.py"), default_timeout=60).run()
    app.switch_page("views/phase2.py").run()
    assert not app.exception and not app.error
    assert app.dataframe[0].value.iloc[0]["Model"] == "Calibrated BN-MLP"
    assert len(app.dataframe[1].value) == 11
    first_score = app.dataframe[2].value.iloc[0]["Model score"]
    app.selectbox(key="phase2_profile").select("Profile B").run()
    assert not app.exception and not app.error
    assert app.dataframe[2].value.iloc[0]["Model score"] > first_score
    app.selectbox(key="phase2_evaluation").select("Original selection CV").run()
    assert not app.exception and not app.error
    assert set(app.dataframe[0].value["Model"]) >= {"MLP", "BN + MLP"}
    app.selectbox(key="phase2_evaluation").select("Original retrospective test").run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 9
    assert "Cutoff" in app.dataframe[0].value
    app.selectbox(key="phase2_evaluation").select("Hospital holdouts").run()
    app.selectbox(key="phase2_hospital").select("Switzerland").run()
    assert not app.exception and not app.error
    assert len(app.dataframe[0].value) == 4


def test_typing_does_not_run_inference_until_predict():
    with patch("heartlens_ui.predict_profile", wraps=study.predict_profile) as infer:
        app = AppTest.from_file(str(study.ROOT / "app.py"), default_timeout=60).run()
        assert not app.exception
        app.number_input(key="input_age").set_value(65)
        assert not app.exception
        infer.assert_not_called()
        app.button[0].click().run()
        assert not app.exception
        assert infer.call_count == 1
        assert infer.call_args.args[0]["age"] == 65
        app.run()
        assert infer.call_count == 1
