import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import plotly.express as px

_MAX_TRAIN_ROWS = 50_000  # sample cap to keep training fast in the UI


# ── Internal helpers ──────────────────────────────────────────────────────────

def _encode_features(X: pd.DataFrame):
    """Label-encode all object/category columns. Returns (X_encoded, encoders_dict)."""
    encoders = {}
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le
    return X_enc, encoders


def _detect_task(series: pd.Series) -> bool:
    """Return True if the series should be treated as a classification target."""
    if series.dtype == object or series.dtype.name == "category":
        return True
    # Treat small-cardinality integers as classification
    return series.dtype.kind in "iu" and series.nunique() <= 15


# ── Public API ────────────────────────────────────────────────────────────────

def render_predictor(
    df: pd.DataFrame,
    key_prefix: str = "ml",
    default_target: str = None,
    default_features: list = None,
):
    """
    Render a self-contained ML prediction section inside a Streamlit page.

    Parameters
    ----------
    df               : DataFrame to train on (will be sampled if > 50 000 rows)
    key_prefix       : unique prefix for all Streamlit widget keys (must differ
                       per call to avoid key collisions)
    default_target   : column selected by default as the prediction target
    default_features : list of columns pre-selected as features (defaults to
                       first 10 non-target columns)
    """
    st.markdown("---")
    st.subheader("Predictions")

    if df is None or df.empty:
        st.info("No data available for predictions.")
        return

    all_cols = df.columns.tolist()

    # ── Target column ─────────────────────────────────────────────────────────
    default_idx = (
        all_cols.index(default_target)
        if default_target and default_target in all_cols
        else 0
    )
    target = st.selectbox(
        "Target column to predict", all_cols, index=default_idx, key=f"{key_prefix}_target"
    )

    is_classification = _detect_task(df[target].dropna())
    n_unique = df[target].nunique()
    task_label = "Classification" if is_classification else "Regression"
    st.caption(f"Task: **{task_label}** — {n_unique} unique values in `{target}`")

    # ── Feature columns ───────────────────────────────────────────────────────
    feature_pool = [c for c in all_cols if c != target]
    if default_features:
        preselected = [f for f in default_features if f in feature_pool]
    else:
        preselected = feature_pool[: min(10, len(feature_pool))]

    selected_features = st.multiselect(
        "Feature columns", feature_pool, default=preselected, key=f"{key_prefix}_features"
    )
    if not selected_features:
        st.info("Select at least one feature column to continue.")
        return

    # ── Prepare data ──────────────────────────────────────────────────────────
    model_df = df[selected_features + [target]].dropna().reset_index(drop=True)

    if len(model_df) < 50:
        st.warning(f"Only {len(model_df)} complete rows — need at least 50 to train a model.")
        return

    sampled = False
    if len(model_df) > _MAX_TRAIN_ROWS:
        model_df = model_df.sample(_MAX_TRAIN_ROWS, random_state=42).reset_index(drop=True)
        sampled = True

    X = model_df[selected_features].copy()
    y = model_df[target].copy()

    X_enc, feat_encoders = _encode_features(X)

    if is_classification:
        target_encoder = LabelEncoder()
        y_enc = pd.Series(
            target_encoder.fit_transform(y.astype(str)), index=y.index
        )
    else:
        target_encoder = None
        y_enc = pd.to_numeric(y, errors="coerce")
        valid = y_enc.notna()
        X_enc = X_enc[valid].reset_index(drop=True)
        y_enc = y_enc[valid].reset_index(drop=True)

    if len(X_enc) < 50:
        st.warning("Not enough valid rows after encoding. Check that the target column is numeric.")
        return

    # ── Train model ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y_enc, test_size=0.2, random_state=42
    )

    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    with st.spinner("Training model…"):
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if sampled:
        st.caption(
            f"Model trained on a random sample of {_MAX_TRAIN_ROWS:,} rows "
            f"(full dataset: {len(df):,} rows)."
        )

    # ── Metrics ───────────────────────────────────────────────────────────────
    col_m1, col_m2, col_m3 = st.columns(3)
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        col_m1.metric("Test Accuracy", f"{acc:.1%}")
        col_m2.metric("Training rows", f"{len(X_train):,}")
        col_m3.metric("Classes", str(len(target_encoder.classes_)))
        acc_label = "strong" if acc >= 0.8 else ("moderate" if acc >= 0.6 else "weak")
        st.info(
            f"**Model performance:** A test accuracy of **{acc:.1%}** means the model correctly predicts "
            f"**`{target}`** for {acc:.0%} of unseen records, a **{acc_label}** result. "
            + ("The model is reliable for decision-making."
               if acc >= 0.8 else
               ("Predictions are directionally useful but carry meaningful uncertainty. Cross-check with domain knowledge."
                if acc >= 0.6 else
                "The model struggles with this target. Consider adding more informative features or reviewing data quality."))
        )
    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        col_m1.metric("RMSE", f"{rmse:.4f}")
        col_m2.metric("R²", f"{r2:.4f}")
        col_m3.metric("Training rows", f"{len(X_train):,}")
        r2_label = "strong" if r2 >= 0.1 else ("weak" if r2 > 0 else "no correlation")
        st.info(
            f"**Model performance:** An R² of **{r2:.4f}** means the selected features explain "
            f"**{r2*100:.1f}% of the variation** in `{target}`, a **{r2_label}** fit. "
            f"An RMSE of **{rmse:.4f}** means predictions are typically off by plus or minus {rmse:.4f} units on average. "
            + ("The model captures the pattern well and forecasts are reliable."
               if r2 >= 0.1 else
               ("The model shows a weak relationship. Treat predictions as directional guidance."
                if r2 > 0 else
                "The model shows no explanatory power. Consider adding stronger predictor features."))
        )

    # ── Feature importance ────────────────────────────────────────────────────
    imp_df = (
        pd.DataFrame({"Feature": selected_features, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    fig_imp = px.bar(
        imp_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(
        yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False
    )
    st.plotly_chart(fig_imp, width="stretch")

    top_feature     = imp_df.iloc[0]["Feature"]
    top_importance  = imp_df.iloc[0]["Importance"]
    second_feature  = imp_df.iloc[1]["Feature"] if len(imp_df) > 1 else None
    second_imp      = imp_df.iloc[1]["Importance"] if len(imp_df) > 1 else 0
    top_two_pct     = (top_importance + second_imp) * 100
    st.info(
        f"**Feature importance insight:** **`{top_feature}`** is the strongest predictor of `{target}` "
        f"(importance score: {top_importance:.3f})"
        + (f", followed by **`{second_feature}`** ({second_imp:.3f}). "
           f"Together they account for **{top_two_pct:.1f}%** of the model's decisions."
           if second_feature else ".")
        + f" Focusing interventions or data collection on `{top_feature}` will have the greatest impact on outcomes."
    )

    # ── Interactive prediction form ───────────────────────────────────────────
    with st.expander("Make a prediction", expanded=False):
        st.caption("Fill in values for each feature to get a model prediction.")

        n_form_cols = min(3, len(selected_features))
        form_cols = st.columns(n_form_cols)
        user_input = {}

        for i, feat in enumerate(selected_features):
            c = form_cols[i % n_form_cols]
            if feat in feat_encoders:
                options = list(feat_encoders[feat].classes_)
                user_input[feat] = c.selectbox(feat, options, key=f"{key_prefix}_in_{feat}")
            else:
                col_vals = X_train[feat] if feat in X_train.columns else X_enc[feat]
                mn = float(col_vals.min())
                mx = float(col_vals.max())
                med = float(col_vals.median())
                user_input[feat] = c.number_input(
                    feat, min_value=mn, max_value=mx, value=med,
                    key=f"{key_prefix}_in_{feat}",
                )

        if st.button("Predict", key=f"{key_prefix}_predict_btn"):
            input_df = pd.DataFrame([user_input])

            for feat, enc in feat_encoders.items():
                if feat in input_df.columns:
                    val_str = str(input_df.at[0, feat])
                    if val_str in set(enc.classes_):
                        input_df[feat] = enc.transform([val_str])
                    else:
                        input_df[feat] = -1  # unseen label fallback

            pred = model.predict(input_df)[0]

            if is_classification and target_encoder is not None:
                label = target_encoder.inverse_transform([int(pred)])[0]
                st.success(f"**Predicted {target}:** `{label}`")

                proba = model.predict_proba(input_df)[0]
                proba_df = pd.DataFrame(
                    {"Class": target_encoder.classes_, "Probability": proba}
                ).sort_values("Probability", ascending=False)
                fig_p = px.bar(
                    proba_df,
                    x="Class",
                    y="Probability",
                    title="Prediction Probabilities",
                    color="Probability",
                    color_continuous_scale="Blues",
                )
                fig_p.update_layout(coloraxis_showscale=False)
                st.plotly_chart(fig_p, width="stretch")
            else:
                st.success(f"**Predicted {target}:** `{pred:.4f}`")
