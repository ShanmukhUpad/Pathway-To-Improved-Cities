import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

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

    # ── Prominent model output headline ──────────────────────────────────────
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        # Find the most commonly predicted class across the test set
        from collections import Counter
        pred_counts    = Counter(y_pred.tolist())
        top_pred_enc   = pred_counts.most_common(1)[0][0]
        top_pred_label = target_encoder.inverse_transform([int(top_pred_enc)])[0]
        top_pred_pct   = pred_counts.most_common(1)[0][1] / len(y_pred) * 100

        st.markdown(f"""
<div style="background:rgba(79,142,247,0.1); border-left:4px solid #4f8ef7;
            padding:16px 20px; border-radius:8px; margin:12px 0;">
  <p style="margin:0; font-size:11px; color:#9eaec4; text-transform:uppercase;
            letter-spacing:0.08em;">Model Prediction — most likely outcome</p>
  <p style="margin:6px 0 2px; font-size:2rem; font-weight:700;
            color:#ffffff; line-height:1.1;">
    {top_pred_label}
    <span style="font-size:1rem; font-weight:400; color:#9eaec4;">
      &nbsp;({top_pred_pct:.0f}% of test cases)
    </span>
  </p>
  <p style="margin:4px 0 0; font-size:13px; color:#9eaec4;">
    Predicting <strong style="color:#ffffff;">{target}</strong>
    from {len(selected_features)} feature(s) · {len(X_train):,} training rows
  </p>
</div>
""", unsafe_allow_html=True)

    else:
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2   = float(r2_score(y_test, y_pred))
        mean_pred = float(np.mean(y_pred))

        st.markdown(f"""
<div style="background:rgba(79,142,247,0.1); border-left:4px solid #4f8ef7;
            padding:16px 20px; border-radius:8px; margin:12px 0;">
  <p style="margin:0; font-size:11px; color:#9eaec4; text-transform:uppercase;
            letter-spacing:0.08em;">Model Prediction — typical value</p>
  <p style="margin:6px 0 2px; font-size:2rem; font-weight:700;
            color:#ffffff; line-height:1.1;">
    {mean_pred:,.2f}
    <span style="font-size:1rem; font-weight:400; color:#9eaec4;">
      &nbsp;{target} &nbsp;(±{rmse:.2f})
    </span>
  </p>
  <p style="margin:4px 0 0; font-size:13px; color:#9eaec4;">
    Average predicted value across test set · {len(X_train):,} training rows
  </p>
</div>
""", unsafe_allow_html=True)

    # ── Metrics ───────────────────────────────────────────────────────────────
    col_m1, col_m2, col_m3 = st.columns(3)
    if is_classification:
        acc_label = "strong" if acc >= 0.8 else ("moderate" if acc >= 0.6 else "weak")
        col_m1.metric("Test Accuracy", f"{acc:.1%}")
        col_m2.metric("Training rows", f"{len(X_train):,}")
        col_m3.metric("Classes", str(len(target_encoder.classes_)))
        st.info(
            f"**Model performance:** A test accuracy of **{acc:.1%}** means the model correctly predicts "
            f"**`{target}`** for {acc:.0%} of unseen records — a **{acc_label}** result. "
            + ("Reliable for decision-making."
               if acc >= 0.8 else
               ("Directionally useful but carries uncertainty. Cross-check with domain knowledge."
                if acc >= 0.6 else
                "The model struggles with this target. Consider adding more informative features or reviewing data quality."))
        )
    else:
        r2_label = "strong" if r2 >= 0.1 else ("weak" if r2 > 0 else "no correlation")
        col_m1.metric("RMSE", f"{rmse:.4f}")
        col_m2.metric("R²", f"{r2:.4f}")
        col_m3.metric("Training rows", f"{len(X_train):,}")
        st.info(
            f"**Model performance:** R² = **{r2:.4f}** — features explain **{r2*100:.1f}%** of variation in `{target}` "
            f"({r2_label} fit). Typical prediction error: ±{rmse:.4f} units. "
            + ("The model captures the pattern well."
               if r2 >= 0.1 else
               ("Weak relationship — treat as directional guidance."
                if r2 > 0 else
                "No explanatory power. Consider stronger predictor features."))
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
    with st.expander("Custom Scenario Predictor", expanded=False):
        st.caption(
            "Set specific values for each feature to get a one-time model prediction. "
            "This is **not** a time forecast — it answers: *'What would the model predict "
            "if conditions were exactly these values?'*"
        )

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


# ── Time-series forecasting helpers ──────────────────────────────────────────

def _detect_frequency(date_series: pd.Series) -> str:
    """
    Detect the dominant temporal frequency of a date series.
    Returns a pandas offset alias: 'D', '7D', 'MS', 'QS', or 'YS'.
    """
    dates = pd.to_datetime(date_series, infer_datetime_format=True, errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return "MS"
    diffs = dates.diff().dropna()
    median_days = diffs.dt.days.median()
    if median_days <= 1.5:
        return "D"
    elif median_days <= 8:
        return "7D"
    elif median_days <= 32:
        return "MS"
    elif median_days <= 95:
        return "QS"
    return "YS"


def _make_lag_df(ts: pd.Series) -> tuple:
    """
    Build lag feature matrix from a time-ordered Series.
    Returns (X DataFrame, y Series, list of feature names).
    Automatically drops features that cause too many NaNs for short series.
    """
    n = len(ts)
    feat = {}

    feat["lag1"] = ts.shift(1)
    if n > 6:
        feat["lag3"] = ts.shift(3)
        feat["rolling3"] = ts.shift(1).rolling(3, min_periods=2).mean()
    if n > 14:
        feat["lag6"] = ts.shift(6)

    feat_df = pd.DataFrame(feat, index=ts.index)
    combined = pd.concat([feat_df, ts.rename("target")], axis=1).dropna()
    feature_names = list(feat.keys())
    return combined[feature_names], combined["target"], feature_names


def _forecast_ahead(model, ts: pd.Series, feature_names: list, n_steps: int) -> list:
    """
    Autoregressively forecast n_steps ahead using the fitted model.
    Each step appends the prediction to history and recomputes lag features.
    """
    history = list(ts.values.astype(float))
    predictions = []

    for _ in range(n_steps):
        row = {}
        n_hist = len(history)
        for fname in feature_names:
            if fname == "lag1":
                row[fname] = history[-1]
            elif fname == "lag3":
                row[fname] = history[-3] if n_hist >= 3 else history[0]
            elif fname == "lag6":
                row[fname] = history[-6] if n_hist >= 6 else history[0]
            elif fname == "rolling3":
                recent = history[-3:] if n_hist >= 3 else history
                row[fname] = float(np.mean(recent))
        # Fallback: replace any NaN with last known value
        for k, v in row.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                row[k] = history[-1]
        pred = float(max(0, model.predict(pd.DataFrame([row]))[0]))
        predictions.append(pred)
        history.append(pred)

    return predictions


# ── Public time-series API ────────────────────────────────────────────────────

def render_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    key_prefix: str = "forecast",
    n_periods: int = None,
):
    """
    Render a time-series forecast section inside a Streamlit page.

    Parameters
    ----------
    df          : source DataFrame
    date_col    : column to use as the time axis
    target_col  : numeric column to forecast
    key_prefix  : unique prefix for Streamlit widget keys
    n_periods   : number of future periods to forecast; if None, shows a slider
    """
    st.markdown("---")
    st.subheader("Time-Series Forecast")

    if df is None or df.empty:
        st.info("No data available.")
        return

    # ── 1. Parse and sort by date ─────────────────────────────────────────────
    df2 = df.copy()
    col_data = df2[date_col]

    # Handle year-only integer columns (e.g., 2020, 2021, 2022)
    if pd.api.types.is_numeric_dtype(col_data):
        numeric_vals = pd.to_numeric(col_data, errors="coerce").dropna()
        if len(numeric_vals) > 0 and ((numeric_vals >= 1900) & (numeric_vals <= 2100)).all():
            df2["_date"] = pd.to_datetime(
                col_data.apply(lambda x: f"{int(x)}-01-01" if pd.notna(x) else None),
                errors="coerce",
            )
        else:
            df2["_date"] = pd.to_datetime(col_data, errors="coerce")
    else:
        df2["_date"] = pd.to_datetime(col_data, infer_datetime_format=True, errors="coerce")

    df2[target_col] = pd.to_numeric(df2[target_col], errors="coerce")
    df2 = df2.dropna(subset=["_date", target_col]).sort_values("_date")

    if df2.empty:
        st.warning("No valid rows after parsing date and target columns.")
        return

    # ── 2. Resample to regular frequency ──────────────────────────────────────
    freq = _detect_frequency(df2["_date"])
    ts = (
        df2.set_index("_date")[target_col]
        .resample(freq)
        .mean()
        .interpolate(method="time", limit=3)
        .dropna()
    )

    if len(ts) < 6:
        st.warning(
            f"Only **{len(ts)}** time periods found after resampling to `{freq}` frequency. "
            "Need at least **6** to generate a forecast. Upload more historical data."
        )
        return

    # ── 3. Forecast horizon ───────────────────────────────────────────────────
    n_forecast = n_periods if n_periods is not None else st.slider(
        "Periods to forecast ahead",
        min_value=1, max_value=24, value=6,
        key=f"{key_prefix}_n_periods",
    )

    # ── 4. Build lag features ─────────────────────────────────────────────────
    X, y, feature_names = _make_lag_df(ts)

    if len(X) < 5:
        st.warning("Not enough complete rows for model training after building lag features.")
        return

    # ── 5. TimeSeriesSplit CV: Ridge vs Random Forest ─────────────────────────
    n_splits = min(5, max(2, len(X) // 3))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    ridge = Ridge(alpha=1.0)
    rf    = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    ridge_cv  = cross_val_score(ridge, X, y, cv=tscv, scoring="neg_mean_squared_error")
    rf_cv     = cross_val_score(rf,    X, y, cv=tscv, scoring="neg_mean_squared_error")
    ridge_rmse = float(np.sqrt(-ridge_cv.mean()))
    rf_rmse    = float(np.sqrt(-rf_cv.mean()))

    if ridge_rmse <= rf_rmse:
        best_model      = ridge
        best_model_name = "Ridge Regression"
        cv_rmse         = ridge_rmse
    else:
        best_model      = rf
        best_model_name = "Random Forest"
        cv_rmse         = rf_rmse

    # ── 6. Train on all data ──────────────────────────────────────────────────
    with st.spinner("Training forecast model…"):
        best_model.fit(X, y)

    # ── 7. Metrics ────────────────────────────────────────────────────────────
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Model", best_model_name)
    cm2.metric("CV RMSE", f"{cv_rmse:.4f}")
    cm3.metric("Training periods", str(len(ts)))
    cm4.metric("Forecast periods", str(n_forecast))

    # ── 8. Autoregressive forecast ────────────────────────────────────────────
    forecast_values = _forecast_ahead(best_model, ts, feature_names, n_forecast)

    _freq_offsets = {"D": "D", "7D": "7D", "MS": "MS", "QS": "QS", "YS": "YS"}
    freq_offset  = _freq_offsets.get(freq, "MS")
    forecast_dates = pd.date_range(
        start=ts.index[-1], periods=n_forecast + 1, freq=freq_offset
    )[1:]

    # ── 9. Plotly chart ───────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts.index, y=ts.values,
        mode="lines", name="Actual",
        line=dict(color="#4f8ef7", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[ts.index[-1]] + list(forecast_dates),
        y=[float(ts.values[-1])] + forecast_values,
        mode="lines+markers", name="Forecast",
        line=dict(color="#e05050", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    fig.add_vrect(
        x0=ts.index[-1], x1=forecast_dates[-1],
        fillcolor="rgba(224,80,80,0.07)", line_width=0,
        annotation_text="Forecast window", annotation_position="top left",
    )
    fig.update_layout(
        title=f"Forecast: {target_col}",
        xaxis_title="Date",
        yaxis_title=target_col,
        legend=dict(orientation="h"),
        margin={"t": 50},
    )
    st.plotly_chart(fig, use_container_width=True)

    if n_forecast > 6:
        st.warning(
            "Forecasts beyond 6 periods accumulate error at each step. "
            "Treat long-horizon predictions as directional guidance only."
        )

    # ── 10. Prominent prediction headline + insight ───────────────────────────
    last_actual   = float(ts.values[-1])
    last_forecast = forecast_values[-1]
    next_val      = forecast_values[0]   # the very next period (most reliable)
    trend = "upward" if last_forecast > last_actual else "downward"
    pct_change = abs((last_forecast - last_actual) / last_actual * 100) if last_actual != 0 else 0
    arrow = "▲" if next_val >= last_actual else "▼"
    diff  = next_val - last_actual

    # Format the next-period date as a human label
    try:
        next_period_label = forecast_dates[0].strftime("%B %Y")
    except Exception:
        next_period_label = str(forecast_dates[0])

    # Big headline banner
    st.markdown(f"""
<div style="background:rgba(79,142,247,0.1); border-left:4px solid #4f8ef7;
            padding:16px 20px; border-radius:8px; margin:12px 0;">
  <p style="margin:0; font-size:11px; color:#9eaec4; text-transform:uppercase;
            letter-spacing:0.08em;">Forecast — {next_period_label}</p>
  <p style="margin:6px 0 0; font-size:2.2rem; font-weight:700; color:#ffffff; line-height:1.1;">
    {next_val:,.2f}
    <span style="font-size:1rem; font-weight:400; color:#9eaec4;">&nbsp;{target_col}</span>
  </p>
  <p style="margin:4px 0 0; font-size:13px; color:#9eaec4;">
    {arrow} {abs(diff):,.2f} from current ({last_actual:,.2f})
  </p>
</div>
""", unsafe_allow_html=True)

    st.info(
        f"**Forecast detail:** {best_model_name} selected (CV RMSE: {cv_rmse:.4f}, "
        f"{n_splits}-fold time-series CV). "
        f"Over {n_forecast} period(s), `{target_col}` trends **{trend}** "
        f"by ~{pct_change:.1f}% from the current value of {last_actual:,.2f}. "
        f"Model trained on {len(ts)} historical periods."
    )
