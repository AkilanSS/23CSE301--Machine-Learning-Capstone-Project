from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Core Temperature Predictor", layout="wide")

ART = Path(__file__).parent / "artifacts"


@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(ART / "preprocessor.joblib")
    models = joblib.load(ART / "models.joblib")
    meta = joblib.load(ART / "meta.joblib")
    metrics = pd.read_csv(ART / "metrics.csv")
    return preprocessor, models, meta, metrics


preprocessor, models, meta, metrics = load_artifacts()

st.title("Core Body Temperature Prediction from Thermal Imaging")
st.caption("Trained regression models predicting True Core Temperature (Slow) from facial IR thermography features.")

# ---------------- Sidebar inputs ----------------
st.sidebar.header("Input features")

row = {}
for col in meta["numerical_cols"]:
    stats = meta["num_stats"][col]
    row[col] = st.sidebar.number_input(
        col,
        min_value=float(stats["min"]),
        max_value=float(stats["max"]),
        value=float(stats["mean"]),
        format="%.2f",
    )

row["Gender"] = st.sidebar.selectbox("Gender", meta["gender"])
row["Ethnicity"] = st.sidebar.selectbox("Ethnicity", meta["ethnicity"])
row["Wearing_Makeup"] = st.sidebar.selectbox("Wearing_Makeup", meta["makeup"])

# Same column order as training, then the fitted preprocessor
X_new = pd.DataFrame([row])[meta["feature_cols"]]
X_proc = preprocessor.transform(X_new)

# ---------------- Tabs ----------------
tab_all, tab_single = st.tabs(["Compare all 10 models", "Single model"])

with tab_all:
    preds = pd.DataFrame(
        {
            "Model": list(models.keys()),
            "Predicted Temp (°C)": [float(m.predict(X_proc)[0]) for m in models.values()],
        }
    ).merge(metrics[["Model", "R2", "RMSE", "MAE"]], on="Model", how="left")

    st.subheader("Predictions from all models")
    st.dataframe(preds.round(3), use_container_width=True, hide_index=True)

    st.bar_chart(preds.set_index("Model")["Predicted Temp (°C)"])

    spread = preds["Predicted Temp (°C)"].max() - preds["Predicted Temp (°C)"].min()
    st.caption(
        f"Range across models: {spread:.3f} °C. R², RMSE and MAE are test-set metrics from training."
    )

with tab_single:
    choice = st.selectbox("Choose a model", list(models.keys()))
    pred = float(models[choice].predict(X_proc)[0])

    c1, c2 = st.columns(2)
    c1.metric("Predicted core temperature", f"{pred:.2f} °C")

    m = metrics[metrics["Model"] == choice]
    if not m.empty:
        c2.write("**Test-set performance**")
        c2.write(
            f"R²: {m['R2'].iloc[0]:.3f}  |  RMSE: {m['RMSE'].iloc[0]:.3f}  |  MAE: {m['MAE'].iloc[0]:.3f}"
        )

    if pred >= 38.0:
        st.warning("Predicted value is at or above the 38.0 °C fever reference.")
    else:
        st.info("Predicted value is below the 38.0 °C fever reference.")