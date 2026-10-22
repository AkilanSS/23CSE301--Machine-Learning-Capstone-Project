import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="ML Workbench", page_icon="📊", layout="wide")
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

@st.cache_resource
def load_artifact(filename):
    path = ARTIFACTS / filename
    if not path.exists():
        return None
    return joblib.load(path)

st.title("Machine Learning Workbench")
st.caption("Regression · Classification · Clustering")
page = st.sidebar.radio("Choose task", ["Regression", "Classification", "Clustering"])

if page == "Regression":
    st.header("Infrared Thermography — Core Temperature Regression")

    # The regression notebook currently exports these four separate files.
    models = load_artifact("models.joblib")
    preprocessor = load_artifact("preprocessor.joblib")
    meta = load_artifact("meta.joblib")
    metrics_path = ARTIFACTS / "metrics.csv"

    if models is None or preprocessor is None or meta is None:
        missing = []
        if models is None:
            missing.append("models.joblib")
        if preprocessor is None:
            missing.append("preprocessor.joblib")
        if meta is None:
            missing.append("meta.joblib")
        st.error(
            "Missing regression artifact(s) in deployment/artifacts: "
            + ", ".join(missing)
            + ". Run the regression notebook export cell and commit these files."
        )
        st.stop()

    feature_cols = list(meta.get("feature_cols", []))
    numerical_cols = list(meta.get("numerical_cols", []))
    if not feature_cols:
        st.error("meta.joblib does not contain feature_cols. Re-run the regression notebook export cell.")
        st.stop()

    # The export cell stores the observed category choices under these keys.
    category_options = {
        "Gender": meta.get("gender", []),
        "Ethnicity": meta.get("ethnicity", []),
        "Wearing_Makeup": meta.get("makeup", []),
    }
    categorical_cols = [c for c in feature_cols if c not in numerical_cols]
    stats = meta.get("num_stats", {})

    st.write("Choose a saved model and enter the feature values. Inputs are passed through the fitted notebook preprocessor before prediction.")

    if metrics_path.exists():
        try:
            metrics_df = pd.read_csv(metrics_path)
            st.subheader("Saved regression model metrics")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Could not read metrics.csv: {exc}")

    model_names = list(models.keys())
    selected = st.selectbox("Regression model", model_names)
    values = {}
    left, right = st.columns(2)

    for i, col in enumerate(feature_cols):
        parent = left if i % 2 == 0 else right
        if col in categorical_cols:
            choices = category_options.get(col, [])
            if choices:
                values[col] = parent.selectbox(col, choices, key=f"reg_{col}")
            else:
                values[col] = parent.text_input(col, key=f"reg_{col}")
        else:
            col_stats = stats.get(col, {}) if isinstance(stats, dict) else {}
            default = col_stats.get("mean", 0.0) if isinstance(col_stats, dict) else 0.0
            minimum = col_stats.get("min") if isinstance(col_stats, dict) else None
            maximum = col_stats.get("max") if isinstance(col_stats, dict) else None
            try:
                default = float(default)
                if not np.isfinite(default):
                    default = 0.0
            except (TypeError, ValueError):
                default = 0.0
            kwargs = {"value": default, "key": f"reg_{col}", "format": "%.5f"}
            try:
                if minimum is not None and np.isfinite(float(minimum)):
                    kwargs["min_value"] = float(minimum)
                if maximum is not None and np.isfinite(float(maximum)):
                    kwargs["max_value"] = float(maximum)
            except (TypeError, ValueError):
                pass
            # Avoid invalid min/max ranges from incomplete metadata.
            if "min_value" in kwargs and "max_value" in kwargs and kwargs["min_value"] > kwargs["max_value"]:
                kwargs.pop("min_value", None)
                kwargs.pop("max_value", None)
            values[col] = parent.number_input(col, **kwargs)

    if st.button("Predict core temperature", type="primary"):
        try:
            row = pd.DataFrame([values], columns=feature_cols)
            transformed = preprocessor.transform(row)
            prediction = models[selected].predict(transformed)[0]
            st.success(f"Predicted core temperature: {float(prediction):.2f} °C")
        except Exception as exc:
            st.error("Prediction failed. Check that the notebook's saved model and preprocessor were exported from the same training run.")
            st.exception(exc)

    st.info("This page loads the existing models.joblib, preprocessor.joblib, meta.joblib, and metrics.csv files. It does not retrain models or run GridSearchCV.")

elif page == "Classification":
    st.header("EEG Motor Imagery Classification")
    bundle = load_artifact("classification_models.pkl")
    if bundle is None:
        st.error("Missing deployment/artifacts/classification_models.pkl. Run the classification export cell and commit the artifact.")
        st.stop()

    models = bundle.get("models", bundle.get("classifiers", {}))
    if not models:
        st.error("The classification artifact does not contain a 'models' dictionary. Check its export cell.")
        st.stop()

    st.success("Saved classification models loaded. No dataset upload is required.")
    model_name = st.selectbox("Classifier", list(models.keys()))

    # Prefer saved sample rows if the notebook export includes them.
    X_sample = bundle.get("sample_features", bundle.get("X_sample", bundle.get("X_train")))
    y_sample = bundle.get("sample_labels", bundle.get("y_sample", bundle.get("y_train")))
    feature_cols = bundle.get("feature_cols", bundle.get("feature_names"))

    if X_sample is not None:
        X_sample = np.asarray(X_sample)
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        n_samples = X_sample.shape[0]
        show_n = st.slider("Built-in samples to classify", 1, min(50, n_samples), min(10, n_samples))
        if st.button("Classify built-in samples", type="primary"):
            try:
                predictions = models[model_name].predict(X_sample[:show_n])
                out = pd.DataFrame({"Predicted class": predictions})
                if y_sample is not None:
                    actual = np.asarray(y_sample).reshape(-1)
                    if len(actual) >= show_n:
                        out.insert(0, "Actual class", actual[:show_n])
                st.dataframe(out, use_container_width=True, hide_index=True)
                st.subheader("Predicted class distribution")
                st.bar_chart(pd.Series(predictions).value_counts().sort_index().rename_axis("Class").to_frame("Count"))
            except Exception as exc:
                st.error("Could not classify the saved sample rows. Confirm the artifact's sample features use the same representation as the fitted classifiers.")
                st.exception(exc)
    else:
        st.info("The saved artifact contains models but no reusable sample feature rows. Add sample_features and sample_labels to the notebook export cell to enable the built-in demonstration. No upload is required for viewing saved results.")

    saved_metrics = bundle.get("metrics", bundle.get("results"))
    if saved_metrics is not None:
        st.subheader("Saved classification metrics")
        try:
            st.dataframe(pd.DataFrame(saved_metrics), use_container_width=True, hide_index=True)
        except Exception:
            st.write(saved_metrics)

else:
    st.header("EEG Motor Imagery Clustering")
    bundle = load_artifact("clustering_models.pkl")
    if bundle is None:
        st.error("Missing artifacts/clustering_models.pkl. Run the clustering export cell in your notebook first.")
        st.stop()
    st.write("This page includes the notebook's K-Means and four Agglomerative linkage methods, saved evaluation metrics, PCA/t-SNE visualizations, and K-Means assignment for new PCA feature rows.")

    # Show evaluation metrics calculated on the original notebook samples.
    saved_metrics = bundle.get("metrics", {})
    if saved_metrics:
        st.subheader("Clustering evaluation on notebook samples")
        metrics_df = pd.DataFrame(saved_metrics).T
        metrics_df.index.name = "Algorithm"
        st.dataframe(metrics_df.round(4), use_container_width=True)

    # Visualize the stored original-sample embeddings and labels.
    training_labels = bundle.get("training_labels", {})
    pca_2d = bundle.get("pca_2d_embedding")
    tsne_2d = bundle.get("tsne_embedding")
    if training_labels and pca_2d is not None:
        st.subheader("PCA 2-D cluster visualization")
        names = list(training_labels.keys())
        chosen = st.selectbox("Algorithm to visualize", names, key="cluster_viz_method")
        lab = np.asarray(training_labels[chosen])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(np.asarray(pca_2d)[:, 0], np.asarray(pca_2d)[:, 1], c=lab, cmap="coolwarm", s=22)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA 2-D — {chosen}")
        st.pyplot(fig)
        plt.close(fig)

        if tsne_2d is not None:
            st.subheader("t-SNE visualization")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(np.asarray(tsne_2d)[:, 0], np.asarray(tsne_2d)[:, 1], c=lab, cmap="coolwarm", s=22)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.set_title(f"t-SNE — {chosen}")
            st.pyplot(fig)
            plt.close(fig)

    st.divider()
    st.subheader("Explore saved clustering samples")
    Z = bundle.get("training_Z", bundle.get("Z", bundle.get("training_features")))
    if Z is not None:
        Z = np.asarray(Z)
        feature_cols = bundle.get("feature_cols", [f"PC{i+1}" for i in range(Z.shape[1])])
        if len(feature_cols) != Z.shape[1]:
            feature_cols = [f"PC{i+1}" for i in range(Z.shape[1])]
        n_samples = min(50, len(Z))
        show_n = st.slider("Number of saved feature rows", 1, n_samples, min(10, n_samples))
        st.dataframe(pd.DataFrame(Z[:show_n], columns=feature_cols), use_container_width=True, hide_index=True)
    else:
        st.info("The clustering artifact has saved metrics and visualizations, but no stored feature matrix for the sample table. Add training_Z to the notebook export cell if you want to display the saved rows.")

    st.caption("No CSV upload is required. This page displays saved clustering results and embeddings. New raw EEG files would still require the full preprocessing pipeline.")


