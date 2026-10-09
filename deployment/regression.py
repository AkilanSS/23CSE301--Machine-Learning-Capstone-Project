
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Infrared Thermography Regression", layout="wide")

st.title("Infrared Thermography — Core Temperature Regression")
st.caption("Interactive Streamlit version of the regression notebook")

BASE_MAPPING = {
    'T_offset': 'Calibration_Offset',
    'Max1R13_': 'Right_Inner_Eye_Max',
    'Max1L13_': 'Left_Inner_Eye_Max',
    'aveAllR13_': 'Right_Inner_Eye_Avg',
    'aveAllL13_': 'Left_Inner_Eye_Avg',
    'T_RC': 'Right_Eye_Avg',
    'T_RC_Dry': 'Right_Eye_Skin',
    'T_RC_Wet': 'Right_Tear_Duct',
    'T_RC_Max': 'Right_Eye_Max',
    'T_LC': 'Left_Eye_Avg',
    'T_LC_Dry': 'Left_Eye_Skin',
    'T_LC_Wet': 'Left_Tear_Duct',
    'T_LC_Max': 'Left_Eye_Max',
    'RCC': 'Right_Eye_Center',
    'LCC': 'Left_Eye_Center',
    'canthiMax': 'Hottest_Eye_Overall',
    'canthi4Max': 'Hottest_Eye_4Points',
    'T_FHCC': 'Forehead_Center',
    'T_FHRC': 'Forehead_Right',
    'T_FHLC': 'Forehead_Left',
    'T_FHBC': 'Forehead_Bottom',
    'T_FHTC': 'Forehead_Top',
    'T_FH_Max': 'Forehead_Overall_Max',
    'T_FHC_Max': 'Forehead_Center_Max',
    'T_Max': 'Face_Overall_Max',
    'T_OR': 'Mouth_Avg',
    'T_OR_Max': 'Mouth_Max'
}

FULL_MAPPING = {
    'SubjectID': 'Subject_ID',
    'aveOralF': 'True_Core_Temp_Fast',
    'aveOralM': 'True_Core_Temp_Slow',
    'T_atm': 'Room_Temperature',
    'Humidity': 'Room_Humidity',
    'Distance': 'Camera_Distance_m',
    'Cosmetics': 'Wearing_Makeup'
}

for old_base, new_base in BASE_MAPPING.items():
    for round_num in range(1, 5):
        FULL_MAPPING[f"{old_base}{round_num}"] = f"{new_base}_R{round_num}"


FEATURE_COLS = [
    'Calibration_Offset',
    'Right_Inner_Eye_Max',
    'Left_Inner_Eye_Max',
    'Right_Inner_Eye_Avg',
    'Left_Inner_Eye_Avg',
    'Forehead_Center',
    'Forehead_Right',
    'Forehead_Left',
    'Forehead_Top',
    'Forehead_Overall_Max',
    'Mouth_Avg',
    'Room_Temperature',
    'Room_Humidity',
    'Camera_Distance_m',
    'Gender',
    'Ethnicity',
    'Wearing_Makeup'
]

NUMERICAL_COLS = [
    'Calibration_Offset',
    'Right_Inner_Eye_Max',
    'Left_Inner_Eye_Max',
    'Right_Inner_Eye_Avg',
    'Left_Inner_Eye_Avg',
    'Forehead_Center',
    'Forehead_Right',
    'Forehead_Left',
    'Forehead_Top',
    'Forehead_Overall_Max',
    'Mouth_Avg',
    'Room_Temperature',
    'Room_Humidity',
    'Camera_Distance_m'
]

CATEGORICAL_COLS = ['Gender', 'Ethnicity', 'Wearing_Makeup']
TARGET = 'True_Core_Temp_Slow'


@st.cache_data(show_spinner=False)
def load_and_prepare(file_bytes):
    source = io.BytesIO(file_bytes)

    rounds = [
        pd.read_excel(source, sheet_name='Round 1', header=2),
        pd.read_excel(source, sheet_name='Round 2', header=2),
        pd.read_excel(source, sheet_name='Round 3', header=2),
        pd.read_excel(source, sheet_name='Round 4', header=2)
    ]

    for frame in rounds:
        frame.rename(columns=FULL_MAPPING, inplace=True)

    # Same pooled IQR filtering used in the notebook
    outlier_cols = ['Room_Humidity', 'Camera_Distance_m', 'Room_Temperature']
    bounds = {}

    for col in outlier_cols:
        pooled = pd.concat([r[col] for r in rounds], ignore_index=True)
        q1 = pooled.quantile(0.25)
        q3 = pooled.quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    cleaned = []
    for frame in rounds:
        mask = pd.Series(True, index=frame.index)
        for col, (low, high) in bounds.items():
            mask &= frame[col].between(low, high)
        frame = frame.loc[mask].copy()
        frame.dropna(inplace=True)
        cleaned.append(frame)

    # Remove round suffixes and combine all four rounds
    combined = []
    for i, frame in enumerate(cleaned, start=1):
        frame = frame.copy()
        frame.columns = frame.columns.str.replace(f'_R{i}', '', regex=False)
        combined.append(frame)

    full_data = pd.concat(combined, ignore_index=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(full_data, groups=full_data['Subject_ID']))

    train_data = full_data.iloc[train_idx].copy()
    test_data = full_data.iloc[test_idx].copy()

    train_model_data = train_data[FEATURE_COLS + [TARGET]].dropna().copy()
    test_model_data = test_data[FEATURE_COLS + [TARGET]].dropna().copy()

    train_groups = train_data.loc[train_model_data.index, 'Subject_ID']

    X_train = train_model_data[FEATURE_COLS]
    y_train = train_model_data[TARGET]
    X_test = test_model_data[FEATURE_COLS]
    y_test = test_model_data[TARGET]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUMERICAL_COLS),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), CATEGORICAL_COLS)
    ])

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    return {
        'rounds': cleaned,
        'full_data': full_data,
        'train_data': train_data,
        'test_data': test_data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_groups': train_groups,
        'preprocessor': preprocessor,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'feature_names': feature_names
    }


@st.cache_resource(show_spinner=True)
def train_models(file_bytes):
    data = load_and_prepare(file_bytes)

    Xtr = data['X_train_processed']
    Xte = data['X_test_processed']
    ytr = data['y_train']
    yte = data['y_test']
    groups = data['train_groups']
    gcv = GroupKFold(n_splits=5)

    results = []
    models = {}
    tuning = {}

    def evaluate(name, model, grid=None):
        if grid is not None:
            search = GridSearchCV(
                model, grid, cv=gcv, scoring='r2', n_jobs=1, pre_dispatch=1
            )
            search.fit(Xtr, ytr, groups=groups)
            fitted = search.best_estimator_
            best_params = search.best_params_
            cv_r2 = search.best_score_
        else:
            fitted = model.fit(Xtr, ytr)
            best_params = {}
            cv_r2 = np.nan

        pred = fitted.predict(Xte)
        results.append({
            'Model': name,
            'R2': r2_score(yte, pred),
            'RMSE': np.sqrt(mean_squared_error(yte, pred)),
            'MAE': mean_absolute_error(yte, pred),
            'CV R2': cv_r2
        })
        models[name] = fitted
        tuning[name] = best_params

    evaluate('Linear Regression', LinearRegression())

    evaluate(
        'Ridge Regression',
        Ridge(),
        {'alpha': [0.1, 1.0, 10.0, 100.0]}
    )

    evaluate(
        'Lasso Regression',
        Lasso(max_iter=10000),
        {'alpha': [0.001, 0.01, 0.1, 1.0]}
    )

    evaluate(
        'ElasticNet Regression',
        ElasticNet(max_iter=10000, random_state=42),
        {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
    )

    # Polynomial regression: choose degree from the notebook's 2/3 comparison
    poly_scores = {}
    for degree in [2, 3]:
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        scores = cross_val_score(
            pipe, Xtr, ytr, cv=gcv, groups=groups, scoring='r2', n_jobs=1
        )
        poly_scores[degree] = scores.mean()

    best_degree = max(poly_scores, key=poly_scores.get)
    evaluate(
        f'Polynomial Regression (deg={best_degree})',
        Pipeline([
            ('poly', PolynomialFeatures(degree=best_degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
    )

    evaluate(
        'Decision Tree Regressor',
        DecisionTreeRegressor(random_state=42),
        {'max_depth': [5, 10, None]}
    )

    evaluate(
        'Random Forest Regressor',
        RandomForestRegressor(random_state=42, n_jobs=1),
        {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
    )

    evaluate(
        'Gradient Boosting Regressor',
        GradientBoostingRegressor(random_state=42),
        {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [2, 3]
        }
    )

    evaluate(
        'Support Vector Regressor (SVR)',
        SVR(),
        {
            'C': [1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale']
        }
    )

    evaluate(
        'K-Nearest Neighbors Regressor',
        KNeighborsRegressor(),
        {'n_neighbors': [3, 7, 15]}
    )

    results_df = pd.DataFrame(results).sort_values('R2', ascending=False).reset_index(drop=True)
    results_df.insert(0, 'Rank', range(1, len(results_df) + 1))

    return {
        **data,
        'results': results_df,
        'models': models,
        'tuning': tuning,
        'poly_scores': poly_scores,
        'best_model_name': results_df.iloc[0]['Model']
    }


uploaded = st.sidebar.file_uploader(
    "Upload ICI.xlsx",
    type=['xlsx'],
    help="Use the same ICI Excel workbook used by the notebook. It must contain Round 1, Round 2, Round 3 and Round 4."
)

if uploaded is None:
    st.info("Upload your ICI.xlsx workbook from the sidebar to start.")
    st.markdown("""
    ### What this app contains
    - Data cleaning and pooled IQR outlier filtering
    - Subject-level 80/20 train-test split
    - StandardScaler + OneHotEncoder preprocessing
    - 10 regression models with GroupKFold hyperparameter tuning
    - R², RMSE and MAE comparison
    - Predicted-vs-actual and residual plots
    - Feature importance / permutation importance
    - Interactive prediction using the best model
    """)
    st.stop()

file_bytes = uploaded.getvalue()

with st.spinner("Preparing data and training the regression models... This may take a few minutes."):
    try:
        result = train_models(file_bytes)
    except Exception as exc:
        st.error("The workbook could not be processed or the models could not be trained. Check that the Excel file contains the expected columns and all four Round sheets.")
        st.exception(exc)
        st.stop()

results_df = result['results']
best_name = result['best_model_name']
best_model = result['models'][best_name]

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "EDA", "Model Comparison", "Best Model", "Prediction"
])

with tab1:
    st.header("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows", len(result['full_data']))
    c2.metric("Training rows", len(result['X_train']))
    c3.metric("Testing rows", len(result['X_test']))
    c4.metric("Best model", best_name)

    st.subheader("Combined dataset")
    st.dataframe(result['full_data'].head(20), use_container_width=True)

    st.subheader("Train / test split")
    split_df = pd.DataFrame({
        'Set': ['Training', 'Testing'],
        'Rows': [len(result['X_train']), len(result['X_test'])],
        'Subjects': [
            result['train_groups'].nunique(),
            result['test_data']['Subject_ID'].nunique()
        ]
    })
    st.dataframe(split_df, use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")

    eda_round = st.selectbox("Round", [1, 2, 3, 4])
    eda_data = result['rounds'][eda_round - 1]

    st.subheader("Target distribution")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(eda_data['True_Core_Temp_Slow'], kde=True, ax=ax)
    ax.set_xlabel("True Core Temperature")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Target relationships")
    feature = st.selectbox(
        "Feature",
        [
            'Right_Inner_Eye_Max_R%d' % eda_round,
            'Forehead_Overall_Max_R%d' % eda_round
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(
        data=eda_data,
        x=feature,
        y='True_Core_Temp_Slow',
        alpha=0.6,
        ax=ax
    )
    ax.set_title(f"{feature} vs True Core Temperature")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation heatmap")
    numeric = eda_data.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(numeric.corr(), cmap='coolwarm', annot=False, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.header("Regression Model Comparison")

    st.dataframe(
        results_df.style.format({
            'R2': '{:.4f}',
            'RMSE': '{:.4f}',
            'MAE': '{:.4f}',
            'CV R2': '{:.4f}'
        }),
        use_container_width=True
    )

    metric = st.selectbox("Metric", ['R2', 'RMSE', 'MAE', 'CV R2'])

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = results_df.sort_values(metric, ascending=(metric != 'R2'))
    sns.barplot(data=plot_df, x=metric, y='Model', ax=ax)
    ax.set_title(f"{metric} Comparison")
    st.pyplot(fig)
    plt.close(fig)

    st.success(f"Best test R² model: {best_name}")

with tab4:
    st.header(f"Best Model: {best_name}")

    best_row = results_df.iloc[0]
    a, b, c = st.columns(3)
    a.metric("R²", f"{best_row['R2']:.4f}")
    b.metric("RMSE", f"{best_row['RMSE']:.4f}")
    c.metric("MAE", f"{best_row['MAE']:.4f}")

    st.subheader("Best hyperparameters")
    st.json(result['tuning'][best_name])

    predictions = best_model.predict(result['X_test_processed'])

    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=result['y_test'], y=predictions, alpha=0.6, ax=ax)
    lo = min(result['y_test'].min(), predictions.min())
    hi = max(result['y_test'].max(), predictions.max())
    ax.plot([lo, hi], [lo, hi], '--', linewidth=2)
    ax.set_xlabel("Actual True Core Temperature")
    ax.set_ylabel("Predicted True Core Temperature")
    ax.set_title(f"Predicted vs Actual — {best_name}")
    st.pyplot(fig)
    plt.close(fig)

    residuals = result['y_test'].to_numpy() - predictions

    st.subheader("Residual plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.6, ax=ax)
    ax.axhline(0, linestyle='--', linewidth=2)
    ax.set_xlabel("Predicted True Core Temperature")
    ax.set_ylabel("Residual (Actual − Predicted)")
    ax.set_title(f"Residual Plot — {best_name}")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Permutation importance")
    perm = permutation_importance(
        best_model,
        result['X_test_processed'],
        result['y_test'],
        n_repeats=5,
        random_state=42,
        scoring='r2',
        n_jobs=1
    )
    perm_df = pd.DataFrame({
        'Feature': result['feature_names'],
        'Importance': perm.importances_mean
    }).sort_values('Importance', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=perm_df, x='Importance', y='Feature', ax=ax)
    ax.set_title("Top 15 Permutation Importances")
    st.pyplot(fig)
    plt.close(fig)

with tab5:
    st.header("Make a Prediction")
    st.write(f"Using the best model: **{best_name}**")

    values = {}

    left, right = st.columns(2)

    for i, col in enumerate(NUMERICAL_COLS):
        container = left if i % 2 == 0 else right
        series = result['X_train'][col]
        values[col] = container.number_input(
            col,
            value=float(series.median()),
            step=0.01
        )

    for col in CATEGORICAL_COLS:
        options = result['X_train'][col].dropna().unique().tolist()
        values[col] = st.selectbox(col, options)

    if st.button("Predict Core Temperature", type="primary"):
        input_df = pd.DataFrame([values])
        input_processed = result['preprocessor'].transform(input_df)
        prediction = best_model.predict(input_processed)[0]

        st.success(f"Predicted True Core Temperature: **{prediction:.2f} °C**")

        if prediction >= 38.0:
            st.warning("The prediction is at or above the 38 °C reference threshold used in the notebook.")
        else:
            st.info("The prediction is below the 38 °C reference threshold used in the notebook.")
