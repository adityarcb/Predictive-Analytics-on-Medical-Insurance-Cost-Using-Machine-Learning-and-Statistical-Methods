"""
Comprehensive Streamlit app for Health Insurance Claim Prediction.

Includes all models and analyses from the notebook:
- Regression: Simple, Multiple, Polynomial, Decision Tree, Random Forest, Gradient Boosting, KNN, MLP
- Classification: Logistic Regression, KNN, Naive Bayes, Decision Tree, SVM, MLP
- Unsupervised: KMeans (with Elbow), Hierarchical Clustering, Gaussian Mixture
- Dimensionality Reduction: PCA
- Ensemble: Bagging, AdaBoost, Gradient Boosting, Random Forest
- Model Evaluation: Cross-Validation, Bias-Variance Analysis
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score, train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


DATA_PATH = Path(__file__).parent / "healthinsurance.csv"

# Columns used throughout the notebook
CATEGORICAL_COLS = ["sex", "hereditary_diseases", "city", "job_title"]
FEATURE_COLS = [
    "age",
    "weight",
    "bmi",
    "no_of_dependents",
    "smoker",
    "bloodpressure",
    "diabetes",
    "regular_ex",
    "sex_encoded",
    "hereditary_diseases_encoded",
    "city_encoded",
    "job_title_encoded",
]


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["bmi"].fillna(df["bmi"].median(), inplace=True)
    df["bloodpressure"].fillna(df["bloodpressure"].median(), inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def encode_features(df: pd.DataFrame):
    df_clean = df.copy()
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_clean[f"{col}_encoded"] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le
    X = df_clean[FEATURE_COLS]
    y = df_clean["claim"]
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    return X[mask], y[mask], label_encoders, df_clean


@st.cache_resource(show_spinner=False)
def prepare_data():
    raw_df = load_data(DATA_PATH)
    X, y, encoders, df_clean = encode_features(raw_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create classification target (high claim = 1 if claim > median)
    y_median = y.median()
    y_clf = (y > y_median).astype(int)
    y_train_clf = (y_train > y_median).astype(int)
    y_test_clf = (y_test > y_median).astype(int)
    
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_clf": y_train_clf,
        "y_test_clf": y_test_clf,
        "encoders": encoders,
        "df_clean": df_clean,
        "y_median": y_median,
    }


@st.cache_resource(show_spinner=False)
def train_all_regression_models():
    data = prepare_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    
    results = {}
    
    # Simple Linear Regression (using BMI as single feature)
    simple_lr = LinearRegression()
    simple_lr.fit(X_train[["bmi"]], y_train)
    preds = simple_lr.predict(X_test[["bmi"]])
    results["Simple Linear Regression"] = {
        "model": simple_lr,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": None,
        "predictions": preds,
    }
    
    # Multiple Linear Regression
    mlr = LinearRegression()
    mlr.fit(X_train, y_train)
    preds = mlr.predict(X_test)
    results["Multiple Linear Regression"] = {
        "model": mlr,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": None,
        "predictions": preds,
    }
    
    # Polynomial Regression (degree 2)
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    poly_lr = LinearRegression()
    poly_lr.fit(X_train_poly, y_train)
    preds = poly_lr.predict(X_test_poly)
    results["Polynomial Regression (deg=2)"] = {
        "model": (poly_lr, poly_features),
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": None,
        "predictions": preds,
    }
    
    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=12, random_state=42)
    dt.fit(X_train, y_train)
    preds = dt.predict(X_test)
    results["Decision Tree"] = {
        "model": dt,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": get_feature_importance(dt),
        "predictions": preds,
    }
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    results["Random Forest"] = {
        "model": rf,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": get_feature_importance(rf),
        "predictions": preds,
    }
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(random_state=42)
    gb.fit(X_train, y_train)
    preds = gb.predict(X_test)
    results["Gradient Boosting"] = {
        "model": gb,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": get_feature_importance(gb),
        "predictions": preds,
    }
    
    # KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=7)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    results["KNN Regressor"] = {
        "model": knn,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": None,
        "predictions": preds,
    }
    
    # MLP Regressor
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train)
    preds = mlp.predict(X_test_scaled)
    results["MLP Regressor"] = {
        "model": (mlp, scaler),
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": None,
        "predictions": preds,
    }
    
    return results, data


@st.cache_resource(show_spinner=False)
def train_all_classification_models():
    data = prepare_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train_clf, y_test_clf = data["y_train_clf"], data["y_test_clf"]
    
    results = {}
    
    # Logistic Regression
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train, y_train_clf)
    preds = log_reg.predict(X_test)
    probs = log_reg.predict_proba(X_test)[:, 1]
    results["Logistic Regression"] = {
        "model": log_reg,
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "predictions": preds,
        "probabilities": probs,
    }
    
    # KNN Classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_scaled, y_train_clf)
    preds = knn.predict(X_test_scaled)
    probs = knn.predict_proba(X_test_scaled)[:, 1]
    results["KNN Classifier"] = {
        "model": (knn, scaler),
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "predictions": preds,
        "probabilities": probs,
    }
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train_clf)
    preds = nb.predict(X_test)
    probs = nb.predict_proba(X_test)[:, 1]
    results["Naive Bayes"] = {
        "model": nb,
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "predictions": preds,
        "probabilities": probs,
    }
    
    # Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=12, random_state=42)
    dt.fit(X_train, y_train_clf)
    preds = dt.predict(X_test)
    probs = dt.predict_proba(X_test)[:, 1]
    results["Decision Tree Classifier"] = {
        "model": dt,
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "feature_importance": get_feature_importance(dt),
        "predictions": preds,
        "probabilities": probs,
    }
    
    # SVM
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train_clf)
    preds = svm.predict(X_test_scaled)
    probs = svm.predict_proba(X_test_scaled)[:, 1]
    results["SVM"] = {
        "model": (svm, scaler),
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "predictions": preds,
        "probabilities": probs,
    }
    
    # MLP Classifier
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    mlp.fit(X_train_scaled, y_train_clf)
    preds = mlp.predict(X_test_scaled)
    probs = mlp.predict_proba(X_test_scaled)[:, 1]
    results["MLP Classifier"] = {
        "model": (mlp, scaler),
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "predictions": preds,
        "probabilities": probs,
    }
    
    return results, data


@st.cache_resource(show_spinner=False)
def train_ensemble_models():
    data = prepare_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    y_train_clf, y_test_clf = data["y_train_clf"], data["y_test_clf"]
    
    results = {"regression": {}, "classification": {}}
    
    # Bagging Regressor
    bag_reg = BaggingRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    bag_reg.fit(X_train, y_train)
    preds = bag_reg.predict(X_test)
    results["regression"]["Bagging"] = {
        "model": bag_reg,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": None,
    }
    
    # AdaBoost Regressor
    ada_reg = AdaBoostRegressor(n_estimators=50, random_state=42)
    ada_reg.fit(X_train, y_train)
    preds = ada_reg.predict(X_test)
    results["regression"]["AdaBoost"] = {
        "model": ada_reg,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": get_feature_importance(ada_reg),
    }
    
    # Gradient Boosting (already in regression, but include here for comparison)
    gb_reg = GradientBoostingRegressor(random_state=42)
    gb_reg.fit(X_train, y_train)
    preds = gb_reg.predict(X_test)
    results["regression"]["Gradient Boosting"] = {
        "model": gb_reg,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": get_feature_importance(gb_reg),
    }
    
    # Random Forest (already in regression, but include here for comparison)
    rf_reg = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train)
    preds = rf_reg.predict(X_test)
    results["regression"]["Random Forest"] = {
        "model": rf_reg,
        "metrics": calculate_regression_metrics(y_test, preds),
        "feature_importance": get_feature_importance(rf_reg),
    }
    
    # Bagging Classifier
    bag_clf = BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    bag_clf.fit(X_train, y_train_clf)
    preds = bag_clf.predict(X_test)
    probs = bag_clf.predict_proba(X_test)[:, 1]
    results["classification"]["Bagging"] = {
        "model": bag_clf,
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
    }
    
    # AdaBoost Classifier
    ada_clf = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada_clf.fit(X_train, y_train_clf)
    preds = ada_clf.predict(X_test)
    probs = ada_clf.predict_proba(X_test)[:, 1]
    results["classification"]["AdaBoost"] = {
        "model": ada_clf,
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "feature_importance": get_feature_importance(ada_clf),
    }
    
    # Gradient Boosting Classifier
    gb_clf = GradientBoostingClassifier(random_state=42)
    gb_clf.fit(X_train, y_train_clf)
    preds = gb_clf.predict(X_test)
    probs = gb_clf.predict_proba(X_test)[:, 1]
    results["classification"]["Gradient Boosting"] = {
        "model": gb_clf,
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "feature_importance": get_feature_importance(gb_clf),
    }
    
    # Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train_clf)
    preds = rf_clf.predict(X_test)
    probs = rf_clf.predict_proba(X_test)[:, 1]
    results["classification"]["Random Forest"] = {
        "model": rf_clf,
        "metrics": calculate_classification_metrics(y_test_clf, preds, probs),
        "feature_importance": get_feature_importance(rf_clf),
    }
    
    return results, data


def calculate_regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def calculate_classification_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0,
        "Log Loss": log_loss(y_true, y_proba),
    }


def get_feature_importance(model):
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
    return None


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def render_header():
    st.title("🏥 Health Insurance Claim Prediction - Comprehensive ML Dashboard")
    st.caption(
        "Complete machine learning pipeline: Regression, Classification, Clustering, "
        "Dimensionality Reduction, Neural Networks, and Ensemble Methods"
    )


def render_data_glance(df: pd.DataFrame):
    st.subheader("📊 Dataset Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Features", df.shape[1])
    c3.metric("Average Claim", format_currency(df["claim"].mean()))
    
    st.dataframe(df.head(10), use_container_width=True)
    
    with st.expander("📈 Data Distributions"):
        c1, c2 = st.columns(2)
        fig1 = px.histogram(df, x="claim", nbins=40, title="Claim Amount Distribution")
        c1.plotly_chart(fig1, use_container_width=True)
        fig2 = px.box(df, x="smoker", y="claim", color="smoker", title="Claims by Smoker Status")
        c2.plotly_chart(fig2, use_container_width=True)
        
        c3, c4 = st.columns(2)
        fig3 = px.histogram(df, x="age", nbins=30, title="Age Distribution")
        c3.plotly_chart(fig3, use_container_width=True)
        fig4 = px.scatter(df, x="bmi", y="claim", color="smoker", title="BMI vs Claim Amount")
        c4.plotly_chart(fig4, use_container_width=True)


def render_regression_metrics(title: str, metrics: dict):
    st.subheader(title)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", format_currency(metrics["MAE"]))
    c2.metric("MSE", format_currency(metrics["MSE"]))
    c3.metric("RMSE", format_currency(metrics["RMSE"]))
    c4.metric("R²", f"{metrics['R2']:.4f}")


def render_classification_metrics(title: str, metrics: dict):
    st.subheader(title)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    c2.metric("Precision", f"{metrics['Precision']:.4f}")
    c3.metric("Recall", f"{metrics['Recall']:.4f}")
    c4.metric("F1 Score", f"{metrics['F1']:.4f}")
    c5.metric("AUC", f"{metrics['AUC']:.4f}")
    c6.metric("Log Loss", f"{metrics['Log Loss']:.4f}")


def render_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode='markers', name='Predictions',
        marker=dict(color='steelblue', opacity=0.6)
    ))
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=title, xaxis_title="Actual Claim Amount ($)",
        yaxis_title="Predicted Claim Amount ($)", height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_residual_plot(y_true, y_pred, title="Residual Plot"):
    residuals = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred, y=residuals, mode='markers',
        marker=dict(color='coral', opacity=0.6)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title, xaxis_title="Predicted Claim Amount ($)",
        yaxis_title="Residuals", height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm, text_auto=True, aspect="auto",
        labels=dict(x="Predicted", y="Actual"),
        title=title, color_continuous_scale="Blues"
    )
    st.plotly_chart(fig, use_container_width=True)


def render_roc_curve(y_true, y_proba, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {auc_score:.3f})',
        line=dict(color='steelblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random Classifier', line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title=title, xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate", height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance(importances: pd.DataFrame):
    st.subheader("🔍 Feature Importance")
    top_feats = importances.head(10)
    fig = px.bar(
        top_feats, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Blues", height=450
    )
    st.plotly_chart(fig, use_container_width=True)


def render_elbow_plot(X_scaled, max_k=10):
    # Ensure no NaN values
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    inertias = []
    k_range = range(2, max_k + 1)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(k_range), y=inertias, mode='lines+markers',
        marker=dict(size=10, color='steelblue'),
        line=dict(width=2, color='steelblue')
    ))
    fig.update_layout(
        title="Elbow Method for Optimal K",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia", height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_dendrogram(X_scaled, linkage_method='ward', max_display=100):
    # Ensure no NaN values
    if isinstance(X_scaled, pd.DataFrame):
        X_scaled = X_scaled.values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    # Sample data if too large for dendrogram
    if len(X_scaled) > max_display:
        indices = np.random.choice(len(X_scaled), max_display, replace=False)
        X_sample = X_scaled[indices]
    else:
        X_sample = X_scaled
        indices = np.arange(len(X_scaled))
    
    linked = linkage(X_sample, method=linkage_method)
    
    fig = go.Figure()
    dendro_data = dendrogram(linked, no_plot=True)
    
    # Extract coordinates from dendrogram data
    icoord = np.array(dendro_data['icoord'])
    dcoord = np.array(dendro_data['dcoord'])
    
    # Plot each segment
    for i in range(len(icoord)):
        fig.add_trace(go.Scatter(
            x=icoord[i], y=dcoord[i],
            mode='lines', line=dict(color='steelblue', width=1.5),
            showlegend=False, hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=f"Dendrogram ({linkage_method.capitalize()} Linkage)",
        xaxis_title="Sample Index", yaxis_title="Distance", height=500
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pca_analysis(X_scaled):
    pca = PCA()
    pca.fit(X_scaled)
    
    # Variance explained
    variance_ratio = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(variance_ratio)
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Variance Explained", "Cumulative Variance"))
    
    fig.add_trace(
        go.Bar(x=list(range(1, min(13, len(variance_ratio) + 1))),
               y=variance_ratio[:12], name="Variance"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(1, min(13, len(cumsum_variance) + 1))),
                   y=cumsum_variance[:12], mode='lines+markers', name="Cumulative"),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Principal Component", row=1, col=1)
    fig.update_xaxes(title_text="Principal Component", row=1, col=2)
    fig.update_yaxes(title_text="Variance Explained", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Variance", row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info(f"First 3 components explain {cumsum_variance[2]:.1%} of variance")


def render_prediction_form(model_info: dict):
    st.subheader("🔮 Make a Prediction")
    encoders = model_info["encoders"]
    df_clean = model_info["df_clean"]
    model = model_info["model"]
    
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age", min_value=18, max_value=100, value=35)
        weight = c2.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        bmi = c3.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        
        c4, c5, c6 = st.columns(3)
        dependents = c4.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
        smoker = c5.selectbox("Smoker", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
        regular_ex = c6.selectbox("Regular Exercise", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
        
        c7, c8, c9 = st.columns(3)
        bloodpressure = c7.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
        diabetes = c8.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
        sex = c9.selectbox("Sex", options=sorted(df_clean["sex"].unique()))
        
        c10, c11 = st.columns(2)
        hereditary = c10.selectbox("Hereditary Diseases", options=sorted(df_clean["hereditary_diseases"].unique()))
        city = c11.selectbox("City", options=sorted(df_clean["city"].unique()))
        job_title = st.selectbox("Job Title", options=sorted(df_clean["job_title"].unique()))
        
        submitted = st.form_submit_button("🚀 Predict Claim Amount", use_container_width=True)
        
        if submitted:
            enc_inputs = {
                "sex_encoded": encoders["sex"].transform([sex])[0],
                "hereditary_diseases_encoded": encoders["hereditary_diseases"].transform([hereditary])[0],
                "city_encoded": encoders["city"].transform([city])[0],
                "job_title_encoded": encoders["job_title"].transform([job_title])[0],
            }
            input_vector = np.array([
                age, weight, bmi, dependents, smoker, bloodpressure,
                diabetes, regular_ex, enc_inputs["sex_encoded"],
                enc_inputs["hereditary_diseases_encoded"],
                enc_inputs["city_encoded"], enc_inputs["job_title_encoded"]
            ]).reshape(1, -1)
            
            # Handle models with scalers
            if isinstance(model, tuple):
                pred = model[0].predict(model[1].transform(input_vector))[0]
            else:
                pred = model.predict(input_vector)[0]
            
            st.success(f"💰 **Estimated Claim Amount: {format_currency(pred)}**")


def main():
    st.set_page_config(page_title="Health Insurance ML Dashboard", layout="wide", initial_sidebar_state="expanded")
    render_header()
    
    # Load all models
    with st.spinner("Loading models and data..."):
        reg_results, data = train_all_regression_models()
        clf_results, _ = train_all_classification_models()
        ensemble_results, _ = train_ensemble_models()
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Data", "📈 Regression", "🎯 Classification", "🔍 Unsupervised",
        "🧠 Neural Networks", "🎭 Ensemble", "📊 Model Evaluation", "🔮 Predict"
    ])
    
    with tab1:
        render_data_glance(data["df_clean"])
    
    with tab2:
        st.header("📈 Regression Models")
        reg_names = list(reg_results.keys())
        col1, col2 = st.columns([3, 1])
        selected_reg = col1.selectbox("Select Regression Model", reg_names, index=reg_names.index("Random Forest") if "Random Forest" in reg_names else 0)
        compare_all = col2.checkbox("Compare All", value=False)
        
        if compare_all:
            st.subheader("📊 All Regression Models Comparison")
            cmp_data = []
            for name, info in reg_results.items():
                m = info["metrics"]
                cmp_data.append({
                    "Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "R²": m["R2"]
                })
            cmp_df = pd.DataFrame(cmp_data).sort_values("R²", ascending=False)
            st.dataframe(cmp_df.style.highlight_max(axis=0, subset=["R²"]), use_container_width=True)
            
            # Comparison chart
            fig = go.Figure()
            for name, info in reg_results.items():
                fig.add_trace(go.Bar(name=name, x=["MAE", "RMSE"], y=[info["metrics"]["MAE"], info["metrics"]["RMSE"]]))
            fig.update_layout(barmode='group', title="Model Comparison (MAE & RMSE)", height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            chosen = reg_results[selected_reg]
            render_regression_metrics(f"{selected_reg} Performance", chosen["metrics"])
            
            if chosen["feature_importance"] is not None:
                render_feature_importance(chosen["feature_importance"])
            
            # Visualizations
            st.subheader("📉 Model Visualizations")
            c1, c2 = st.columns(2)
            with c1:
                render_actual_vs_predicted(data["y_test"], chosen["predictions"], f"{selected_reg}: Actual vs Predicted")
            with c2:
                render_residual_plot(data["y_test"], chosen["predictions"], f"{selected_reg}: Residual Plot")
    
    with tab3:
        st.header("🎯 Classification Models")
        clf_names = list(clf_results.keys())
        col1, col2 = st.columns([3, 1])
        selected_clf = col1.selectbox("Select Classification Model", clf_names, index=0)
        compare_all_clf = col2.checkbox("Compare All", value=False, key="clf_compare")
        
        if compare_all_clf:
            st.subheader("📊 All Classification Models Comparison")
            cmp_data = []
            for name, info in clf_results.items():
                m = info["metrics"]
                cmp_data.append({
                    "Model": name, "Accuracy": m["Accuracy"], "Precision": m["Precision"],
                    "Recall": m["Recall"], "F1": m["F1"], "AUC": m["AUC"]
                })
            cmp_df = pd.DataFrame(cmp_data).sort_values("Accuracy", ascending=False)
            st.dataframe(cmp_df.style.highlight_max(axis=0, subset=["Accuracy", "F1", "AUC"]), use_container_width=True)
        else:
            chosen = clf_results[selected_clf]
            render_classification_metrics(f"{selected_clf} Performance", chosen["metrics"])
            
            if "feature_importance" in chosen and chosen["feature_importance"] is not None:
                render_feature_importance(chosen["feature_importance"])
            
            # Visualizations
            st.subheader("📊 Classification Visualizations")
            c1, c2 = st.columns(2)
            with c1:
                render_confusion_matrix(data["y_test_clf"], chosen["predictions"], f"{selected_clf}: Confusion Matrix")
            with c2:
                render_roc_curve(data["y_test_clf"], chosen["probabilities"], f"{selected_clf}: ROC Curve")
    
    with tab4:
        st.header("🔍 Unsupervised Learning")
        cluster_type = st.selectbox("Clustering Algorithm", ["KMeans", "Hierarchical", "Gaussian Mixture"], index=0)
        
        # Get clean feature data without NaN values
        # Use the same X that was used for training (already cleaned)
        X_for_cluster = pd.concat([data["X_train"], data["X_test"]], axis=0)
        
        # Ensure no NaN values remain (should already be clean, but double-check)
        if X_for_cluster.isnull().any().any():
            X_for_cluster = X_for_cluster.fillna(X_for_cluster.median())
        
        # Convert to numpy array for clustering
        X_for_cluster_np = X_for_cluster.values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_for_cluster_np)
        
        if cluster_type == "KMeans":
            st.subheader("KMeans Clustering")
            with st.expander("Elbow Method"):
                render_elbow_plot(X_scaled, max_k=10)
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, 1)
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            labels = km.fit_predict(X_scaled)
            
            c1, c2 = st.columns(2)
            c1.metric("Silhouette Score", f"{silhouette_score(X_scaled, labels):.3f}")
            c2.metric("Inertia", f"{km.inertia_:,.0f}")
            
        elif cluster_type == "Hierarchical":
            st.subheader("Hierarchical Clustering")
            linkage_method = st.selectbox("Linkage Method", ["ward", "complete", "average", "single"], index=0)
            n_clusters = st.slider("Number of Clusters", 2, 10, 3, 1)
            
            with st.expander("Dendrogram"):
                render_dendrogram(X_scaled, linkage_method=linkage_method)
            
            agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
            labels = agg.fit_predict(X_scaled)
            
            st.metric("Silhouette Score", f"{silhouette_score(X_scaled, labels):.3f}")
            
        else:  # Gaussian Mixture
            st.subheader("Gaussian Mixture Model")
            n_clusters = st.slider("Number of Components", 2, 10, 3, 1)
            gm = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = gm.fit_predict(X_scaled)
            
            st.metric("Silhouette Score", f"{silhouette_score(X_scaled, labels):.3f}")
        
        # Cluster visualization
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        plot_df = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": labels
        })
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster", title="Clusters (PCA Projection)", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster analysis
        # Get corresponding rows from original dataframe
        y_all = pd.concat([data["y_train"], data["y_test"]], axis=0)
        # Get indices from the concatenated X
        all_indices = X_for_cluster.index
        # Get corresponding rows from df_clean
        df_original = data["df_clean"].loc[all_indices].copy()
        df_original["Cluster"] = labels
        st.subheader("Cluster Characteristics")
        cluster_stats = df_original.groupby("Cluster")[["age", "bmi", "claim"]].mean()
        st.dataframe(cluster_stats, use_container_width=True)
    
    with tab5:
        st.header("🧠 Neural Networks")
        nn_type = st.radio("Network Type", ["MLP Regressor", "MLP Classifier"], horizontal=True)
        
        if nn_type == "MLP Regressor":
            if "MLP Regressor" in reg_results:
                info = reg_results["MLP Regressor"]
                render_regression_metrics("MLP Regressor Performance", info["metrics"])
                render_actual_vs_predicted(data["y_test"], info["predictions"], "MLP Regressor: Actual vs Predicted")
        else:
            if "MLP Classifier" in clf_results:
                info = clf_results["MLP Classifier"]
                render_classification_metrics("MLP Classifier Performance", info["metrics"])
                render_confusion_matrix(data["y_test_clf"], info["predictions"], "MLP Classifier: Confusion Matrix")
                render_roc_curve(data["y_test_clf"], info["probabilities"], "MLP Classifier: ROC Curve")
    
    with tab6:
        st.header("🎭 Ensemble Methods")
        ensemble_type = st.radio("Task Type", ["Regression", "Classification"], horizontal=True)
        
        if ensemble_type == "Regression":
            st.subheader("Regression Ensemble Comparison")
            cmp_data = []
            for name, info in ensemble_results["regression"].items():
                m = info["metrics"]
                cmp_data.append({"Model": name, "MAE": m["MAE"], "RMSE": m["RMSE"], "R²": m["R2"]})
            cmp_df = pd.DataFrame(cmp_data).sort_values("R²", ascending=False)
            st.dataframe(cmp_df.style.highlight_max(axis=0, subset=["R²"]), use_container_width=True)
            
            selected_ens = st.selectbox("Select Ensemble Model", list(ensemble_results["regression"].keys()), index=0)
            chosen = ensemble_results["regression"][selected_ens]
            render_regression_metrics(f"{selected_ens} Performance", chosen["metrics"])
            if chosen["feature_importance"] is not None:
                render_feature_importance(chosen["feature_importance"])
        else:
            st.subheader("Classification Ensemble Comparison")
            cmp_data = []
            for name, info in ensemble_results["classification"].items():
                m = info["metrics"]
                cmp_data.append({
                    "Model": name, "Accuracy": m["Accuracy"], "Precision": m["Precision"],
                    "Recall": m["Recall"], "F1": m["F1"], "AUC": m["AUC"]
                })
            cmp_df = pd.DataFrame(cmp_data).sort_values("Accuracy", ascending=False)
            st.dataframe(cmp_df.style.highlight_max(axis=0, subset=["Accuracy", "F1"]), use_container_width=True)
            
            selected_ens = st.selectbox("Select Ensemble Model", list(ensemble_results["classification"].keys()), index=0)
            chosen = ensemble_results["classification"][selected_ens]
            render_classification_metrics(f"{selected_ens} Performance", chosen["metrics"])
            if "feature_importance" in chosen and chosen["feature_importance"] is not None:
                render_feature_importance(chosen["feature_importance"])
    
    with tab7:
        st.header("📊 Model Evaluation")
        eval_type = st.radio("Evaluation Method", ["Cross-Validation", "Bias-Variance Analysis"], horizontal=True)
        
        if eval_type == "Cross-Validation":
            st.subheader("K-Fold Cross-Validation")
            cv_type = st.selectbox("CV Type", ["K-Fold (k=5)", "Leave-One-Out"], index=0)
            model_for_cv = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Linear Regression"], index=0)
            
            if model_for_cv == "Random Forest":
                model_cv = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
            elif model_for_cv == "Gradient Boosting":
                model_cv = GradientBoostingRegressor(random_state=42)
            else:
                model_cv = LinearRegression()
            
            if cv_type == "K-Fold (k=5)":
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model_cv, data["X_train"], data["y_train"], cv=kfold, scoring='r2', n_jobs=-1)
                st.write(f"**R² Scores:** {cv_scores}")
                st.write(f"**Mean R²:** {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                
                fig = go.Figure()
                fig.add_trace(go.Bar(x=[f"Fold {i+1}" for i in range(5)], y=cv_scores, marker_color='steelblue'))
                fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {cv_scores.mean():.4f}")
                fig.update_layout(title=f"{model_for_cv} - K-Fold CV Results", yaxis_title="R² Score", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Leave-One-Out CV is computationally expensive. Using a sample of 200 for demonstration.")
                sample_size = min(200, len(data["X_train"]))
                indices = np.random.choice(len(data["X_train"]), sample_size, replace=False)
                X_sample = data["X_train"].iloc[indices]
                y_sample = data["y_train"].iloc[indices]
                
                loo = LeaveOneOut()
                cv_scores = cross_val_score(model_cv, X_sample, y_sample, cv=loo, scoring='r2', n_jobs=-1)
                st.write(f"**Mean R²:** {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
                st.write(f"**Min:** {cv_scores.min():.4f}, **Max:** {cv_scores.max():.4f}")
        
        else:  # Bias-Variance Analysis
            st.subheader("Bias-Variance Trade-off Analysis")
            
            # Simple model (Linear Regression)
            simple_model = LinearRegression()
            simple_model.fit(data["X_train"], data["y_train"])
            simple_train_pred = simple_model.predict(data["X_train"])
            simple_test_pred = simple_model.predict(data["X_test"])
            simple_train_error = mean_squared_error(data["y_train"], simple_train_pred)
            simple_test_error = mean_squared_error(data["y_test"], simple_test_pred)
            
            # Complex model (Deep Decision Tree)
            complex_model = DecisionTreeRegressor(max_depth=20, min_samples_split=2, random_state=42)
            complex_model.fit(data["X_train"], data["y_train"])
            complex_train_pred = complex_model.predict(data["X_train"])
            complex_test_pred = complex_model.predict(data["X_test"])
            complex_train_error = mean_squared_error(data["y_train"], complex_train_pred)
            complex_test_error = mean_squared_error(data["y_test"], complex_test_pred)
            
            # Balanced model (Random Forest)
            balanced_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            balanced_model.fit(data["X_train"], data["y_train"])
            balanced_train_pred = balanced_model.predict(data["X_train"])
            balanced_test_pred = balanced_model.predict(data["X_test"])
            balanced_train_error = mean_squared_error(data["y_train"], balanced_train_pred)
            balanced_test_error = mean_squared_error(data["y_test"], balanced_test_pred)
            
            results_df = pd.DataFrame({
                "Model": ["Simple (Linear)", "Complex (Deep Tree)", "Balanced (Random Forest)"],
                "Training MSE": [simple_train_error, complex_train_error, balanced_train_error],
                "Test MSE": [simple_test_error, complex_test_error, balanced_test_error],
                "Gap (Variance)": [
                    abs(simple_test_error - simple_train_error),
                    abs(complex_test_error - complex_train_error),
                    abs(balanced_test_error - balanced_train_error)
                ]
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Training MSE", x=results_df["Model"], y=results_df["Training MSE"], marker_color='skyblue'))
            fig.add_trace(go.Bar(name="Test MSE", x=results_df["Model"], y=results_df["Test MSE"], marker_color='salmon'))
            fig.update_layout(barmode='group', title="Bias-Variance Trade-off: Training vs Test Error", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=results_df["Model"], y=results_df["Gap (Variance)"], marker_color='mediumseagreen'))
            fig2.update_layout(title="Model Variance (Overfitting Indicator)", yaxis_title="Error Gap", height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    with tab8:
        model_choice = st.selectbox("Select Model for Prediction", ["Random Forest", "Gradient Boosting", "MLP Regressor"], index=0)
        
        if model_choice == "Random Forest":
            model_info = {
                "model": reg_results["Random Forest"]["model"],
                "encoders": data["encoders"],
                "df_clean": data["df_clean"],
            }
        elif model_choice == "Gradient Boosting":
            model_info = {
                "model": reg_results["Gradient Boosting"]["model"],
                "encoders": data["encoders"],
                "df_clean": data["df_clean"],
            }
        else:
            model_info = {
                "model": reg_results["MLP Regressor"]["model"],
                "encoders": data["encoders"],
                "df_clean": data["df_clean"],
            }
        
        render_prediction_form(model_info)
        
        # PCA Analysis
        st.divider()
        st.subheader("📊 PCA Analysis")
        # Use the same clean data as clustering
        X_for_pca = pd.concat([data["X_train"], data["X_test"]], axis=0)
        if X_for_pca.isnull().any().any():
            X_for_pca = X_for_pca.fillna(X_for_pca.median())
        X_scaled_pca = StandardScaler().fit_transform(X_for_pca.values)
        render_pca_analysis(X_scaled_pca)


if __name__ == "__main__":
    main()
