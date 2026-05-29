"""
modeling.py
Reusable modeling pipeline for ACIS claim severity prediction and premium optimization.

Includes:
- Data preparation (feature engineering, encoding, split)
- Model training (Linear Regression, Random Forest, XGBoost)
- Evaluation (RMSE, R²)
- SHAP interpretability
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


def prepare_modeling_data(df, target_col='totalclaims', min_claim=0, test_size=0.2, random_state=42):
    """
    Prepare data for modeling: feature engineering, encoding, train/test split.

    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned insurance data
    target_col : str
        Target variable to predict
    min_claim : float
        If > 0, filter to rows where target > min_claim (for severity modeling)
    test_size : float
        Proportion for test set
    random_state : int
        Reproducibility seed

    Returns:
    --------
    dict with X_train, X_test, y_train, y_test, feature_names, encoder_info
    """
    df_model = df.copy()

    # Filter for severity modeling (only policies with claims > 0)
    if min_claim > 0:
        df_model = df_model[df_model[target_col] > min_claim].copy()
        print(f"Severity modeling: filtered to {len(df_model):,} rows with {target_col} > {min_claim}")

    # FEATURE ENGINEERING

    # Vehicle age (as of 2015, the end of dataset period)
    df_model['vehicle_age'] = 2015 - df_model['registrationyear']
    df_model['vehicle_age'] = df_model['vehicle_age'].clip(lower=0, upper=50)

    # Policy duration (months since first transaction for this policy)
    policy_start = df_model.groupby('policyid')['transactionmonth'].transform('min')
    df_model['policy_duration_months'] = ((df_model['transactionmonth'] - policy_start).dt.days / 30.44).round().astype(int)
    df_model['policy_duration_months'] = df_model['policy_duration_months'].clip(lower=0)

    # Vehicle power per cylinder (efficiency proxy)
    df_model['power_per_cylinder'] = np.where(
        df_model['cylinders'].astype(float) > 0,
        df_model['kilowatts'] / df_model['cylinders'].astype(float),
        0
    )

    # Is new vehicle flag
    df_model['is_new_vehicle'] = (df_model['newvehicle'] == 'Yes').astype(int)

    # Has tracking device
    df_model['has_tracking'] = (df_model['trackingdevice'] == 'Yes').astype(int)

    # Has alarm/immobiliser
    df_model['has_alarm'] = (df_model['alarmimmobiliser'] == 'Yes').astype(int)

    # Premium per term ratio
    df_model['premium_term_ratio'] = np.where(
        df_model['calculatedpremiumperterm'] > 0,
        df_model['totalpremium'] / df_model['calculatedpremiumperterm'],
        0
    )

    # SELECT FEATURES

    # Numerical features
    numerical_features = [
        'vehicle_age', 'policy_duration_months', 'cubiccapacity', 'kilowatts',
        'suminsured', 'calculatedpremiumperterm', 'customvalueestimate',
        'power_per_cylinder', 'premium_term_ratio',
        'is_new_vehicle', 'has_tracking', 'has_alarm'
    ]

    # Only include features that exist in the dataframe
    numerical_features = [f for f in numerical_features if f in df_model.columns]

    # Categorical features (high-cardinality ones excluded to prevent overfitting)
    categorical_features = [
        'province', 'vehicletype', 'gender', 'covercategory'
    ]
    categorical_features = [f for f in categorical_features if f in df_model.columns]

    # HANDLE MISSING VALUES

    for col in numerical_features:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    for col in categorical_features:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna('Unknown')

    # ENCODE CATEGORICALS

    encoder_info = {}
    X = df_model[numerical_features].copy()

    for col in categorical_features:
        le = LabelEncoder()
        df_model[col + '_encoded'] = le.fit_transform(df_model[col].astype(str))
        X[col + '_encoded'] = df_model[col + '_encoded']
        encoder_info[col] = le

    # Final feature list
    feature_names = list(X.columns)

    # Target
    y = df_model[target_col].copy()

    # TRAIN/TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train set: {len(X_train):,} rows")
    print(f"Test set:  {len(X_test):,} rows")
    print(f"Features:  {len(feature_names)} ({len(numerical_features)} numerical, {len(categorical_features)} categorical)")

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'encoder_info': encoder_info,
        'df_model': df_model
    }


def train_models(X_train, y_train, X_test, y_test, feature_names):
    """
    Train and evaluate three models: Linear Regression, Random Forest, XGBoost.

    Returns:
    --------
    dict with trained models and metrics
    """
    results = {}

    # 1. LINEAR REGRESSION (Baseline)
    print("\n>>> Training Linear Regression...")
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    r2_lr = r2_score(y_test, y_pred_lr)

    results['LinearRegression'] = {
        'model': lr,
        'rmse': rmse_lr,
        'r2': r2_lr,
        'predictions': y_pred_lr
    }
    print(f"    RMSE: {rmse_lr:,.2f} | R²: {r2_lr:.4f}")

    # 2. RANDOM FOREST
    print("\n>>> Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    results['RandomForest'] = {
        'model': rf,
        'rmse': rmse_rf,
        'r2': r2_rf,
        'predictions': y_pred_rf,
        'feature_importance': dict(zip(feature_names, rf.feature_importances_))
    }
    print(f"    RMSE: {rmse_rf:,.2f} | R²: {r2_rf:.4f}")

    # 3. XGBOOST
    print("\n>>> Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    r2_xgb = r2_score(y_test, y_pred_xgb)

    results['XGBoost'] = {
        'model': xgb_model,
        'rmse': rmse_xgb,
        'r2': r2_xgb,
        'predictions': y_pred_xgb,
        'feature_importance': dict(zip(feature_names, xgb_model.feature_importances_))
    }
    print(f"    RMSE: {rmse_xgb:,.2f} | R²: {r2_xgb:.4f}")

    return results


def create_comparison_table(results):
    """
    Create a DataFrame comparing all models.
    """
    rows = []
    for name, res in results.items():
        rows.append({
            'Model': name,
            'RMSE': f"{res['rmse']:,.2f}",
            'R²': f"{res['r2']:.4f}",
            'Rank': 0  # Will fill after sorting
        })

    df = pd.DataFrame(rows)
    # Sort by RMSE (lower is better)
    df['RMSE_raw'] = df['RMSE'].str.replace(',', '').astype(float)
    df = df.sort_values('RMSE_raw')
    df['Rank'] = range(1, len(df) + 1)
    df = df.drop('RMSE_raw', axis=1)

    return df[['Rank', 'Model', 'RMSE', 'R²']]


def plot_feature_importance(results, model_name='RandomForest', top_n=10):
    """
    Plot top N feature importances from a tree-based model.
    """
    if model_name not in results or 'feature_importance' not in results[model_name]:
        print(f"Feature importance not available for {model_name}")
        return None

    importances = results[model_name]['feature_importance']
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

    features = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(features[::-1], values[::-1], color='steelblue', alpha=0.8)
    ax.set_xlabel('Importance', fontsize=11)
    ax.set_title(f'Top {top_n} Features - {model_name}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig


def explain_with_shap(model, X_sample, feature_names, max_display=10):
    """
    Generate SHAP summary plot for model interpretability.

    Parameters:
    -----------
    model : trained model
    X_sample : pd.DataFrame or np.array
        Sample of data for SHAP computation (use subset for speed)
    feature_names : list
        Feature names
    max_display : int
        Number of top features to show

    Returns:
    --------
    shap_values, explainer
    """
    import shap

    print("\n>>> Computing SHAP values (this may take a minute)...")

    # Use TreeExplainer for tree-based models, KernelExplainer for others
    if hasattr(model, 'tree_method') or isinstance(model, RandomForestRegressor):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict, X_sample)

    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_sample, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.title('SHAP Feature Importance (Impact on Claim Severity)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return shap_values, explainer


def calculate_premium_formula(claim_prob, predicted_severity, expense_loading=50, profit_margin=0.15):
    """
    Calculate premium using the risk-based pricing formula.

    Premium = (P(claim) × Predicted Severity) + Expense Loading + Profit Margin

    Parameters:
    -----------
    claim_prob : float
        Probability of claim (0 to 1)
    predicted_severity : float
        Expected claim amount if claim occurs
    expense_loading : float
        Fixed administrative cost per policy
    profit_margin : float
        Percentage markup on expected claims cost

    Returns:
    --------
    float : Recommended premium
    """
    expected_claim_cost = claim_prob * predicted_severity
    premium = expected_claim_cost * (1 + profit_margin) + expense_loading
    return premium