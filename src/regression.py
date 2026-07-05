import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import os

from src.preprocessing import get_inverter_cols

def prepare_regression_data(df, inv_id=22):
    cols = get_inverter_cols(df, inv_id)

    df_active = df[df[cols['dc_current']] > 0.5].copy()

    X = df_active[[
        cols['dc_current'],
        cols['dc_voltage'],
        cols['ac_current'],
        cols['ac_voltage']
    ]]
    y = df_active[cols['ac_power']]

    X = X.rename(columns={
        cols['dc_current']: 'dc_current',
        cols['dc_voltage']: 'dc_voltage',
        cols['ac_current']: 'ac_current',
        cols['ac_voltage']: 'ac_voltage'
    })

    return X, y

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0.1
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def train_and_compare_regressors(X, y):
    print("Dividiendo datos de regresión (80% entrenamiento, 20% prueba)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Regresión Lineal': LinearRegression(),
        'XGBoost': XGBRegressor(random_state=42, n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=50, max_depth=8, n_jobs=-1)
    }

    metrics_list = []
    trained_models = {}
    predictions = {}

    for name, model in models.items():
        print(f"Entrenando {name}...")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = root_mean_squared_error(y_test, y_test_pred)
        mape = calculate_mape(y_test, y_test_pred)

        r2_tr = r2_score(y_train, y_train_pred)
        mae_tr = mean_absolute_error(y_train, y_train_pred)
        rmse_tr = root_mean_squared_error(y_train, y_train_pred)
        mape_tr = calculate_mape(y_train, y_train_pred)

        metrics_list.append({
            'Modelo': name,
            'Train R2': r2_tr,
            'Train MAE (kW)': mae_tr,
            'Train RMSE (kW)': rmse_tr,
            'Train MAPE (%)': mape_tr,
            'Test R2': r2,
            'Test MAE (kW)': mae,
            'Test RMSE (kW)': rmse,
            'Test MAPE (%)': mape
        })

        trained_models[name] = model
        predictions[name] = y_test_pred

    print("\n--- Validación Cruzada y Ajuste de Hiperparámetros con RandomizedSearchCV ---")

    cv_scores_lr = cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"Regresión Lineal - CV R²: {cv_scores_lr.mean():.4f} ± {cv_scores_lr.std():.4f}")

    print("Buscando mejores hiperparámetros para XGBoost con RandomizedSearchCV...")
    xgb_param_dist = {
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [3, 4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 1.0]
    }
    rs_xgb = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        xgb_param_dist,
        n_iter=20,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rs_xgb.fit(X_train, y_train)
    print(f"Mejores parámetros XGBoost: {rs_xgb.best_params_}")
    print(f"Mejor CV R² XGBoost: {rs_xgb.best_score_:.4f}")

    cv_scores_xgb_tuned = cross_val_score(rs_xgb.best_estimator_, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print(f"XGBoost Tuneado - CV R²: {cv_scores_xgb_tuned.mean():.4f} ± {cv_scores_xgb_tuned.std():.4f}")

    tuned_name = 'XGBoost (Tuneado)'
    xgb_tuned_pred = rs_xgb.predict(X_test)
    metrics_list.append({
        'Modelo': tuned_name,
        'Mejores Params': str(rs_xgb.best_params_),
        'Train R2': rs_xgb.score(X_train, y_train),
        'Train MAE (kW)': mean_absolute_error(y_train, rs_xgb.predict(X_train)),
        'Train RMSE (kW)': root_mean_squared_error(y_train, rs_xgb.predict(X_train)),
        'Train MAPE (%)': calculate_mape(y_train, rs_xgb.predict(X_train)),
        'Test R2': r2_score(y_test, xgb_tuned_pred),
        'Test MAE (kW)': mean_absolute_error(y_test, xgb_tuned_pred),
        'Test RMSE (kW)': root_mean_squared_error(y_test, xgb_tuned_pred),
        'Test MAPE (%)': calculate_mape(y_test, xgb_tuned_pred)
    })

    trained_models[tuned_name] = rs_xgb.best_estimator_
    predictions[tuned_name] = xgb_tuned_pred

    df_metrics = pd.DataFrame(metrics_list)

    base_dir = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
    tables_dir = os.path.join(base_dir, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(tables_dir, "supervised_regression_metrics.csv"), index=False)
    print(f"\nMétricas de regresión guardadas en {os.path.join(tables_dir, 'supervised_regression_metrics.csv')}")

    return df_metrics, X_test, y_test, predictions
