import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import os

from src.preprocessing import get_inverter_cols

def prepare_forecasting_data(df):
    print("Agregando potencia AC total de la planta...")

    ac_cols = []
    for i in range(1, 25):
        cols = get_inverter_cols(df, i)
        ac_cols.append(cols['ac_power'])

    df_power = pd.DataFrame(index=df.index)
    df_power['total_ac_power'] = df[ac_cols].sum(axis=1)

    print("Remuestreando a frecuencia horaria...")
    df_hourly = df_power.resample('1h').mean()

    df_hourly = df_hourly.interpolate(method='time')

    print("Generando características de rezago y estadísticas móviles...")
    df_feat = df_hourly.copy()

    for lag in [1, 2, 24, 48]:
        df_feat[f'lag_{lag}'] = df_feat['total_ac_power'].shift(lag)

    df_feat['rolling_mean_3'] = df_feat['total_ac_power'].shift(1).rolling(window=3).mean()
    df_feat['rolling_mean_24'] = df_feat['total_ac_power'].shift(1).rolling(window=24).mean()
    df_feat['rolling_std_3'] = df_feat['total_ac_power'].shift(1).rolling(window=3).std()

    df_feat['hour'] = df_feat.index.hour
    df_feat['month'] = df_feat.index.month
    df_feat['day_of_week'] = df_feat.index.dayofweek
    df_feat['day_of_year'] = df_feat.index.dayofyear

    df_feat.dropna(inplace=True)

    return df_feat

def run_forecasting_analysis(df_feat, test_days=30):
    print(f"Dividiendo datos de forecasting (Prueba: últimos {test_days} días)...")

    split_date = df_feat.index.max() - pd.Timedelta(days=test_days)

    train = df_feat.loc[df_feat.index < split_date]
    test = df_feat.loc[df_feat.index >= split_date]

    features = [c for c in df_feat.columns if c != 'total_ac_power']
    target = 'total_ac_power'

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    print("Evaluando Modelo de Persistencia Base...")
    y_test_pred_base = X_test['lag_24']

    print("Entrenando Regresión Lineal con Rezagos...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_test_pred_lr = lr_model.predict(X_test)

    print("Buscando mejores hiperparámetros para XGBoost Forecaster con RandomizedSearchCV...")
    xgb_param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    rs_xgb = RandomizedSearchCV(
        XGBRegressor(random_state=42, n_jobs=-1),
        xgb_param_dist,
        n_iter=15,
        cv=2,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    rs_xgb.fit(X_train, y_train)
    print(f"Mejores parámetros XGBoost Forecasting: {rs_xgb.best_params_}")
    print(f"Mejor CV RMSE: {-rs_xgb.best_score_:.2f} kW")

    y_test_pred_xgb = rs_xgb.predict(X_test)

    def calculate_metrics(y_true, y_pred):
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mask = y_true > 1.0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return rmse, mae, mape

    base_rmse, base_mae, base_mape = calculate_metrics(y_test, y_test_pred_base)
    lr_rmse, lr_mae, lr_mape = calculate_metrics(y_test, y_test_pred_lr)
    xgb_rmse, xgb_mae, xgb_mape = calculate_metrics(y_test, y_test_pred_xgb)

    metrics = [
        {
            'Modelo': 'Persistencia Base (Lag-24)',
            'RMSE (kW)': base_rmse,
            'MAE (kW)': base_mae,
            'MAPE (%)': base_mape
        },
        {
            'Modelo': 'Regresión Lineal con Rezagos',
            'RMSE (kW)': lr_rmse,
            'MAE (kW)': lr_mae,
            'MAPE (%)': lr_mape
        },
        {
            'Modelo': 'XGBoost (Tuneado)',
            'RMSE (kW)': xgb_rmse,
            'MAE (kW)': xgb_mae,
            'MAPE (%)': xgb_mape
        }
    ]

    df_metrics = pd.DataFrame(metrics)

    base_dir = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
    tables_dir = os.path.join(base_dir, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(tables_dir, "forecasting_metrics.csv"), index=False)
    print(f"\nMétricas de forecasting guardadas en {os.path.join(tables_dir, 'forecasting_metrics.csv')}")

    return df_metrics, train, test, y_test_pred_base, y_test_pred_lr, y_test_pred_xgb
