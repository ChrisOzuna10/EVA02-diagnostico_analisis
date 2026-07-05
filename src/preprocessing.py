import pandas as pd
import numpy as np

def get_inverter_cols(df, inv_id):
    inv_str = f"inv_{inv_id:02d}"
    cols = df.columns

    dc_current = [c for c in cols if c.startswith(f"{inv_str}_dc_current")][0]
    ac_current = [c for c in cols if c.startswith(f"{inv_str}_ac_current")][0]
    ac_voltage = [c for c in cols if c.startswith(f"{inv_str}_ac_voltage")][0]

    ac_power_candidates = [c for c in cols if c.startswith(f"{inv_str}_ac_power") or c.startswith(f"{inv_str}_ac_power_iinv")]
    ac_power = ac_power_candidates[0]

    dc_voltage_candidates = [c for c in cols if c.startswith(f"{inv_str}_dc_voltage")]
    dc_voltage = dc_voltage_candidates[0] if len(dc_voltage_candidates) > 0 else None

    return {
        'dc_current': dc_current,
        'dc_voltage': dc_voltage,
        'ac_current': ac_current,
        'ac_voltage': ac_voltage,
        'ac_power': ac_power
    }

def clean_and_impute(df):
    print("Iniciando limpieza e imputación de datos...")
    df_clean = df.copy()

    old_typo_col = "inv_15_ac_power_iinv_149653"
    new_correct_col = "inv_15_ac_power_inv_149653"
    if old_typo_col in df_clean.columns:
        df_clean.rename(columns={old_typo_col: new_correct_col}, inplace=True)
        print(f"Columna renombrada: {old_typo_col} -> {new_correct_col}")

    null_counts_before = df_clean.isnull().sum().sum()
    if null_counts_before > 0:
        df_clean.ffill(inplace=True)
        df_clean.bfill(inplace=True)
        print(f"Imputadas {null_counts_before} celdas con valores nulos via ffill/bfill.")

    dc_voltage_cols = [c for c in df_clean.columns if '_dc_voltage_' in c]
    mean_dc_voltage = df_clean[dc_voltage_cols].mean(axis=1)

    new_inv05_voltage_col = "inv_05_dc_voltage_inv_149600"
    df_clean[new_inv05_voltage_col] = mean_dc_voltage.astype('float32')
    print(f"Imputado voltaje DC faltante del Inversor 05 como promedio de los otros {len(dc_voltage_cols)} inversores.")

    null_counts_after = df_clean.isnull().sum().sum()
    print(f"Limpieza completada. Valores nulos restantes: {null_counts_after}")

    return df_clean

def feature_engineering(df):
    print("Realizando ingeniería de características...")
    df_feat = df.copy()

    for i in range(1, 25):
        cols = get_inverter_cols(df_feat, i)

        dc_i = df_feat[cols['dc_current']]
        dc_v = df_feat[cols['dc_voltage']]
        ac_p = df_feat[cols['ac_power']]

        dc_power_kw = (dc_i * dc_v) / 1000.0
        df_feat[f"inv_{i:02d}_dc_power_kW"] = dc_power_kw.astype('float32')

        efficiency = np.where(
            dc_power_kw > 0.01,
            (ac_p / dc_power_kw) * 100.0,
            0.0
        )
        df_feat[f"inv_{i:02d}_efficiency"] = np.clip(efficiency, 0.0, 100.0).astype('float32')

    print("Ingeniería de características completada.")
    return df_feat

def filter_daylight(df):
    dc_current_cols = [c for c in df.columns if '_dc_current_' in c]
    mean_dc_current = df[dc_current_cols].mean(axis=1)

    is_daylight = mean_dc_current > 0.5
    df_daylight = df[is_daylight]

    print(f"Filtrado a horas diurnas: {df_daylight.shape[0]:,} filas (de {df.shape[0]:,} originales).")
    return df_daylight
