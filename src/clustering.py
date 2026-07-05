import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os

from src.preprocessing import get_inverter_cols

def extract_inverter_features(df_daylight):
    print("Extrayendo características por inversor para clustering...")
    inverter_features = []

    for i in range(1, 25):
        cols = get_inverter_cols(df_daylight, i)

        dc_i = df_daylight[cols['dc_current']]
        ac_p = df_daylight[cols['ac_power']]
        eff = df_daylight[f"inv_{i:02d}_efficiency"]

        median_eff = float(eff.median())
        mean_eff = float(eff.mean())
        std_eff = float(eff.std())

        active_daylight = dc_i > 1.0
        outage_count = ((active_daylight) & (ac_p < 0.1)).sum()
        total_active = active_daylight.sum()
        outage_rate = float(outage_count / total_active) if total_active > 0 else 0.0

        max_ac_power = float(ac_p.max())

        inverter_features.append({
            'inverter_id': i,
            'median_efficiency': median_eff,
            'mean_efficiency': mean_eff,
            'std_efficiency': std_eff,
            'outage_rate': outage_rate,
            'max_ac_power': max_ac_power
        })

    df_inv = pd.DataFrame(inverter_features)
    print(f"Características extraídas para {df_inv.shape[0]} inversores.")
    return df_inv

def run_kmeans_analysis(df_inv, max_k=6):
    print("Ejecutando búsqueda de parámetros K-Means (Codo & Silueta)...")

    features = ['median_efficiency', 'std_efficiency', 'outage_rate']
    X = df_inv[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {
        'k_values': [],
        'inertia': [],
        'silhouette': []
    }

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        results['k_values'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X_scaled, labels))

    return results, X_scaled

def perform_final_clustering(df_inv, X_scaled, n_clusters=3):
    print(f"Ajustando K-Means final con K={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_result = df_inv.copy()
    df_result['cluster'] = labels

    cluster_means = df_result.groupby('cluster')['median_efficiency'].mean().sort_values()

    if n_clusters == 3:
        mapping = {
            cluster_means.index[0]: "Crítico (Sensor/Falla)",
            cluster_means.index[1]: "Sub-óptimo (Pérdidas/Estrés)",
            cluster_means.index[2]: "Óptimo (Saludable)"
        }
    else:
        mapping = {}
        for rank, c_id in enumerate(cluster_means.index):
            mapping[c_id] = f"Nivel {rank}"

    df_result['health_status'] = df_result['cluster'].map(mapping)

    base_dir = ".." if os.path.basename(os.getcwd()) == "notebooks" else "."
    tables_dir = os.path.join(base_dir, "output", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    df_result.to_csv(os.path.join(tables_dir, "inverter_health_clustering.csv"), index=False)
    print(f"Resultados de clustering guardados en {os.path.join(tables_dir, 'inverter_health_clustering.csv')}")

    return df_result
