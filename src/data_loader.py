import pandas as pd
import numpy as np
import os
import time

def load_data(file_path):
    start_time = time.time()
    print(f"Cargando datos desde {os.path.basename(file_path)}...")

    df_head = pd.read_csv(file_path, nrows=2)

    dtypes = {}
    for col in df_head.columns:
        if col != 'measured_on':
            dtypes[col] = 'float32'

    df = pd.read_csv(file_path, dtype=dtypes)

    df['measured_on'] = pd.to_datetime(df['measured_on'])
    df.set_index('measured_on', inplace=True)

    elapsed = time.time() - start_time
    mem_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"Cargadas {df.shape[0]:,} filas y {df.shape[1]} columnas en {elapsed:.2f} segundos.")
    print(f"Uso de memoria del DataFrame: {mem_usage:.2f} MB")

    return df
