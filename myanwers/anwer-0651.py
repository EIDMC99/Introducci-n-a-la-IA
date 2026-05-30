import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def procesar_datos_transporte(df, target_col):
    df = df.copy()

    # 1. Eliminar outliers con IQR (solo columnas numéricas, excepto target)
    numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_iqr = [c for c in numericas if c != target_col]
    
    for col in cols_iqr:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # 2. Crear variable costo_por_km
    df["costo_por_km"] = df["costo"] / df["distancia"]

    # 3. Normalizar variables numéricas con MinMaxScaler (excepto target)
    cols_escalar = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    scaler = MinMaxScaler()
    df[cols_escalar] = scaler.fit_transform(df[cols_escalar])

    # 4. Calcular correlación con variable objetivo
    corr = df.corr()[target_col].abs()

    # 5. Seleccionar variables con correlación > 0.3
    columnas_seleccionadas = corr[corr > 0.3].index.tolist()

    return df[columnas_seleccionadas]


# --- (Aquí está el generador) ---

if __name__ == "__main__":
    # 1. Obtenemos los datos de prueba del generador
    entrada, salida_esperada = generar_caso_de_uso_procesar_datos_transporte()

    # 2. Llamamos a la función con los datos del generador
    resultado = procesar_datos_transporte(entrada["df"], entrada["target_col"])

    # 3. Mostramos resultados
    print("--- DataFrame resultante ---")
    print(resultado.head())

    print(f"\nColumnas resultado : {sorted(resultado.columns.tolist())}")
    print(f"Columnas esperadas : {sorted(salida_esperada.columns.tolist())}")
    print(f"Filas resultado    : {len(resultado)} | Filas esperadas: {len(salida_esperada)}")

    # 4. Verificación
    cols_ok = sorted(resultado.columns.tolist()) == sorted(salida_esperada.columns.tolist())
    filas_ok = len(resultado) == len(salida_esperada)

    print(f"\n¿Columnas correctas?: {'✅ SÍ' if cols_ok else 'NO'}")
    print(f"¿Filas correctas?   : {'✅ SÍ' if filas_ok else 'NO'}")
    print(f"\n¿El resultado es correcto?: {'✅ SÍ' if cols_ok and filas_ok else 'NO'}")
