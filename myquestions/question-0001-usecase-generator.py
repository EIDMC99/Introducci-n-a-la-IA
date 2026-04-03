import pandas as pd
import numpy as np
import random
 
 
def generar_caso_de_uso_calcular_puntualidad_ferroviaria():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función calcular_puntualidad_ferroviaria(df, linea_col, retraso_col, umbral_min).
    """
 
    # 1. Configuración aleatoria
    n_rows = random.randint(20, 60)
    n_lineas = random.randint(2, 5)
    lineas = [f"Línea_{chr(65 + i)}" for i in range(n_lineas)]  # Línea_A, Línea_B, ...
 
    umbral_min = random.choice([5.0, 10.0, 15.0, 20.0])
 
    linea_col = "linea"
    retraso_col = "retraso_min"
 
    # 2. Construir DataFrame aleatorio
    data = {
        linea_col: random.choices(lineas, k=n_rows),
        "vagones": np.random.randint(3, 12, size=n_rows),
        "pasajeros": np.random.randint(50, 500, size=n_rows),
        retraso_col: np.random.uniform(0, 40, size=n_rows),
    }
    df = pd.DataFrame(data)
 
    # Introducir NaNs en retraso_col (~15% de filas)
    nan_mask = np.random.choice([True, False], size=n_rows, p=[0.15, 0.85])
    df.loc[nan_mask, retraso_col] = np.nan
 
    # ---------------------------------------------------------
    # 3. INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "linea_col": linea_col,
        "retraso_col": retraso_col,
        "umbral_min": umbral_min,
    }
 
    # ---------------------------------------------------------
    # 4. OUTPUT esperado (replicamos la lógica de la función)
    # ---------------------------------------------------------
 
    # A. Eliminar filas con retraso nulo
    df_clean = df.dropna(subset=[retraso_col]).copy()
 
    # B. Columna booleana 'puntual'
    df_clean["puntual"] = df_clean[retraso_col] < umbral_min
 
    # C. Agrupar por línea
    grouped = df_clean.groupby(linea_col).agg(
        tasa_puntualidad=("puntual", "mean"),
        retraso_promedio=(retraso_col, "mean"),
    ).reset_index()
 
    # D. Mediana global con numpy
    mediana_global = np.median(df_clean[retraso_col].values)
    grouped["supera_mediana"] = grouped["retraso_promedio"] > mediana_global
 
    # E. Ordenar de mayor a menor tasa_puntualidad
    output_data = (
        grouped[[linea_col, "tasa_puntualidad", "retraso_promedio", "supera_mediana"]]
        .sort_values("tasa_puntualidad", ascending=False)
        .reset_index(drop=True)
    )
 
    return input_data, output_data
 
 
# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_calcular_puntualidad_ferroviaria()
 
    print("=== INPUT ===")
    print(f"linea_col   : {entrada['linea_col']}")
    print(f"retraso_col : {entrada['retraso_col']}")
    print(f"umbral_min  : {entrada['umbral_min']}")
    print("DataFrame (primeras 8 filas):")
    print(entrada["df"].head(8))
 
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
