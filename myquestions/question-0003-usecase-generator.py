import pandas as pd
import numpy as np
import random
 
 
def generar_caso_de_uso_analizar_demanda_bus():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función analizar_demanda_bus(df, ruta_col, hora_col, pasajeros_col).
    """
 
    # 1. Configuración aleatoria
    n_rutas = random.randint(3, 7)
    rutas = [f"Ruta_{i+1:02d}" for i in range(n_rutas)]
    horas = list(range(0, 24))           # 0..23
    dias  = random.randint(5, 15)        # días de registro
 
    ruta_col      = "ruta"
    hora_col      = "hora"
    pasajeros_col = "pasajeros"
 
    # 2. Generar registros: una fila por (ruta, hora, día)
    registros = []
    for _ in range(dias):
        for ruta in rutas:
            for hora in horas:
                # Más pasajeros en horas pico
                if hora in [6, 7, 8, 9, 17, 18, 19, 20]:
                    pax = int(np.random.normal(loc=random.randint(200, 400), scale=40))
                else:
                    pax = int(np.random.normal(loc=random.randint(40, 120), scale=20))
                pax = max(0, pax)
                registros.append({ruta_col: ruta, hora_col: hora, pasajeros_col: pax})
 
    df = pd.DataFrame(registros)
 
    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------
    input_data = {
        "df":           df.copy(),
        "ruta_col":     ruta_col,
        "hora_col":     hora_col,
        "pasajeros_col": pasajeros_col,
    }
 
    # ---------------------------------------------------------
    # OUTPUT esperado (replicamos la lógica de la función)
    # ---------------------------------------------------------
    HORAS_PICO = [6, 7, 8, 9, 17, 18, 19, 20]
 
    # A. Clasificar tipo_hora
    df_work = df.copy()
    df_work["tipo_hora"] = df_work[hora_col].apply(
        lambda h: "pico" if h in HORAS_PICO else "valle"
    )
 
    # B. Promedio de pasajeros por (ruta, tipo_hora)
    # C. Pivot table
    pivot = pd.pivot_table(
        df_work,
        values=pasajeros_col,
        index=ruta_col,
        columns="tipo_hora",
        aggfunc="mean",
    )
 
    # D. Ratio pico/valle
    pivot["ratio_pico_valle"] = pivot["pico"] / pivot["valle"]
 
    # E. Ordenar y resetear índice
    output_data = (
        pivot.sort_values("ratio_pico_valle", ascending=False)
        .reset_index()
    )
 
    return input_data, output_data
 
 
# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_analizar_demanda_bus()
 
    print("=== INPUT ===")
    print(f"ruta_col      : {entrada['ruta_col']}")
    print(f"hora_col      : {entrada['hora_col']}")
    print(f"pasajeros_col : {entrada['pasajeros_col']}")
    print(f"Filas en df   : {len(entrada['df'])}")
    print("DataFrame (primeras 10 filas):")
    print(entrada["df"].head(10))
 
    print("\n=== OUTPUT ESPERADO ===")
    print(salida_esperada)
