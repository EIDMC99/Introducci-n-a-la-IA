import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_ferrocarriles():
    lineas = ['Metro A', 'Cercanías B', 'AVE Norte', 'Feve']
    # Crear datos aleatorios incluyendo nulos
    data = {
        'linea': [random.choice(lineas) for _ in range(20)],
        'retraso': [random.choice([0, 2, 5, 10, 15, 20, None]) for _ in range(20)]
    }
    df_in = pd.DataFrame(data)
    umbral = random.uniform(5.0, 12.0)
    
    # Lógica interna para calcular el output esperado
    df_clean = df_in.dropna(subset=['retraso']).copy()
    df_clean['puntual'] = df_clean['retraso'] < umbral
    
    output = df_clean.groupby('linea').agg(
        tasa_puntualidad=('puntual', 'mean'),
        retraso_promedio=('retraso', 'mean')
    ).reset_index()
    
    output['supera_mediana'] = output['retraso_promedio'] > output['retraso_promedio'].median()
    output = output.sort_values(by='tasa_puntualidad', ascending=False)
    
    input_dict = {
        "df": df_in,
        "linea_col": "linea",
        "retraso_col": "retraso",
        "umbral_min": umbral
    }
    
    return input_dict, output
