import numpy as np
import random
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
 
 
def generar_caso_de_uso_predecir_retraso_vuelo():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función predecir_retraso_vuelo(X, y).
    """
 
    # 1. Configuración aleatoria
    n_samples = random.randint(80, 200)
    random_state = 42
 
    # 2. Generar X con 4 columnas operativas:
    #    distancia_km, ocupacion_pct, temperatura_c, viento_kmh
    distancia   = np.random.uniform(200, 5000, size=n_samples)
    ocupacion   = np.random.uniform(40, 100, size=n_samples)
    temperatura = np.random.uniform(-10, 40, size=n_samples)
    viento      = np.random.uniform(0, 120, size=n_samples)
 
    X = np.column_stack([distancia, ocupacion, temperatura, viento])
 
    # Introducir NaNs en X (~10 % de celdas) por fallos de sensor
    nan_mask = np.random.choice([True, False], size=X.shape, p=[0.10, 0.90])
    X[nan_mask] = np.nan
 
    # 3. Generar y (binario: 1 = retraso > 30 min)
    y = np.random.randint(0, 2, size=n_samples)
 
    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------
    input_data = {
        "X": X.copy(),
        "y": y.copy(),
    }
 
    # ---------------------------------------------------------
    # OUTPUT esperado (replicamos la lógica de la función)
    # ---------------------------------------------------------
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(n_estimators=100, random_state=random_state)),
    ])
 
    output_data = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
 
    return input_data, output_data
 
 
# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_predecir_retraso_vuelo()
 
    print("=== INPUT ===")
    print(f"Shape de X : {entrada['X'].shape}")
    print(f"NaNs en X  : {np.isnan(entrada['X']).sum()}")
    print(f"Shape de y : {entrada['y'].shape}")
    print(f"Distribución y: 0={( entrada['y']==0).sum()}  1={(entrada['y']==1).sum()}")
 
    print("\n=== OUTPUT ESPERADO (F1 por fold) ===")
    print(salida_esperada)
    print(f"F1 promedio: {salida_esperada.mean():.4f}")
