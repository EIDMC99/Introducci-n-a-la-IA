import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
 
def generar_caso_de_uso_segmentar_riesgo_barcos():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función segmentar_riesgo_barcos(X, k_values).
    """
 
    # 1. Configuración aleatoria
    n_barcos    = random.randint(40, 120)
    k_min       = random.randint(2, 3)
    k_max       = random.randint(5, 8)
    k_values    = list(range(k_min, k_max + 1))
    random_state = 42
 
    # 2. Generar X con 4 columnas operativas:
    #    tonelaje (t), antigüedad (años), inspecciones_fallidas, velocidad_promedio (nudos)
    # Creamos clústeres sintéticos para que el silhouette sea significativo
    n_clusters_true = random.randint(k_min, k_max)
    centers = np.random.uniform(
        low=[1000, 1, 0, 5],
        high=[50000, 40, 10, 25],
        size=(n_clusters_true, 4),
    )
    labels_true = np.random.randint(0, n_clusters_true, size=n_barcos)
    noise = np.random.randn(n_barcos, 4) * np.array([3000, 3, 1, 2])
    X = centers[labels_true] + noise
    # Aseguramos valores no negativos
    X = np.clip(X, a_min=[0, 0, 0, 1], a_max=None)
 
    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------
    input_data = {
        "X":        X.copy(),
        "k_values": k_values,
    }
 
    # ---------------------------------------------------------
    # OUTPUT esperado (replicamos la lógica de la función)
    # ---------------------------------------------------------
 
    # A. Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
 
    # B. Calcular silhouette para cada k
    silhouette_scores = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
 
    # C. k óptimo (mayor silhouette)
    best_idx = np.argmax(silhouette_scores)
    output_data = int(k_values[best_idx])
 
    return input_data, output_data
 
 
# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_segmentar_riesgo_barcos()
 
    print("=== INPUT ===")
    print(f"Shape de X : {entrada['X'].shape}")
    print(f"k_values   : {entrada['k_values']}")
    print("Primeras 5 filas de X (tonelaje, antigüedad, insp. fallidas, velocidad):")
    print(entrada["X"][:5].round(2))
 
    print("\n=== OUTPUT ESPERADO ===")
    print(f"k óptimo: {salida_esperada}")
