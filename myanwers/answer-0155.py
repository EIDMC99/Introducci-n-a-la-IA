import numpy as np
from sklearn.decomposition import PCA

def pca_varianza_minima(X, varianza_objetivo=0.90):
    # 1. Convertir a array por si viene como otra estructura
    X = np.array(X)

    # 2. Configurar PCA para conservar la varianza objetivo
    modelo = PCA(n_components=varianza_objetivo)

    # 3. Ajustar y transformar
    X_transformado = modelo.fit_transform(X)

    # 4. Obtener cuántos componentes fueron necesarios
    n_componentes = modelo.n_components_

    return (X_transformado, n_componentes)


# --- (Aquí ya tienes pegado el generador generar_caso_de_uso_pca_varianza_minima) ---

if __name__ == "__main__":
    # 1. Obtener datos de prueba
    entrada, salida_esperada = generar_caso_de_uso_pca_varianza_minima()

    # 2. Llamar a la función
    resultado = pca_varianza_minima(entrada['X'], entrada['varianza_objetivo'])

    X_resultado, n_resultado = resultado
    X_esperado, n_esperado   = salida_esperada

    # 3. Mostrar resultados
    print(f"Varianza objetivo      : {entrada['varianza_objetivo']:.4f}")
    print(f"Shape X original       : {entrada['X'].shape}")
    print(f"\nShape resultado        : {X_resultado.shape}")
    print(f"Shape esperado         : {X_esperado.shape}")
    print(f"\nn_componentes resultado: {n_resultado}")
    print(f"n_componentes esperado : {n_esperado}")

    # 4. Verificación
    shape_ok = X_resultado.shape == X_esperado.shape
    n_ok     = n_resultado == n_esperado
    vals_ok  = np.allclose(np.abs(X_resultado), np.abs(X_esperado))  # abs por signo de componentes

    print(f"\n¿Shape correcto?       : {'✅ SÍ' if shape_ok else '❌ NO'}")
    print(f"¿n_componentes correcto?: {'✅ SÍ' if n_ok else '❌ NO'}")
    print(f"¿Valores correctos?    : {'✅ SÍ' if vals_ok else '❌ NO'}")
    print(f"\n¿El resultado es correcto?: {'✅ SÍ' if shape_ok and n_ok and vals_ok else '❌ NO'}")
