import SimpleImputer
from sklearn.preprocessing import StandardScaler

def procesar_observaciones_estelares(df, columna_objetivo):
    # 1. Separar X e y
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo].to_numpy()

    # 2. Imputar valores faltantes con la media de cada columna
    imputador = SimpleImputer(strategy="mean")
    X_imputada = imputador.fit_transform(X)

    # 3. Escalar a media 0 y desviación estándar 1
    escalador = StandardScaler()
    X_escalada = escalador.fit_transform(X_imputada)

    return (X_escalada, y)


# --- (Aquí ya tienes pegado el generador generar_caso_de_uso_procesar_observaciones_estelares) ---

if __name__ == "__main__":
    # 1. Obtener datos de prueba
    entrada, salida_esperada = generar_caso_de_uso_procesar_observaciones_estelares()
    X_esp, y_esp = salida_esperada

    # 2. Llamar a la función
    X_res, y_res = procesar_observaciones_estelares(
        entrada["df"],
        entrada["columna_objetivo"]
    )

    # 3. Mostrar resultados
    print(f"columna_objetivo : {entrada['columna_objetivo']}")
    print(f"\nShape X resultado : {X_res.shape} | Shape X esperado : {X_esp.shape}")
    print(f"Shape y resultado : {y_res.shape} | Shape y esperado : {y_esp.shape}")
    print(f"\nPrimeras 3 filas X procesada:\n{X_res[:3]}")
    print(f"\ny resultado : {y_res}")
    print(f"y esperado  : {y_esp}")

    # 4. Verificación
    X_ok = np.allclose(X_res, X_esp)
    y_ok = np.array_equal(y_res, y_esp)

    print(f"\n¿X procesada correcta?: {'✅ SÍ' if X_ok else '❌ NO'}")
    print(f"¿y correcto?          : {'✅ SÍ' if y_ok else '❌ NO'}")
    print(f"\n¿El resultado es correcto?: {'✅ SÍ' if X_ok and y_ok else '❌ NO'}")
