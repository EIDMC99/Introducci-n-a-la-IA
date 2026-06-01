import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def procesar_observaciones_estelares(df, columna_objetivo):
    X = df.drop(columns=[columna_objetivo])
    y = df[columna_objetivo].to_numpy()
    imputador = SimpleImputer(strategy="mean")
    X_imputada = imputador.fit_transform(X)
    escalador = StandardScaler()
    X_escalada = escalador.fit_transform(X_imputada)
    return (X_escalada, y)

    print(f"\n¿X procesada correcta?: {'✅ SÍ' if X_ok else '❌ NO'}")
    print(f"¿y correcto?          : {'✅ SÍ' if y_ok else '❌ NO'}")
    print(f"\n¿El resultado es correcto?: {'✅ SÍ' if X_ok and y_ok else '❌ NO'}")
