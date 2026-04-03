import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def generar_caso_de_uso_aviones():
    # Generar matriz aleatoria con NaNs (25 filas para que cv=5 no falle)
    X = np.random.rand(25, 4) * 100
    mask = np.random.choice([True, False], size=X.shape, p=[0.1, 0.9])
    X[mask] = np.nan
    y = np.random.randint(0, 2, 25)
    
    # Lógica interna para calcular el output esperado
    imp = SimpleImputer(strategy='median')
    X_filled = imp.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    output = cross_val_score(clf, X_scaled, y, cv=5, scoring='f1')
    
    input_dict = {"X": X, "y": y}
    
    return input_dict, output
