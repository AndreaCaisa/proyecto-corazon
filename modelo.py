import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Función que entrena el modelo (después la usará la API)
def entrenar_modelo(n_arboles=100, profundidad=5):
    # 1. Leer los datos
    datos = pd.read_csv("corazon.csv")

    # 2. Rellenar los datos faltantes con la mediana
    datos = datos.fillna(datos.median())

    # 3. Separar entradas (X) y respuesta (y)
    X = datos.drop(columns=["diagnostico"])
    y = datos["diagnostico"]

    # 4. Separar 80% para entrenar y 20% para probar
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5. Crear y entrenar el modelo
    modelo = RandomForestClassifier(
        n_estimators=n_arboles, max_depth=profundidad, random_state=42
    )
    modelo.fit(X_train, y_train)

    # 6. Probar el modelo con los datos que nunca vio
    prediccion = modelo.predict(X_test)

    # 7. Calcular las métricas
    metricas = {
        "exactitud": float(accuracy_score(y_test, prediccion)),
        "precision": float(precision_score(y_test, prediccion)),
        "recall": float(recall_score(y_test, prediccion)),
        "f1": float(f1_score(y_test, prediccion)),
        "matriz": confusion_matrix(y_test, prediccion).tolist(),
        # Qué tanto pesa cada dato en la decisión del modelo
        "importancias": dict(zip(X.columns, modelo.feature_importances_.tolist())),
    }

    # 8. Guardar el modelo entrenado en un archivo
    joblib.dump(modelo, "modelo.joblib")

    return metricas


# Esto solo se ejecuta si corres este archivo directamente
if __name__ == "__main__":
    resultado = entrenar_modelo()
    print("Exactitud:", resultado["exactitud"])
    print("Precisión:", resultado["precision"])
    print("Recall:", resultado["recall"])
    print("F1:", resultado["f1"])
    print("Matriz de confusión:", resultado["matriz"])