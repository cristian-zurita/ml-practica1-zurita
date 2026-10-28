from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from src.data_loader import (
    load_raw_data,
    preprocess,
    split_data,
)


def main():
    # 1. Localizar la carpeta principal del proyecto.
    raiz = Path(__file__).resolve().parents[1]

    # 2. Cargar los datos originales (sin escalar).
    ruta = raiz / "data/raw/sensor_readings_24.data"
    df = load_raw_data(ruta)

    # 3. Separar características y etiquetas.
    X, y = preprocess(df)

    # 4. Dividir entrenamiento y prueba.
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 5. Construir nuestro pipeline.
    modelo = Pipeline([
        ("escalador", RobustScaler()),
        ("clasificador", SVC()),
    ])

    # 6. Entrenar el pipeline completo.
    modelo.fit(X_train, y_train)

    # 7. Evaluar utilizando datos de prueba.
    predicciones = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, predicciones)
    f1_macro = f1_score(
        y_test,
        predicciones,
        average="macro",
    )

    print("Accuracy:", round(accuracy, 4))
    print("F1 macro:", round(f1_macro, 4))


if __name__ == "__main__":
    main()
