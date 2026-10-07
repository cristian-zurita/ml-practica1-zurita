"""
Entrenamiento y evaluación del primer modelo supervisado
para el dataset Wall-Following Robot Navigation.

Este script utiliza los datos previamente procesados en la Clase 4
para entrenar un modelo Random Forest capaz de clasificar la acción
de navegación del robot a partir de 24 lecturas de sensores ultrasónicos.

Flujo general:
1. Carga `data/processed/dataset.parquet`.
2. Separa nuevamente los registros pertenecientes a train y test
   utilizando la columna `split`.
3. Separa las variables de entrada X de la variable objetivo y.
4. Entrena un modelo Random Forest utilizando exclusivamente
   el conjunto de entrenamiento.
5. Utiliza el modelo entrenado para predecir las clases de X_test.
6. Evalúa las predicciones mediante:
   - Accuracy.
   - F1-score ponderado.
   - F1-score macro.
   - Matriz de confusión.
7. Guarda el modelo entrenado en:
   `models/random_forest.pkl`.

Entradas:
- `data/processed/dataset.parquet`

Salidas:
- `models/random_forest.pkl`
- Métricas de evaluación mostradas en terminal.

Funciones principales:
- train_random_forest(X_train, y_train):
  entrena y devuelve un modelo Random Forest.

- evaluate_model(model, X_test, y_test):
  genera predicciones y calcula las métricas de evaluación.

Nota:
El modelo se entrena únicamente con el conjunto `train`.
El conjunto `test` se utiliza exclusivamente para evaluar el rendimiento
sobre observaciones que no participaron en el entrenamiento.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score
)


# Nombres de las 24 variables de entrada.
SENSOR_COLUMNS = [f"US{i}" for i in range(1, 25)]


def train_random_forest(X_train, y_train):
    """
    Entrena un modelo Random Forest utilizando los datos de entrenamiento.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables de entrada utilizadas para entrenar el modelo.

    y_train : pandas.Series
        Clases reales correspondientes a X_train.

    Returns
    -------
    RandomForestClassifier
        Modelo Random Forest entrenado.
    """

    # Se mantienen los hiperparámetros por defecto.
    # random_state permite reproducir el mismo entrenamiento.
    model = RandomForestClassifier(
        random_state=42
    )

    # Aquí ocurre realmente el aprendizaje del modelo.
    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo utilizando datos que no participaron en el entrenamiento.

    Parameters
    ----------
    model : RandomForestClassifier
        Modelo previamente entrenado.

    X_test : pandas.DataFrame
        Variables de entrada reservadas para evaluación.

    y_test : pandas.Series
        Clases reales correspondientes a X_test.

    Returns
    -------
    dict
        Diccionario con Accuracy, F1 ponderado, F1 macro
        y matriz de confusión.
    """

    # El modelo genera una clase para cada observación de X_test.
    y_pred = model.predict(X_test)

    # Porcentaje global de predicciones correctas.
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # F1 ponderado considerando la cantidad de muestras de cada clase.
    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # F1 macro da el mismo peso a cada clase.
    # Es especialmente útil porque nuestro dataset está desbalanceado.
    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    # Orden fijo de clases utilizado por el modelo.
    labels = model.classes_

    # Matriz de confusión.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    # Convertir la matriz a DataFrame para visualizarla mejor.
    cm_df = pd.DataFrame(
        cm,
        index=[f"Real: {label}" for label in labels],
        columns=[f"Pred: {label}" for label in labels]
    )

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm_df
    }


def main():
    """
    Ejecuta el flujo completo de entrenamiento y evaluación.
    """

    # Localizar automáticamente la raíz del proyecto.
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "dataset.parquet"
    )

    model_path = (
        project_root
        / "models"
        / "random_forest.pkl"
    )

    # 1. Cargar dataset procesado.
    df = pd.read_parquet(dataset_path)

    # 2. Recuperar los conjuntos definidos durante el preprocesamiento.
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    # 3. Separar variables de entrada y objetivo.
    X_train = train_df[SENSOR_COLUMNS]
    y_train = train_df["Class"]

    X_test = test_df[SENSOR_COLUMNS]
    y_test = test_df["Class"]

    print("Datos utilizados para entrenamiento:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nDatos reservados para evaluación:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # 4. Entrenar Random Forest.
    model = train_random_forest(
        X_train,
        y_train
    )

    # 5. Evaluar sobre TEST.
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n===== RESULTADOS RANDOM FOREST =====")

    print(
        f"Accuracy: {metrics['accuracy']:.4f}"
    )

    print(
        f"F1-score ponderado: {metrics['f1_weighted']:.4f}"
    )

    print(
        f"F1-score macro: {metrics['f1_macro']:.4f}"
    )

    print("\nMatriz de confusión:")
    print(metrics["confusion_matrix"].to_string())

    # 6. Guardar el modelo entrenado.
    joblib.dump(
        model,
        model_path
    )

    print("\nModelo guardado en:")
    print(model_path)


if __name__ == "__main__":
    main()