"""
Entrenamiento y evaluación de modelos supervisados
para el dataset Wall-Following Robot Navigation.

Este script utiliza los datos previamente procesados para entrenar,
evaluar y comparar dos modelos de clasificación supervisada:

1. Random Forest.
2. Support Vector Machine (SVM).

El objetivo es determinar la acción de navegación del robot a partir de
24 lecturas de sensores ultrasónicos.

Flujo general:
1. Carga `data/processed/dataset.parquet`.
2. Recupera los conjuntos de entrenamiento y prueba utilizando
   la columna `split`.
3. Separa las 24 variables de entrada de la variable objetivo `Class`.
4. Entrena un modelo Random Forest.
5. Evalúa Random Forest mediante:
   - Accuracy.
   - F1-score ponderado.
   - F1-score macro.
   - Matriz de confusión.
   - Tiempo de entrenamiento.
6. Guarda el modelo Random Forest entrenado.
7. Entrena un modelo Support Vector Machine (SVM).
8. Evalúa SVM utilizando las mismas métricas y el mismo conjunto de prueba.
9. Guarda el modelo SVM entrenado.
10. Genera una tabla comparativa entre ambos modelos.

Entradas:
- `data/processed/dataset.parquet`

Salidas:
- `models/random_forest.pkl`
- `models/svm.pkl`
- Métricas y comparación mostradas en terminal.

Funciones principales:
- train_random_forest(X_train, y_train):
  entrena un modelo Random Forest.

- train_svm(X_train, y_train):
  entrena un modelo Support Vector Machine.

- evaluate_model(model, X_test, y_test):
  evalúa cualquier modelo de clasificación utilizando Accuracy,
  F1-score y matriz de confusión.

Consideraciones:
- Ambos modelos utilizan exactamente los mismos conjuntos de entrenamiento
  y prueba para permitir una comparación justa.
- El conjunto de prueba no participa en el entrenamiento.
- Los datos utilizados ya fueron escalados durante la etapa de
  preprocesamiento.
- El escalado es particularmente relevante para SVM.
"""

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score
)
from sklearn.svm import SVC


# Nombres de las 24 variables utilizadas como entrada.
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

    # Se utilizan los hiperparámetros por defecto.
    # random_state=42 permite reproducir el mismo entrenamiento.
    model = RandomForestClassifier(
        random_state=42
    )

    # Aquí ocurre el aprendizaje del modelo.
    model.fit(
        X_train,
        y_train
    )

    return model


def train_svm(X_train, y_train):
    """
    Entrena un modelo Support Vector Machine utilizando los datos
    de entrenamiento.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables de entrada utilizadas para entrenar el modelo.

    y_train : pandas.Series
        Clases reales correspondientes a X_train.

    Returns
    -------
    SVC
        Modelo SVM entrenado.
    """

    # SVC utiliza por defecto el kernel RBF.
    # Por ahora se mantienen los hiperparámetros predeterminados
    # para realizar una comparación inicial sin optimización.
    model = SVC()

    # Aquí ocurre el aprendizaje del modelo.
    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo utilizando datos que no participaron
    durante el entrenamiento.

    Parameters
    ----------
    model
        Modelo de clasificación previamente entrenado.

    X_test : pandas.DataFrame
        Variables de entrada reservadas para evaluación.

    y_test : pandas.Series
        Clases reales correspondientes a X_test.

    Returns
    -------
    dict
        Diccionario con Accuracy, F1 ponderado,
        F1 macro y matriz de confusión.
    """

    # El modelo genera una predicción para cada observación de X_test.
    y_pred = model.predict(X_test)

    # Accuracy:
    # proporción total de predicciones correctas.
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # F1 ponderado:
    # considera el número de observaciones existentes en cada clase.
    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted"
    )

    # F1 macro:
    # asigna el mismo peso a cada clase.
    # Es especialmente útil porque el dataset está desbalanceado.
    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    # Mantener el mismo orden de clases utilizado por el modelo.
    labels = model.classes_

    # Construcción de la matriz de confusión.
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    # Convertir la matriz a DataFrame para facilitar su lectura.
    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Real: {label}"
            for label in labels
        ],
        columns=[
            f"Pred: {label}"
            for label in labels
        ]
    )

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "confusion_matrix": cm_df
    }


def main():
    """
    Ejecuta el flujo completo de entrenamiento, evaluación,
    almacenamiento y comparación de Random Forest y SVM.
    """

    # ---------------------------------------------------------
    # Rutas del proyecto
    # ---------------------------------------------------------

    # Localizar automáticamente la raíz del proyecto.
    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "dataset.parquet"
    )

    random_forest_path = (
        project_root
        / "models"
        / "random_forest.pkl"
    )

    svm_path = (
        project_root
        / "models"
        / "svm.pkl"
    )

    # ---------------------------------------------------------
    # 1. Cargar dataset procesado
    # ---------------------------------------------------------

    df = pd.read_parquet(
        dataset_path
    )

    # ---------------------------------------------------------
    # 2. Recuperar TRAIN y TEST
    # ---------------------------------------------------------

    train_df = df[
        df["split"] == "train"
    ].copy()

    test_df = df[
        df["split"] == "test"
    ].copy()

    # ---------------------------------------------------------
    # 3. Separar X e y
    # ---------------------------------------------------------

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

    # =========================================================
    # RANDOM FOREST
    # =========================================================

    print("\n===== RANDOM FOREST =====")

    # Registrar el instante antes de iniciar el entrenamiento.
    rf_start = perf_counter()

    # Entrenar Random Forest.
    random_forest = train_random_forest(
        X_train,
        y_train
    )

    # Calcular únicamente el tiempo empleado en entrenamiento.
    rf_training_time = (
        perf_counter() - rf_start
    )

    # Evaluar Random Forest utilizando TEST.
    rf_metrics = evaluate_model(
        random_forest,
        X_test,
        y_test
    )

    print(
        f"Accuracy: "
        f"{rf_metrics['accuracy']:.4f}"
    )

    print(
        f"F1-score ponderado: "
        f"{rf_metrics['f1_weighted']:.4f}"
    )

    print(
        f"F1-score macro: "
        f"{rf_metrics['f1_macro']:.4f}"
    )

    print(
        f"Tiempo de entrenamiento: "
        f"{rf_training_time:.4f} segundos"
    )

    print("\nMatriz de confusión:")

    print(
        rf_metrics[
            "confusion_matrix"
        ].to_string()
    )

    # Guardar Random Forest entrenado.
    joblib.dump(
        random_forest,
        random_forest_path
    )

    print("\nModelo Random Forest guardado en:")
    print(random_forest_path)

    # =========================================================
    # SUPPORT VECTOR MACHINE
    # =========================================================

    print("\n===== SUPPORT VECTOR MACHINE =====")

    # Registrar instante inicial.
    svm_start = perf_counter()

    # Entrenar SVM.
    svm_model = train_svm(
        X_train,
        y_train
    )

    # Calcular tiempo de entrenamiento.
    svm_training_time = (
        perf_counter() - svm_start
    )

    # Evaluar SVM utilizando exactamente el mismo TEST.
    svm_metrics = evaluate_model(
        svm_model,
        X_test,
        y_test
    )

    print(
        f"Accuracy: "
        f"{svm_metrics['accuracy']:.4f}"
    )

    print(
        f"F1-score ponderado: "
        f"{svm_metrics['f1_weighted']:.4f}"
    )

    print(
        f"F1-score macro: "
        f"{svm_metrics['f1_macro']:.4f}"
    )

    print(
        f"Tiempo de entrenamiento: "
        f"{svm_training_time:.4f} segundos"
    )

    print("\nMatriz de confusión:")

    print(
        svm_metrics[
            "confusion_matrix"
        ].to_string()
    )

    # Guardar SVM entrenado.
    joblib.dump(
        svm_model,
        svm_path
    )

    print("\nModelo SVM guardado en:")
    print(svm_path)

    # =========================================================
    # COMPARACIÓN GENERAL
    # =========================================================

    print("\n===== COMPARACIÓN GENERAL =====")

    comparison = pd.DataFrame(
        {
            "Modelo": [
                "Random Forest",
                "SVM"
            ],

            "Accuracy": [
                rf_metrics["accuracy"],
                svm_metrics["accuracy"]
            ],

            "F1 ponderado": [
                rf_metrics["f1_weighted"],
                svm_metrics["f1_weighted"]
            ],

            "F1 macro": [
                rf_metrics["f1_macro"],
                svm_metrics["f1_macro"]
            ],

            "Tiempo entrenamiento (s)": [
                rf_training_time,
                svm_training_time
            ]
        }
    )

    print(
        comparison.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()