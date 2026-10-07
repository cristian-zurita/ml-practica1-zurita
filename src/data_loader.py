"""
Módulo de carga y preprocesamiento del dataset
Wall-Following Robot Navigation.

Este script prepara los datos que posteriormente serán utilizados para
entrenar y evaluar los modelos de Machine Learning de la práctica.

Flujo general:
1. Carga el archivo original `sensor_readings_24.data`.
2. Asigna nombres a las 24 variables correspondientes a los sensores
   ultrasónicos (`US1` a `US24`) y a la variable objetivo `Class`.
3. Separa las variables de entrada X de la variable objetivo y.
4. Divide el dataset en conjuntos de entrenamiento y prueba utilizando
   una proporción 80/20.
5. Utiliza una división estratificada para conservar aproximadamente
   la misma proporción de clases en train y test.
6. Aplica RobustScaler a las variables numéricas.
7. El escalador se ajusta exclusivamente con los datos de entrenamiento
   para evitar fuga de información (data leakage).
8. Los datos de prueba se transforman utilizando únicamente los parámetros
   aprendidos a partir del conjunto de entrenamiento.
9. Reconstruye los datos procesados junto con la variable objetivo.
10. Añade una columna `split` para identificar si cada observación pertenece
    al conjunto `train` o `test`.
11. Guarda el resultado final en:
    `data/processed/dataset.parquet`.

Entradas:
- `data/raw/sensor_readings_24.data`

Salidas:
- `data/processed/dataset.parquet`

Funciones principales:
- load_raw_data(path): carga el dataset original.
- preprocess(df): separa las variables de entrada X y la variable objetivo y.
- split_data(X, y): realiza la división estratificada 80/20.

Nota sobre fuga de datos:
El escalado se realiza después de dividir el dataset. RobustScaler aprende
la mediana y el rango intercuartílico únicamente de X_train. Posteriormente,
estos mismos parámetros se utilizan para transformar X_test sin volver a
ajustar el escalador.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# Nombres de las 24 variables de entrada.
SENSOR_COLUMNS = [f"US{i}" for i in range(1, 25)]

# 24 sensores + variable objetivo.
COLUMNS = SENSOR_COLUMNS + ["Class"]


def load_raw_data(path):
    """
    Carga el dataset original y asigna nombres a las columnas.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS
    )

    return df


def preprocess(df):
    """
    Separa las variables de entrada X y la variable objetivo y.

    El escalado todavía no se realiza aquí porque primero debemos
    separar entrenamiento y prueba para evitar fuga de datos.
    """
    X = df[SENSOR_COLUMNS].copy()
    y = df["Class"].copy()

    return X, y


def split_data(X, y):
    """
    Divide los datos en entrenamiento y prueba usando una proporción 80/20.

    stratify=y mantiene aproximadamente la misma proporción
    de clases en ambos subconjuntos.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def main():
    """
    Ejecuta el flujo completo de preprocesamiento.
    """

    # Obtener automáticamente la raíz del proyecto.
    project_root = Path(__file__).resolve().parents[1]

    raw_path = (
        project_root
        / "data"
        / "raw"
        / "sensor_readings_24.data"
    )

    processed_path = (
        project_root
        / "data"
        / "processed"
        / "dataset.parquet"
    )

    # 1. Cargar dataset.
    df = load_raw_data(raw_path)

    # 2. Separar variables de entrada y objetivo.
    X, y = preprocess(df)

    # 3. Dividir ANTES de escalar.
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 4. Crear el escalador.
    scaler = RobustScaler()

    # El scaler aprende SOLO de entrenamiento.
    X_train_scaled = scaler.fit_transform(X_train)

    # Test usa los parámetros aprendidos en train.
    X_test_scaled = scaler.transform(X_test)

    # Convertir los arrays nuevamente a DataFrame.
    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=SENSOR_COLUMNS,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=SENSOR_COLUMNS,
        index=X_test.index
    )

    # Recuperar variable objetivo.
    train_processed = X_train_scaled.copy()
    train_processed["Class"] = y_train

    test_processed = X_test_scaled.copy()
    test_processed["Class"] = y_test

    # Identificar el subconjunto al que pertenece cada observación.
    train_processed["split"] = "train"
    test_processed["split"] = "test"

    # Unir para guardar un solo archivo procesado.
    processed_df = pd.concat(
        [train_processed, test_processed],
        ignore_index=True
    )

    # Guardar en formato Parquet.
    processed_df.to_parquet(
        processed_path,
        index=False
    )

    # Verificaciones.
    print("Dataset original:", df.shape)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print("\nDistribución de clases en TRAIN:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nDistribución de clases en TEST:")
    print(y_test.value_counts(normalize=True).round(4))

    print("\nScaler ajustado únicamente con X_train.")
    print("Archivo generado:")
    print(processed_path)


if __name__ == "__main__":
    main()