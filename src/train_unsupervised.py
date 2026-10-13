"""
Entrenamiento y evaluación de modelos no supervisados
para el dataset Wall-Following Robot Navigation.

Este script utiliza las 24 variables de sensores ultrasónicos previamente
procesadas para identificar agrupaciones naturales en los datos sin utilizar
la variable objetivo `Class` durante el entrenamiento.

Se emplean dos algoritmos:

1. K-Means.
2. DBSCAN.

Flujo general:
1. Carga `data/processed/dataset.parquet`.
2. Selecciona únicamente las 24 variables numéricas de sensores.
3. Determina un número adecuado de clusters para K-Means mediante
   el método del codo.
4. Entrena K-Means con el valor de k seleccionado.
5. Calcula el Silhouette Score de K-Means.
6. Entrena DBSCAN.
7. Calcula el número de clusters encontrados por DBSCAN.
8. Identifica los puntos considerados ruido por DBSCAN.
9. Calcula el Silhouette Score de DBSCAN cuando sea posible.
10. Guarda los modelos entrenados.

Entradas:
- `data/processed/dataset.parquet`

Salidas:
- `models/kmeans.pkl`
- `models/dbscan.pkl`
- Valores de inercia utilizados para el método del codo.
- Silhouette Score de ambos algoritmos.
- Información sobre clusters y ruido.

Funciones principales:
- find_optimal_k(X):
  entrena diferentes modelos K-Means y obtiene la inercia
  necesaria para analizar el método del codo.

- train_kmeans(X, k):
  entrena K-Means utilizando el número de clusters seleccionado.

- train_dbscan(X, eps, min_samples):
  entrena DBSCAN utilizando los parámetros especificados.

Consideraciones:
- La variable `Class` no participa durante el entrenamiento.
- La columna `split` tampoco se utiliza como variable de entrada.
- Los datos ya fueron escalados durante la etapa de preprocesamiento.
"""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt

import joblib
import pandas as pd

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score


# Las mismas 24 variables utilizadas durante las etapas anteriores.
SENSOR_COLUMNS = [f"US{i}" for i in range(1, 25)]


def find_optimal_k(X, k_min=2, k_max=10):
    """
    Calcula la inercia de K-Means para diferentes valores de k.

    Los resultados pueden utilizarse posteriormente para construir
    la gráfica del método del codo.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables utilizadas para clustering.

    k_min : int
        Número mínimo de clusters a evaluar.

    k_max : int
        Número máximo de clusters a evaluar.

    Returns
    -------
    pandas.DataFrame
        Tabla con los valores de k y sus respectivas inercias.
    """

    results = []

    for k in range(k_min, k_max + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X)

        results.append(
            {
                "k": k,
                "inertia": model.inertia_
            }
        )

    return pd.DataFrame(results)


def train_kmeans(X, k):
    """
    Entrena un modelo K-Means.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables utilizadas para clustering.

    k : int
        Número de clusters.

    Returns
    -------
    KMeans
        Modelo K-Means entrenado.
    """

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X)

    return model


def train_dbscan(X, eps=0.5, min_samples=5):
    """
    Entrena un modelo DBSCAN.

    Parameters
    ----------
    X : pandas.DataFrame
        Variables utilizadas para clustering.

    eps : float
        Distancia máxima entre puntos vecinos.

    min_samples : int
        Número mínimo de observaciones necesarias para formar
        una región densa.

    Returns
    -------
    DBSCAN
        Modelo DBSCAN entrenado.
    """

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    model.fit(X)

    return model


def evaluate_dbscan_parameters(
    X,
    eps_values,
    min_samples_values
):
    """
    Evalúa distintas combinaciones de parámetros para DBSCAN.

    Para cada combinación se calcula:
    - Número de clusters encontrados.
    - Número y porcentaje de puntos considerados ruido.
    - Silhouette Score cuando existen al menos dos clusters.

    El Silhouette Score durante esta búsqueda se calcula utilizando
    únicamente los puntos que DBSCAN no clasificó como ruido.
    """

    results = []

    for min_samples in min_samples_values:

        for eps in eps_values:

            model = train_dbscan(
                X,
                eps=eps,
                min_samples=min_samples
            )

            labels = model.labels_

            # DBSCAN utiliza -1 para identificar ruido.
            noise_mask = labels == -1

            noise_count = int(
                noise_mask.sum()
            )

            noise_percentage = (
                noise_count / len(labels)
            ) * 100

            unique_labels = set(labels)

            n_clusters = len(unique_labels)

            if -1 in unique_labels:
                n_clusters -= 1

            # -------------------------------------------------
            # Silhouette Score
            # -------------------------------------------------

            non_noise_mask = ~noise_mask

            non_noise_labels = labels[
                non_noise_mask
            ]

            non_noise_cluster_count = len(
                set(non_noise_labels)
            )

            if (
                non_noise_cluster_count >= 2
                and non_noise_mask.sum() > non_noise_cluster_count
            ):

                silhouette = silhouette_score(
                    X.loc[non_noise_mask],
                    non_noise_labels,
                    sample_size=min(
                        2000,
                        int(non_noise_mask.sum())
                    ),
                    random_state=42
                )

            else:
                silhouette = None

            results.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "clusters": n_clusters,
                    "noise_points": noise_count,
                    "noise_percentage": noise_percentage,
                    "silhouette": silhouette
                }
            )

    return pd.DataFrame(results)

def main():
    """
    Ejecuta la primera etapa del análisis no supervisado.

    En esta etapa:
    1. Carga el dataset procesado.
    2. Selecciona únicamente las 24 variables de sensores.
    3. Evalúa distintos valores de k para K-Means.
    4. Muestra las inercias obtenidas.
    5. Genera y guarda la gráfica del método del codo.

    La selección final de k se realizará después de analizar
    los resultados obtenidos.
    """

    # ---------------------------------------------------------
    # Rutas del proyecto
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "dataset.parquet"
    )

    elbow_plot_path = (
        project_root
        / "reports"
        / "clase_07_elbow.png"
    )

    # ---------------------------------------------------------
    # 1. Cargar dataset procesado
    # ---------------------------------------------------------

    df = pd.read_parquet(dataset_path)

    # ---------------------------------------------------------
    # 2. Seleccionar únicamente variables de entrada
    # ---------------------------------------------------------

    # No utilizamos Class porque estamos realizando
    # aprendizaje no supervisado.
    #
    # Tampoco utilizamos split porque únicamente identifica
    # si una fila pertenecía a train o test.
    X = df[SENSOR_COLUMNS].copy()

    print("Dataset utilizado para clustering:")
    print("X:", X.shape)

    print("\nColumnas utilizadas:")
    print(X.columns.tolist())

    # ---------------------------------------------------------
    # 3. Método del codo
    # ---------------------------------------------------------

    print("\n===== MÉTODO DEL CODO =====")

    start = perf_counter()

    elbow_results = find_optimal_k(
        X,
        k_min=2,
        k_max=10
    )

    elapsed_time = perf_counter() - start

    print(
        elbow_results.to_string(
            index=False
        )
    )

    print(
        f"\nTiempo de evaluación: "
        f"{elapsed_time:.4f} segundos"
    )

    # ---------------------------------------------------------
    # 4. Crear gráfica del método del codo
    # ---------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        elbow_results["k"],
        elbow_results["inertia"],
        marker="o"
    )

    plt.title("Método del codo - K-Means")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.xticks(elbow_results["k"])
    plt.grid(True)
    plt.tight_layout()

    # Guardar la gráfica como evidencia del experimento.
    plt.savefig(
        elbow_plot_path,
        dpi=150
    )


    print("\nGráfica guardada en:")
    print(elbow_plot_path)

        # =========================================================
    # 5. Entrenamiento definitivo de K-Means
    # =========================================================

    # A partir del método del codo se selecciona k = 4.
    selected_k = 4

    print("\n===== K-MEANS =====")
    print(f"Número de clusters seleccionado: {selected_k}")

    kmeans_start = perf_counter()

    kmeans_model = train_kmeans(
        X,
        selected_k
    )

    kmeans_training_time = (
        perf_counter() - kmeans_start
    )

    # Obtener el cluster asignado a cada observación.
    kmeans_labels = kmeans_model.labels_

    # Evaluar la separación interna de los clusters.
    kmeans_silhouette = silhouette_score(
        X,
        kmeans_labels
    )

    print(
        f"Silhouette Score: "
        f"{kmeans_silhouette:.4f}"
    )

    print(
        f"Tiempo de entrenamiento: "
        f"{kmeans_training_time:.4f} segundos"
    )

    print("\nCantidad de observaciones por cluster:")

    cluster_counts = pd.Series(
        kmeans_labels
    ).value_counts().sort_index()

    print(cluster_counts)

    # ---------------------------------------------------------
    # 6. Guardar modelo K-Means
    # ---------------------------------------------------------

    kmeans_path = (
        project_root
        / "models"
        / "kmeans.pkl"
    )

    joblib.dump(
        kmeans_model,
        kmeans_path
    )

    print("\nModelo K-Means guardado en:")
    print(kmeans_path)

        # =========================================================
    # 7. Búsqueda inicial de parámetros para DBSCAN
    # =========================================================

    print("\n===== BÚSQUEDA DE PARÁMETROS DBSCAN =====")

    eps_values = [
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0
    ]

    min_samples_values = [
        5,
        10,
        15,
        20
    ]

    dbscan_search = evaluate_dbscan_parameters(
        X,
        eps_values,
        min_samples_values
    )

    print(
        dbscan_search.round(
            {
                "noise_percentage": 2,
                "silhouette": 4
            }
        ).to_string(
            index=False
        )
    )

        # =========================================================
    # 8. Entrenamiento definitivo de DBSCAN
    # =========================================================

    # Parámetros seleccionados a partir de la búsqueda experimental.
    selected_eps = 5.5
    selected_min_samples = 20

    print("\n===== DBSCAN DEFINITIVO =====")

    print(
        f"eps seleccionado: "
        f"{selected_eps}"
    )

    print(
        f"min_samples seleccionado: "
        f"{selected_min_samples}"
    )

    dbscan_start = perf_counter()

    dbscan_model = train_dbscan(
        X,
        eps=selected_eps,
        min_samples=selected_min_samples
    )

    dbscan_training_time = (
        perf_counter() - dbscan_start
    )

    dbscan_labels = dbscan_model.labels_

    # ---------------------------------------------------------
    # Número de clusters y ruido
    # ---------------------------------------------------------

    unique_labels = set(
        dbscan_labels
    )

    dbscan_cluster_count = len(
        unique_labels
    )

    if -1 in unique_labels:
        dbscan_cluster_count -= 1

    noise_mask = (
        dbscan_labels == -1
    )

    noise_count = int(
        noise_mask.sum()
    )

    noise_percentage = (
        noise_count
        / len(dbscan_labels)
    ) * 100

    print(
        f"Número de clusters encontrados: "
        f"{dbscan_cluster_count}"
    )

    print(
        f"Puntos considerados ruido: "
        f"{noise_count}"
    )

    print(
        f"Porcentaje de ruido: "
        f"{noise_percentage:.2f} %"
    )

    # ---------------------------------------------------------
    # Silhouette Score definitivo
    # ---------------------------------------------------------

    # DBSCAN utiliza -1 para representar ruido.
    # El silhouette se calcula únicamente con los puntos
    # asignados a clusters reales.
    non_noise_mask = ~noise_mask

    X_non_noise = X.loc[
        non_noise_mask
    ]

    labels_non_noise = dbscan_labels[
        non_noise_mask
    ]

    dbscan_silhouette = silhouette_score(
        X_non_noise,
        labels_non_noise
    )

    print(
        f"Silhouette Score: "
        f"{dbscan_silhouette:.4f}"
    )

    print(
        f"Tiempo de entrenamiento: "
        f"{dbscan_training_time:.4f} segundos"
    )

    # ---------------------------------------------------------
    # Distribución de clusters
    # ---------------------------------------------------------

    print("\nCantidad de observaciones por etiqueta:")

    dbscan_counts = pd.Series(
        dbscan_labels
    ).value_counts().sort_index()

    print(dbscan_counts)

    print(
        "\nNota: la etiqueta -1 representa "
        "observaciones consideradas ruido."
    )

    # ---------------------------------------------------------
    # 9. Guardar DBSCAN
    # ---------------------------------------------------------

    dbscan_path = (
        project_root
        / "models"
        / "dbscan.pkl"
    )

    joblib.dump(
        dbscan_model,
        dbscan_path
    )

    print("\nModelo DBSCAN guardado en:")
    print(dbscan_path)


if __name__ == "__main__":
    main()