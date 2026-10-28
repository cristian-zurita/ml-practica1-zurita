"""Práctica 2 - Sesión 2: matriz de costos del clasificador binario.

Ejecutar desde el repositorio de P1:
    python scripts/p2_02_costos.py

Reproduce la configuración de p2_01_metricas.ipynb y evalúa ambos modelos
sobre el mismo conjunto de prueba. Los costos son supuestos de la práctica,
no valores medidos en una implementación industrial real.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

COSTO_FP = 50  # USD por falsa alarma
COSTO_FN = 500  # USD por evento positivo no detectado
CLASE_POSITIVA = "Slight-Left-Turn"
RANDOM_STATE = 42


def encontrar_dataset():
    """Busca el parquet en el repositorio, sin depender del directorio actual."""
    origenes = [Path.cwd(), *Path.cwd().parents, Path(__file__).resolve().parent]
    origenes += list(Path(__file__).resolve().parents)
    for origen in origenes:
        ruta = origen / "data" / "processed" / "dataset.parquet"
        if ruta.is_file():
            return ruta
    raise FileNotFoundError(
        "No se encontró data/processed/dataset.parquet. "
        "Guarda este script dentro del repositorio de la Práctica 1."
    )


def evaluar_costo(nombre, etiquetas, predicciones):
    """Calcula FP, FN y costo total a partir de predicciones binarias."""
    tn, fp, fn, tp = confusion_matrix(etiquetas, predicciones, labels=[0, 1]).ravel()
    costo_fp = int(fp) * COSTO_FP
    costo_fn = int(fn) * COSTO_FN
    return {
        "Modelo": nombre,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Costo FP (USD)": costo_fp,
        "Costo FN (USD)": costo_fn,
        "Costo total (USD)": costo_fp + costo_fn,
    }


def main():
    """Entrena modelos comparables y presenta la evaluación económica."""
    ruta = encontrar_dataset()
    datos = pd.read_parquet(ruta)
    columnas_sensores = [f"US{i}" for i in range(1, 25)]
    columnas_requeridas = columnas_sensores + ["Class"]
    faltantes = [nombre for nombre in columnas_requeridas if nombre not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    x = datos[columnas_sensores].apply(pd.to_numeric, errors="raise")
    y = datos["Class"].astype(str).str.strip().eq(CLASE_POSITIVA).astype(int)
    if x.isna().any().any():
        raise ValueError("Hay valores ausentes en las variables de entrada.")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("No hay ambas clases en el conjunto de datos.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    bosque = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    trivial = DummyClassifier(strategy="constant", constant=0)
    bosque.fit(x_train, y_train)
    trivial.fit(x_train, y_train)

    resultados = pd.DataFrame([
        evaluar_costo("Random Forest", y_test, bosque.predict(x_test)),
        evaluar_costo("Siempre 0", y_test, trivial.predict(x_test)),
    ]).set_index("Modelo")
    print("\nMATRIZ DE COSTOS — PRÁCTICA 2, SESIÓN 2")
    print(f"Dataset: {ruta}")
    print(f"FP = ${COSTO_FP}; FN = ${COSTO_FN}")
    print("Costo total = FP × 50 + FN × 500\n")
    print(resultados.to_string())

    costo_bosque = int(resultados.loc["Random Forest", "Costo total (USD)"])
    costo_trivial = int(resultados.loc["Siempre 0", "Costo total (USD)"])
    ahorro = costo_trivial - costo_bosque
    porcentaje = ahorro / costo_trivial * 100 if costo_trivial else 0.0
    print(f"\nAhorro con Random Forest frente al modelo trivial: ${ahorro:,.0f}")
    print(f"Reducción relativa del costo: {porcentaje:.2f}%")
    print(
        "Conclusión: Random Forest "
        + ("reduce" if ahorro > 0 else "no reduce")
        + " el costo de clasificación respecto al modelo trivial "
        "en el conjunto de prueba, con los costos hipotéticos de la guía."
    )


if __name__ == "__main__":
    main()
