"""Práctica 2 · Sesión 3: optimización del umbral según el costo.

Colocar en src/, junto a p2_02_costos.py, y ejecutar desde la raíz:
    python src/p2_03_threshold.py

El umbral se elige exclusivamente en validación; el conjunto de prueba se
reserva para comparar de forma honesta el umbral 0.50 con el seleccionado.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from p2_02_costos import (
    CLASE_POSITIVA,
    COSTO_FN,
    COSTO_FP,
    RANDOM_STATE,
    encontrar_dataset,
    evaluar_costo,
)


def entrenar_bosque(x, y):
    """Entrena el mismo Random Forest de las sesiones 1 y 2."""
    modelo = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    )
    modelo.fit(x, y)
    return modelo


def predecir_con_umbral(probabilidades, umbral):
    """Convierte la probabilidad de la clase 1 en decisiones binarias."""
    return (probabilidades >= umbral).astype(int)


def main():
    """Elige un umbral en validación y lo contrasta en test sin reajustarlo."""
    ruta_dataset = encontrar_dataset()
    datos = pd.read_parquet(ruta_dataset)
    columnas = [f"US{i}" for i in range(1, 25)]
    faltantes = [c for c in columnas + ["Class"] if c not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas del dataset: {faltantes}")

    x = datos[columnas].apply(pd.to_numeric, errors="raise")
    y = datos["Class"].astype(str).str.strip().eq(CLASE_POSITIVA).astype(int)
    if x.isna().any().any():
        raise ValueError("El dataset contiene valores faltantes en los sensores.")
    if y.nunique() != 2:
        raise ValueError("El dataset no contiene las dos clases binarias.")

    # Misma partición de prueba (20 %) utilizada en las sesiones 1 y 2.
    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # Separamos validación SOLO del 80 % de entrenamiento.
    x_ajuste, x_validacion, y_ajuste, y_validacion = train_test_split(
        x_entrenamiento,
        y_entrenamiento,
        test_size=0.25,
        stratify=y_entrenamiento,
        random_state=RANDOM_STATE,
    )
    modelo_validacion = entrenar_bosque(x_ajuste, y_ajuste)
    probabilidades_val = modelo_validacion.predict_proba(x_validacion)[:, 1]

    filas = []
    for umbral in np.round(np.arange(0.10, 0.901, 0.05), 2):
        predicciones = predecir_con_umbral(probabilidades_val, umbral)
        resultado = evaluar_costo("Validación", y_validacion, predicciones)
        filas.append({"Umbral": float(umbral), **{
            clave: resultado[clave] for clave in ("FP", "FN", "Costo total (USD)")
        }})
    tabla = pd.DataFrame(filas)
    # Ante empates en costo, preferimos el umbral más próximo a 0.50.
    candidatas = tabla.assign(distancia=(tabla["Umbral"] - 0.50).abs())
    mejor = candidatas.sort_values(
        ["Costo total (USD)", "distancia", "Umbral"]
    ).iloc[0]
    umbral_optimo = float(mejor["Umbral"])

    print("\nTHRESHOLD TUNING — PRÁCTICA 2, SESIÓN 3")
    print(f"Dataset: {ruta_dataset}")
    print(f"Costo FP = ${COSTO_FP}, costo FN = ${COSTO_FN}")
    print(f"Validación: {len(y_validacion)} filas; prueba: {len(y_prueba)} filas")
    print("\nBÚSQUEDA DEL UMBRAL (solo validación):")
    print(tabla.to_string(index=False, formatters={"Umbral": "{:.2f}".format}))
    print(f"\nUmbral elegido en validación: {umbral_optimo:.2f}")
    print(f"Costo mínimo de validación: ${int(mejor['Costo total (USD)']):,}")

    # Entrenamos el modelo definitivo con TODO el 80 % de entrenamiento,
    # igual que en P2-1 y P2-2. No usamos el test para seleccionar el umbral.
    modelo_final = entrenar_bosque(x_entrenamiento, y_entrenamiento)
    probabilidades_prueba = modelo_final.predict_proba(x_prueba)[:, 1]
    fila_base = evaluar_costo(
        "Umbral 0.50", y_prueba, predecir_con_umbral(probabilidades_prueba, 0.50)
    )
    fila_optimo = evaluar_costo(
        f"Umbral {umbral_optimo:.2f}",
        y_prueba,
        predecir_con_umbral(probabilidades_prueba, umbral_optimo),
    )
    comparacion = pd.DataFrame([fila_base, fila_optimo]).set_index("Modelo")
    print("\nCOMPARACIÓN EN PRUEBA (sin optimizar con test):")
    print(comparacion[["TN", "FP", "FN", "TP", "Costo total (USD)"]].to_string())
    diferencia = fila_base["Costo total (USD)"] - fila_optimo["Costo total (USD)"]
    if diferencia > 0:
        print(f"\nAhorro frente al umbral 0.50: ${diferencia:,.0f}")
    elif diferencia == 0:
        print("\nEl umbral seleccionado no cambia el costo en prueba.")
    else:
        print(f"\nEl costo en prueba AUMENTÓ ${-diferencia:,.0f}.")
        print("Esto puede ocurrir: el umbral se seleccionó sin mirar el test.")

    carpeta_salida = ruta_dataset.parents[2] / "reports" / "figures"
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_grafico = carpeta_salida / "threshold_tuning.png"
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tabla["Umbral"], tabla["Costo total (USD)"], marker="o")
    ax.axvline(0.50, color="gray", linestyle="--", label="Umbral estándar: 0.50")
    ax.axvline(umbral_optimo, color="green", linestyle=":", label=f"Elegido: {umbral_optimo:.2f}")
    ax.scatter([umbral_optimo], [mejor["Costo total (USD)"]], color="green", zorder=3)
    ax.set(
        title="Costo de clasificación según umbral (validación)",
        xlabel="Umbral de decisión",
        ylabel="Costo total hipotético (USD)",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta_grafico, dpi=160)
    plt.close(fig)
    print(f"\nGráfico guardado en: {ruta_grafico}")
    print("Nota: las cantidades son costos hipotéticos establecidos en la guía.")


if __name__ == "__main__":
    main()
