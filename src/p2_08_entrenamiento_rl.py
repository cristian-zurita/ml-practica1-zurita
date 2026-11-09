"""Práctica 2, sesión P2-8: entrenamiento de PPO en CartPole-v1.

Entrega:
    models/ppo_cartpole.zip
    reports/figures/ppo_recompensa_acumulada.png
    reports/rl/p2_08_cartpole.monitor.csv

Ejecutar desde la raíz del repositorio:
    python src/p2_08_entrenamiento_rl.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # Guarda la gráfica sin abrir ventanas.
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_cartpole"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "ppo_recompensa_acumulada.png"
MONITOR_BASENAME = PROJECT_ROOT / "reports" / "rl" / "p2_08_cartpole"
MONITOR_CSV = Path(str(MONITOR_BASENAME) + ".monitor.csv")
TIMESTEPS = 12_288  # Supera los 10 000 pasos solicitados en P2-8.
SEED = 42


def leer_registro(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extrae pasos acumulados y recompensas de los episodios Monitor."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el registro: {path}")
    with path.open(newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(line for line in archivo if not line.startswith("#"))
        filas = list(lector)
    if not filas:
        raise ValueError("No se registraron episodios completos; no se puede graficar.")
    recompensas = np.array([float(fila["r"]) for fila in filas], dtype=float)
    duraciones = np.array([int(fila["l"]) for fila in filas], dtype=int)
    return np.cumsum(duraciones), recompensas


def graficar_recompensas(pasos: np.ndarray, recompensas: np.ndarray) -> None:
    """Grafica recompensa por episodio y media móvil (últimos 20 episodios)."""
    ventana = min(20, len(recompensas))
    medias = np.array(
        [recompensas[max(0, indice - ventana + 1):indice + 1].mean()
         for indice in range(len(recompensas))]
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(pasos, recompensas, alpha=0.35, linewidth=1, label="Recompensa de cada episodio")
    ax.plot(pasos, medias, linewidth=2, label=f"Media móvil (hasta {ventana} episodios)")
    ax.set_title("P2-8 — PPO en CartPole-v1: evolución de la recompensa")
    ax.set_xlabel("Pasos de entrenamiento acumulados")
    ax.set_ylabel("Recompensa total por episodio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    """Entrena PPO, guarda su modelo y exporta la curva de aprendizaje."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MONITOR_BASENAME.parent.mkdir(parents=True, exist_ok=True)

    entorno = Monitor(gym.make("CartPole-v1"), filename=str(MONITOR_BASENAME))
    try:
        modelo = PPO(
            "MlpPolicy",
            entorno,
            seed=SEED,
            verbose=1,
            device="cpu",  # CartPole es ligero; evita depender de CUDA.
            n_steps=1024,
            batch_size=64,
        )
        print("\nP2-8 — ENTRENAMIENTO CON PPO")
        print(f"Entorno: CartPole-v1 | Pasos solicitados: {TIMESTEPS} | Semilla: {SEED}")
        modelo.learn(total_timesteps=TIMESTEPS)
        modelo.save(str(MODEL_PATH))  # Crea models/ppo_cartpole.zip.
    finally:
        entorno.close()

    pasos, recompensas = leer_registro(MONITOR_CSV)
    graficar_recompensas(pasos, recompensas)
    print("\nRESUMEN DE ENTRENAMIENTO")
    print(f"Episodios completados: {len(recompensas)}")
    print(f"Pasos observados en episodios completos: {pasos[-1]}")
    print(f"Recompensa media total: {recompensas.mean():.2f}")
    print(f"Recompensa media últimos {min(20, len(recompensas))} episodios: "
          f"{recompensas[-20:].mean():.2f}")
    print(f"Modelo: {MODEL_PATH}.zip")
    print(f"Gráfica: {FIGURE_PATH}")
    print(f"Registro de episodios: {MONITOR_CSV}")
    print("NOTA: esta curva describe el ENTRENAMIENTO; la evaluación será P2-9.")


if __name__ == "__main__":
    main()
