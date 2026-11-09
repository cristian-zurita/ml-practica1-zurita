"""Práctica 2, sesión 9: evaluación independiente de PPO vs. agente aleatorio.

Ejecutar desde la raíz del proyecto:
    python src/p2_09_evaluacion_rl.py

Requiere gymnasium, stable-baselines3 y el modelo de P2-8.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "ppo_cartpole.zip"
OUTPUT_DIR = ROOT / "reports" / "rl"
NUM_EPISODES = 10
SEEDS = tuple(range(1000, 1000 + NUM_EPISODES))


def evaluate_episode(model: PPO | None, seed: int) -> dict[str, int | float | bool]:
    """Ejecuta un episodio sin entrenamiento y devuelve su resultado.

    model=None usa acciones aleatorias. El entorno recibe una semilla
    fija para hacer comparables los estados iniciales de ambos agentes.
    """
    env = gym.make("CartPole-v1")
    try:
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed + 10_000)
        total_reward = 0.0
        steps = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if model is None:
                action = int(env.action_space.sample())
            else:
                prediction, _ = model.predict(obs, deterministic=True)
                action = int(np.asarray(prediction).item())

            obs, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            total_reward += float(reward)

        return {
            "seed": seed,
            "pasos": steps,
            "recompensa": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    finally:
        env.close()


def format_stats(values: list[float]) -> tuple[float, float, float]:
    """Retorna media, mínima y máxima de las recompensas observadas."""
    return mean(values), min(values), max(values)


def save_reports(ppo_results: list[dict], random_results: list[dict]) -> None:
    """Guarda resultados reproducibles y un informe Markdown conciso."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "p2_09_evaluacion.csv"
    md_path = OUTPUT_DIR / "p2_09_reporte.md"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["agente", "episodio", "seed", "pasos", "recompensa", "terminated", "truncated"])
        for agent_name, rows in (("PPO", ppo_results), ("Aleatorio", random_results)):
            for i, row in enumerate(rows, start=1):
                writer.writerow([
                    agent_name, i, row["seed"], row["pasos"],
                    row["recompensa"], row["terminated"], row["truncated"],
                ])

    ppo_rewards = [float(row["recompensa"]) for row in ppo_results]
    rand_rewards = [float(row["recompensa"]) for row in random_results]
    ppo_avg, ppo_min, ppo_max = format_stats(ppo_rewards)
    rand_avg, rand_min, rand_max = format_stats(rand_rewards)
    delta = ppo_avg - rand_avg

    lines = [
        "# P2-9 — Evaluación de aprendizaje por refuerzo",
        "",
        "Se evaluó el modelo PPO de P2-8 frente a un agente de acciones aleatorias",
        "en CartPole-v1 durante 10 episodios por agente. No se entrenó de nuevo.",
        "Se utilizaron las mismas semillas de inicio para ambos agentes (1000 a 1009)",
        "y predicción determinista para PPO. Las acciones aleatorias se sembraron",
        "independientemente, para permitir la reproducción del experimento.",
        "",
        "| Agente | Recompensa media | Mínima | Máxima |",
        "|---|---:|---:|---:|",
        f"| PPO | {ppo_avg:.2f} | {ppo_min:.0f} | {ppo_max:.0f} |",
        f"| Aleatorio | {rand_avg:.2f} | {rand_min:.0f} | {rand_max:.0f} |",
        "",
        f"Diferencia de recompensa media (PPO − aleatorio): {delta:+.2f}.",
        "",
        "## Interpretación",
        (
            "El agente PPO obtuvo una recompensa media mayor en estos diez episodios."
            if delta > 0 else
            "El agente PPO no superó la recompensa media del agente aleatorio en estos diez episodios."
        ),
        "Los resultados son una evaluación inicial: diez episodios no bastan para",
        "garantizar el comportamiento ante todas las condiciones iniciales.",
        "",
        "## Archivos",
        "- Modelo evaluado: `models/ppo_cartpole.zip`",
        "- Resultados por episodio: `reports/rl/p2_09_evaluacion.csv`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCSV: {csv_path}")
    print(f"Informe: {md_path}")


def main() -> None:
    """Carga el PPO y ejecuta la evaluación de ambos agentes."""
    if not MODEL_PATH.is_file():
        raise SystemExit(
            f"No se encontró el modelo: {MODEL_PATH}\n"
            "Primero ejecuta P2-8 para generar models/ppo_cartpole.zip."
        )

    print("P2-9 — EVALUACIÓN INDEPENDIENTE DE PPO VS. AGENTE ALEATORIO")
    print(f"Modelo: {MODEL_PATH}")
    print(f"Entorno: CartPole-v1 | Episodios por agente: {NUM_EPISODES}")
    print("Semillas compartidas: 1000 a 1009 | PPO: deterministic=True")
    print("No se realizan actualizaciones de entrenamiento.\n")

    model = PPO.load(str(MODEL_PATH), device="cpu")
    ppo_results = [evaluate_episode(model, seed) for seed in SEEDS]
    random_results = [evaluate_episode(None, seed) for seed in SEEDS]

    print(f"{'Episodio':>8} {'PPO':>10} {'Aleatorio':>10} {'Diferencia':>11}")
    for index, (ppo, random) in enumerate(zip(ppo_results, random_results), start=1):
        a, b = float(ppo["recompensa"]), float(random["recompensa"])
        print(f"{index:>8} {a:>10.1f} {b:>10.1f} {a-b:>+11.1f}")

    ppo_stats = format_stats([float(x["recompensa"]) for x in ppo_results])
    random_stats = format_stats([float(x["recompensa"]) for x in random_results])
    print("\nRESUMEN (media / mínima / máxima)")
    print(f"PPO:      {ppo_stats[0]:.2f} / {ppo_stats[1]:.1f} / {ppo_stats[2]:.1f}")
    print(f"Aleatorio: {random_stats[0]:.2f} / {random_stats[1]:.1f} / {random_stats[2]:.1f}")
    print(f"Diferencia de medias: {ppo_stats[0]-random_stats[0]:+.2f}")
    save_reports(ppo_results, random_results)


if __name__ == "__main__":
    main()
