"""Práctica 2 — Sesión 7: entorno RL personalizado con Gymnasium.

Objetivo: un agente se desplaza por una línea (0..8) hasta la meta (8).
Acción 0 = izquierda; acción 1 = derecha.

Ejecutar desde la raíz del proyecto:
    python src/p2_07_entorno_rl.py
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env


class LineaMetaEnv(gym.Env[int, int]):
    """Entorno discreto: llegar a la última casilla antes del límite de pasos."""

    metadata = {"render_modes": ["ansi"], "render_fps": 2}

    def __init__(self, longitud: int = 9, max_pasos: int = 40):
        super().__init__()
        if longitud < 2 or max_pasos < 1:
            raise ValueError("longitud debe ser >= 2 y max_pasos >= 1")

        self.longitud = longitud
        self.meta = longitud - 1
        self.max_pasos = max_pasos
        self.action_space = spaces.Discrete(2)  # 0: izquierda; 1: derecha
        self.observation_space = spaces.Discrete(longitud)  # posición actual
        self.render_mode = "ansi"
        self.posicion = 0
        self.pasos = 0
        self.finalizado = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reinicia el episodio; devuelve (observación, información)."""
        super().reset(seed=seed)
        self.posicion = 0
        self.pasos = 0
        self.finalizado = False
        return self.posicion, {"meta": self.meta, "pasos": self.pasos}

    def step(self, action: int):
        """Aplica una acción y devuelve la API de cinco elementos de Gymnasium."""
        if self.finalizado:
            raise RuntimeError("El episodio terminó: llama a reset() antes de step().")
        if not self.action_space.contains(action):
            raise ValueError(f"Acción inválida: {action!r}; solo 0 o 1")

        desplazamiento = -1 if action == 0 else 1
        self.posicion = max(0, min(self.meta, self.posicion + desplazamiento))
        self.pasos += 1

        terminated = self.posicion == self.meta  # objetivo alcanzado
        truncated = self.pasos >= self.max_pasos and not terminated  # tiempo agotado
        recompensa = 10.0 if terminated else -0.1
        self.finalizado = terminated or truncated
        info = {"meta": self.meta, "pasos": self.pasos, "exito": terminated}
        return self.posicion, recompensa, terminated, truncated, info

    def render(self):
        """Representación textual: A=agente, G=meta, .=casilla libre."""
        celdas = []
        for i in range(self.longitud):
            if i == self.posicion == self.meta:
                celdas.append("AG")
            elif i == self.posicion:
                celdas.append("A")
            elif i == self.meta:
                celdas.append("G")
            else:
                celdas.append(".")
        return "[" + " | ".join(celdas) + "]"


def probar_agente_aleatorio(episodios: int = 5):
    """Ejecuta cinco episodios sin aprendizaje y resume sus recompensas."""
    env = LineaMetaEnv()
    check_env(env)  # comprueba espacios y firmas de reset/step/render

    resultados = []
    print("\nP2-7 — ENTORNO DE LÍNEA PERSONALIZADO")
    print("Espacios:", env.observation_space, "(estado),", env.action_space, "(acción)")
    print("Recompensas: +10 por alcanzar la meta; -0.1 por cada otro paso")
    print("Límite de pasos:", env.max_pasos)

    for numero in range(1, episodios + 1):
        estado, info = env.reset(seed=42 + numero)
        env.action_space.seed(100 + numero)
        total = 0.0
        terminado = False
        pasos = 0
        print(f"\nEpisodio {numero} — Inicio: {env.render()}")

        while not terminado:
            accion = env.action_space.sample()
            estado, recompensa, terminated, truncated, info = env.step(accion)
            total += recompensa
            pasos += 1
            terminado = terminated or truncated
            if pasos <= 5 or terminado:
                print(
                    f"  paso={pasos:02d} acción={accion} posición={estado} "
                    f"recompensa={recompensa:>4.1f} done={terminado}"
                )

        etiqueta = "META" if info["exito"] else "LÍMITE DE PASOS"
        print(f"  Final: {env.render()} — {etiqueta} — total={total:.1f}")
        resultados.append((numero, pasos, total, info["exito"]))

    print("\nRESUMEN")
    print("Episodio | Pasos | Recompensa total | Meta alcanzada")
    for numero, pasos, total, exito in resultados:
        print(f"{numero:>8} | {pasos:>5} | {total:>16.1f} | {'Sí' if exito else 'No'}")

    # Verificación adicional: una política sencilla debe poder alcanzar la meta.
    env.reset(seed=999)
    for _ in range(env.meta):
        estado, _, terminated, truncated, _ = env.step(1)
    assert estado == env.meta and terminated and not truncated
    print("\nVerificación de la meta: OK (ir siempre a la derecha llega a la meta).")
    env.close()


if __name__ == "__main__":
    probar_agente_aleatorio()
