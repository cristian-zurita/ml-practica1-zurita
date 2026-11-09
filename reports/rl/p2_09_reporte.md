# P2-9 — Evaluación de aprendizaje por refuerzo

Se evaluó el modelo PPO de P2-8 frente a un agente de acciones aleatorias
en CartPole-v1 durante 10 episodios por agente. No se entrenó de nuevo.
Se utilizaron las mismas semillas de inicio para ambos agentes (1000 a 1009)
y predicción determinista para PPO. Las acciones aleatorias se sembraron
independientemente, para permitir la reproducción del experimento.

| Agente | Recompensa media | Mínima | Máxima |
|---|---:|---:|---:|
| PPO | 339.80 | 219 | 500 |
| Aleatorio | 25.00 | 12 | 47 |

Diferencia de recompensa media (PPO − aleatorio): +314.80.

## Interpretación
El agente PPO obtuvo una recompensa media mayor en estos diez episodios.
Los resultados son una evaluación inicial: diez episodios no bastan para
garantizar el comportamiento ante todas las condiciones iniciales.

## Archivos
- Modelo evaluado: `models/ppo_cartpole.zip`
- Resultados por episodio: `reports/rl/p2_09_evaluacion.csv`
