import numpy as np
import matplotlib.pyplot as plt
import json

# Cargar métricas
rewards = np.load("rewards_per_episode_1000.npy")
epsilon_values = np.load("epsilon_values_1000.npy")
losses = np.load("losses_per_episode_1000.npy", allow_pickle=True)
step_counts = np.load("step_counts_1000.npy")
cumulative_rewards = np.load("cumulative_rewards_1000.npy")

# Gráfico de recompensas por episodio
plt.figure()
plt.plot(rewards)
plt.title("Recompensas por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Recompensa")
plt.savefig("recompensas_por_episodio.png")

# Gráfico de epsilon por episodio
plt.figure()
plt.plot(epsilon_values)
plt.title("Valor de Epsilon por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Epsilon")
plt.savefig("epsilon_por_episodio.png")

# Gráfico de pérdida (loss) por episodio
plt.figure()
plt.plot(losses)
plt.title("Pérdida por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Pérdida")
plt.savefig("perdida_por_episodio.png")

# Gráfico de pasos por episodio
plt.figure()
plt.plot(step_counts)
plt.title("Número de Pasos por Episodio")
plt.xlabel("Episodio")
plt.ylabel("Pasos")
plt.savefig("pasos_por_episodio.png")

# Gráfico de ganancia acumulada
plt.figure()
plt.plot(cumulative_rewards)
plt.title("Ganancia Acumulada")
plt.xlabel("Episodio")
plt.ylabel("Ganancia Acumulada")
plt.savefig("ganancia_acumulada.png")

# Mostrar todos los gráficos
plt.show()

# Cargar y mostrar conteo de acciones
with open("action_counts.json", "r") as f:
    action_counts = json.load(f)

print("Frecuencia de acciones tomadas:")
total_actions = sum(action_counts.values())
for action, count in action_counts.items():
    percentage = (count / total_actions) * 100 if total_actions > 0 else 0
    print(f"  Acción: {action}, Frecuencia: {count}, Porcentaje: {percentage:.2f}%")
