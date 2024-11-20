# agent/train_agent.py

import numpy as np
import sys
import os
import random
import json
import pickle  # Para guardar objetos complejos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.blackjack_game import BlackjackGameEnv
from agent.dqn_agent import DQNAgent
from environment.state_representation import get_state_representation_for_hand

def train_agent(episodes=50000, batch_size=64):
    env = BlackjackGameEnv()
    state_size = 4  # ['player_value', 'dealer_upcard', 'true_count', 'hand_type']
    action_size = 4  # ['hit', 'stand', 'double', 'split']

    print("Initializing DQNAgent...")
    agent = DQNAgent(state_size, action_size)
    print("DQNAgent initialized.")

    rewards_per_episode = []
    epsilon_values = []
    losses_per_episode = []
    step_counts = []
    cumulative_rewards = []
    cumulative_reward = 0
    action_counts = {'hit': 0, 'stand': 0, 'double': 0, 'split': 0}
    average_rewards = []
    wins_per_episode = []

    for e in range(1, episodes + 1):
        env.reset()
        done = False
        total_reward = 0
        step_count = 0

        current_hand_index = 0  # Empezar con la primera mano

        while not done:
            if current_hand_index >= len(env.player.hands):
                # Todas las manos han sido procesadas
                break

            hand = env.player.hands[current_hand_index]

            if hand.is_finished or hand.is_bust():
                current_hand_index += 1
                continue  # Pasar a la siguiente mano

            # Obtener el estado para esta mano
            state_dict = get_state_representation_for_hand(hand, env.dealer, env.true_count)
            # Aplanar el estado para la red neuronal
            state_values = [
                state_dict['player_value'],
                state_dict['dealer_upcard'],
                state_dict['true_count'],
                state_dict['hand_type']
            ]
            state = np.reshape(state_values, [1, state_size])

            # Obtener acciones válidas
            valid_actions = env.get_valid_actions(hand)
            # El agente toma una acción
            action = agent.act(state, valid_actions)
            # Mapear índice de acción a nombre de acción
            action_names = ["hit", "stand", "double", "split"]
            action_name = action_names[action]
            # Registrar la acción tomada
            action_counts[action_name] += 1

            # Verificar si la acción es válida
            if action_name not in valid_actions:
                # Seleccionar una acción válida al azar
                action_name = random.choice(valid_actions)
                action = action_names.index(action_name)

            # Tomar un paso en el entorno para esta mano
            actions = {current_hand_index: action_name}
            next_state_dict, reward, done = env.step(actions)
            total_reward += reward
            step_count += 1

            # Obtener el siguiente estado para esta mano
            # Actualizar referencia de la mano en caso de que haya ocurrido un split
            hand = env.player.hands[current_hand_index]
            next_state_dict = get_state_representation_for_hand(hand, env.dealer, env.true_count)
            next_state_values = [
                next_state_dict['player_value'],
                next_state_dict['dealer_upcard'],
                next_state_dict['true_count'],
                next_state_dict['hand_type']
            ]
            next_state = np.reshape(next_state_values, [1, state_size])

            # Almacenar la experiencia
            agent.remember(state, action, reward, next_state, done)

            # Si la mano ha terminado o se ha pasado, pasar a la siguiente
            if hand.is_finished or hand.is_bust():
                current_hand_index += 1

            # No incrementar el índice si se hizo un split para procesar la misma mano de nuevo
            if action_name == 'split':
                # No incrementamos current_hand_index para volver a procesar la mano dividida
                pass

            if done:
                break

        # Reproducir experiencia y entrenar el modelo
        if len(agent.memory) > batch_size:
            loss = agent.replay(batch_size)
            if loss is not None and not np.isnan(loss):
                losses_per_episode.append(float(loss))
            else:
                losses_per_episode.append(0.0)
        else:
            losses_per_episode.append(0.0)  # Sin entrenamiento en este episodio

        # Registrar métricas
        rewards_per_episode.append(total_reward)
        epsilon_values.append(agent.epsilon)
        step_counts.append(step_count)
        cumulative_reward += total_reward
        cumulative_rewards.append(cumulative_reward)

        # Calcular recompensa promedio y tasa de victorias
        win = 1 if total_reward > 0 else 0
        wins_per_episode.append(win)
        average_reward = np.mean(rewards_per_episode[-100:])  # Promedio móvil de los últimos 100 episodios
        average_rewards.append(average_reward)

        # Guardar el modelo y las métricas cada 1000 episodios (checkpoints)
        if e % 1000 == 0:
            # Guardar el modelo
            agent.save(f"dqn_blackjack_model_{e}.keras")
            # Guardar el historial de decisiones
            agent.history_tracker.save_history(f"decision_history_{e}.json")
            # Guardar las métricas acumuladas hasta el momento
            np.save(f"rewards_per_episode_{e}.npy", np.array(rewards_per_episode))
            np.save(f"epsilon_values_{e}.npy", np.array(epsilon_values))
            np.save(f"losses_per_episode_{e}.npy", np.array(losses_per_episode))
            np.save(f"step_counts_{e}.npy", np.array(step_counts))
            np.save(f"cumulative_rewards_{e}.npy", np.array(cumulative_rewards))
            np.save(f"average_rewards_{e}.npy", np.array(average_rewards))
            np.save(f"wins_per_episode_{e}.npy", np.array(wins_per_episode))
            with open(f"action_counts_{e}.json", "w") as f:
                json.dump(action_counts, f)
            print(f"Checkpoint guardado en el episodio {e}")

        # Opcional: imprimir información del progreso
        if e % 100 == 0:
            print(f"Episodio {e}/{episodes} - Recompensa Promedio: {average_reward:.2f}, Epsilon: {agent.epsilon:.4f}")

    # Guardar historial de decisiones y métricas al final del entrenamiento
    agent.save("dqn_blackjack_model_final.keras")
    agent.history_tracker.save_history("decision_history_final.json")
    np.save("rewards_per_episode_final.npy", np.array(rewards_per_episode))
    np.save("epsilon_values_final.npy", np.array(epsilon_values))
    np.save("losses_per_episode_final.npy", np.array(losses_per_episode))
    np.save("step_counts_final.npy", np.array(step_counts))
    np.save("cumulative_rewards_final.npy", np.array(cumulative_rewards))
    np.save("average_rewards_final.npy", np.array(average_rewards))
    np.save("wins_per_episode_final.npy", np.array(wins_per_episode))
    with open("action_counts_final.json", "w") as f:
        json.dump(action_counts, f)
    print("Entrenamiento completado y datos finales guardados.")

if __name__ == "__main__":
    train_agent(episodes=1000, batch_size=32)
