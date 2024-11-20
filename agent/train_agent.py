# agent/train_agent.py

import numpy as np
import sys
import os
import random
import json
import pickle  # Para guardar objetos complejos
import tensorflow as tf
import time  # Para medir el tiempo transcurrido
import logging  # Para mensajes estructurados
from datetime import datetime  # Para timestamp
from pathlib import Path  # Para manejar rutas de archivos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.blackjack_game import BlackjackGameEnv
from agent.dqn_agent import DQNAgent
from environment.state_representation import get_state_representation_for_hand

def train_agent(episodes=50000, batch_size=64):
    # Configurar el logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Verificar si la GPU está disponible
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logger.info("GPU detectada y configurada para el entrenamiento.")
        except:
            logger.warning("No se pudo configurar la GPU. Usando la CPU para el entrenamiento.")
    else:
        logger.info("No se detectó GPU. Usando la CPU para el entrenamiento.")

    env = BlackjackGameEnv()
    state_size = 4  # ['player_value', 'dealer_upcard', 'true_count', 'hand_type']
    action_size = 4  # ['hit', 'stand', 'double', 'split']

    logger.info("Inicializando DQNAgent...")
    agent = DQNAgent(state_size, action_size)
    logger.info("DQNAgent inicializado.")

    # Crear directorio para guardar los resultados con timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path(f"training_results_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Directorio de resultados creado: {results_dir}")

    rewards_per_episode = []
    epsilon_values = []
    losses_per_episode = []
    step_counts = []
    cumulative_rewards = []
    cumulative_reward = 0
    action_counts = {'hit': 0, 'stand': 0, 'double': 0, 'split': 0}
    average_rewards = []
    wins_per_episode = []

    start_time = time.time()

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
            agent.save(results_dir / f"dqn_blackjack_model_{e}.keras")
            # Guardar el historial de decisiones
            agent.history_tracker.save_history(results_dir / f"decision_history_{e}.json")
            # Guardar las métricas acumuladas hasta el momento
            np.save(results_dir / f"rewards_per_episode_{e}.npy", np.array(rewards_per_episode))
            np.save(results_dir / f"epsilon_values_{e}.npy", np.array(epsilon_values))
            np.save(results_dir / f"losses_per_episode_{e}.npy", np.array(losses_per_episode))
            np.save(results_dir / f"step_counts_{e}.npy", np.array(step_counts))
            np.save(results_dir / f"cumulative_rewards_{e}.npy", np.array(cumulative_rewards))
            np.save(results_dir / f"average_rewards_{e}.npy", np.array(average_rewards))
            np.save(results_dir / f"wins_per_episode_{e}.npy", np.array(wins_per_episode))
            with open(results_dir / f"action_counts_{e}.json", "w") as f:
                json.dump(action_counts, f)
            logger.info(f"Checkpoint guardado en el episodio {e}")

        # Imprimir información del progreso cada 100 episodios
        if e % 100 == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Episodio {e}/{episodes} - Recompensa Promedio: {average_reward:.2f}, Pérdida Promedio: {losses_per_episode[-1]:.4f}, Epsilon: {agent.epsilon:.4f}, Tiempo Transcurrido: {elapsed_time/60:.2f} minutos")

    # Guardar historial de decisiones y métricas al final del entrenamiento
    agent.save(results_dir / "dqn_blackjack_model_final.keras")
    agent.history_tracker.save_history(results_dir / "decision_history_final.json")
    np.save(results_dir / "rewards_per_episode_final.npy", np.array(rewards_per_episode))
    np.save(results_dir / "epsilon_values_final.npy", np.array(epsilon_values))
    np.save(results_dir / "losses_per_episode_final.npy", np.array(losses_per_episode))
    np.save(results_dir / "step_counts_final.npy", np.array(step_counts))
    np.save(results_dir / "cumulative_rewards_final.npy", np.array(cumulative_rewards))
    np.save(results_dir / "average_rewards_final.npy", np.array(average_rewards))
    np.save(results_dir / "wins_per_episode_final.npy", np.array(wins_per_episode))
    with open(results_dir / "action_counts_final.json", "w") as f:
        json.dump(action_counts, f)
    logger.info("Entrenamiento completado y datos finales guardados.")

if __name__ == "__main__":
    train_agent(episodes=200, batch_size=64)
