# agent/dqn_agent.py

import numpy as np
import random
from collections import deque
import tensorflow as tf
from agent.history_tracker import HistoryTracker

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # Aumentamos el tamaño de la memoria
        self.gamma = 0.99    # Factor de descuento
        self.epsilon = 1.0   # Tasa de exploración inicial
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995  # Disminución más lenta de epsilon
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history_tracker = HistoryTracker()
        self.loss_per_episode = []

    def _build_model(self):
        # Red Neuronal más profunda para capturar patrones complejos
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Salida lineal para Q-valores
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        # Almacenar experiencia en memoria
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        # Mapear acciones a índices
        action_indices = {'hit': 0, 'stand': 1, 'double': 2, 'split': 3}
        valid_action_indices = [action_indices[action] for action in valid_actions]

        if np.random.rand() <= self.epsilon:
            # Seleccionar una acción válida al azar
            action = random.choice(valid_action_indices)
        else:
            # Predecir Q-valores para todas las acciones
            act_values = self.model.predict(state, verbose=0)[0]
            # Aplicar máscara a Q-valores de acciones inválidas
            masked_q_values = np.full(self.action_size, -np.inf)
            for idx in valid_action_indices:
                masked_q_values[idx] = act_values[idx]
            action = np.argmax(masked_q_values)

        # Mapear índice a nombre de acción
        action_names = ["hit", "stand", "double", "split"]
        action_name = action_names[action]

        # Registrar la decisión
        state_dict = {
            "player_value": state[0][0],
            "dealer_upcard": state[0][1],
            "true_count": state[0][2],
            "hand_type": int(state[0][3])
        }
        category = self.get_state_category(state_dict)
        self.history_tracker.record_action(category, state_dict, action_name)

        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([experience[0][0] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        actions = [experience[1] for experience in minibatch]
        rewards = [experience[2] for experience in minibatch]
        dones = [experience[4] for experience in minibatch]

        # Predecir Q-valores para estados actuales y siguientes
        target_f = self.model.predict(states, verbose=0)
        target_q_values_next = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            if dones[i]:
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * np.amax(target_q_values_next[i])
            target_f[i][actions[i]] = target

        # Entrenar el modelo en batch
        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_per_episode.append(loss)

        # Actualizar epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss


    def get_state_category(self, state):
        hand_type = state['hand_type']
        if hand_type == 0:
            return 'soft_totals'
        elif hand_type == 1:
            return 'pair_splitting'
        else:
            return 'hard_totals'

    def save(self, name):
        # Guardar el modelo entrenado
        self.model.save(name)

    def load(self, name):
        # Cargar un modelo guardado
        self.model = tf.keras.models.load_model(name)
