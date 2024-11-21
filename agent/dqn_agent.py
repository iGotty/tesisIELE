# agent/dqn_agent.py

import numpy as np
import random
from collections import deque
import tensorflow as tf
from agent.history_tracker import HistoryTracker

class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.0001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)  # Increased memory size
        self.gamma = 0.99    # Discount factor
        self.epsilon = 1.0   # Initial exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999  # Slower decay for longer exploration
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history_tracker = HistoryTracker()
        self.loss_per_episode = []

    def _build_model(self):
        # Deeper and wider neural network architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Linear output for Q-values
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions):
        # Map actions to indices
        action_indices = {'hit': 0, 'stand': 1, 'double': 2, 'split': 3}
        valid_action_indices = [action_indices[action] for action in valid_actions]

        if np.random.rand() <= self.epsilon:
            # Randomly select a valid action
            action = random.choice(valid_action_indices)
        else:
            # Predict Q-values for all actions
            act_values = self.model.predict(state, verbose=0)[0]
            # Apply mask to invalidate invalid actions
            masked_q_values = np.full(self.action_size, -np.inf)
            for idx in valid_action_indices:
                masked_q_values[idx] = act_values[idx]
            action = np.argmax(masked_q_values)

        # Map index to action name
        action_names = ["hit", "stand", "double", "split"]
        action_name = action_names[action]

        # Record the decision
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
        # Train the model using random samples from memory
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([experience[0][0] for experience in minibatch])
        next_states = np.array([experience[3][0] for experience in minibatch])
        actions = [experience[1] for experience in minibatch]
        rewards = [experience[2] for experience in minibatch]
        dones = [experience[4] for experience in minibatch]

        # Predict Q-values for current states and next states
        target_q_values = self.model.predict(states, verbose=0)
        target_q_values_next = self.model.predict(next_states, verbose=0)

        # Update the Q-values for the actions taken
        for i in range(batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_q_values_next[i])

        # Train the model on the batch
        history = self.model.fit(states, target_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_per_episode.append(loss)

        # Reduce exploration rate
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
        # Save the trained model
        self.model.save(name)

    def load(self, name):
        # Load a saved model
        self.model = tf.keras.models.load_model(name)
