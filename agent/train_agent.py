# agent/train_agent.py

import numpy as np
import sys
import os
import random
import json
import tensorflow as tf
import time
import logging
from datetime import datetime
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.blackjack_game import BlackjackGameEnv
from agent.dqn_agent import DQNAgent
from environment.state_representation import get_state_representation_for_hand

def train_agent(episodes=50000, batch_size=1024):
    # Configure logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()

    # Check if GPU is available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            logger.info("GPU detected and configured for training.")
        except:
            logger.warning("Could not set memory growth for GPU. Using CPU for training.")
    else:
        logger.info("No GPU detected. Using CPU for training.")

    env = BlackjackGameEnv()
    state_size = 4  # ['player_value', 'dealer_upcard', 'true_count', 'hand_type']
    action_size = 4  # ['hit', 'stand', 'double', 'split']

    logger.info("Initializing DQNAgent...")
    agent = DQNAgent(state_size, action_size)
    logger.info("DQNAgent initialized.")

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path(f"training_results_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory created: {results_dir}")

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

        current_hand_index = 0  # Start with the first hand

        while not done:
            if current_hand_index >= len(env.player.hands):
                # All hands have been processed
                break

            hand = env.player.hands[current_hand_index]

            if hand.is_finished or hand.is_bust():
                current_hand_index += 1
                continue  # Move to the next hand

            # Get the state representation for this hand
            state_dict = get_state_representation_for_hand(hand, env.dealer, env.true_count)
            # Flatten the state for the neural network
            state_values = [
                state_dict['player_value'],
                state_dict['dealer_upcard'],
                state_dict['true_count'],
                state_dict['hand_type']
            ]
            state = np.reshape(state_values, [1, state_size])

            # Get valid actions
            valid_actions = env.get_valid_actions(hand)
            # Agent selects an action
            action = agent.act(state, valid_actions)
            # Map action index to action name
            action_names = ["hit", "stand", "double", "split"]
            action_name = action_names[action]
            # Record the action taken
            action_counts[action_name] += 1

            # Verify if the action is valid
            if action_name not in valid_actions:
                # Select a valid action randomly
                action_name = random.choice(valid_actions)
                action = action_names.index(action_name)

            # Take a step in the environment for this hand
            actions = {current_hand_index: action_name}
            next_state_dict, reward, done = env.step(actions)
            total_reward += reward
            step_count += 1

            # Get the next state for this hand
            # Update hand reference in case a split occurred
            hand = env.player.hands[current_hand_index]
            next_state_dict = get_state_representation_for_hand(hand, env.dealer, env.true_count)
            next_state_values = [
                next_state_dict['player_value'],
                next_state_dict['dealer_upcard'],
                next_state_dict['true_count'],
                next_state_dict['hand_type']
            ]
            next_state = np.reshape(next_state_values, [1, state_size])

            # Store the experience
            agent.remember(state, action, reward, next_state, done)

            # Move to the next hand if current hand is finished or bust
            if hand.is_finished or hand.is_bust():
                current_hand_index += 1

            # Do not increment hand index if a split occurred to process the same hand again
            if action_name == 'split':
                pass

            if done:
                break

        # Replay and train the model
        if len(agent.memory) > batch_size:
            loss = agent.replay(batch_size)
            if loss is not None and not np.isnan(loss):
                losses_per_episode.append(float(loss))
            else:
                losses_per_episode.append(0.0)
        else:
            losses_per_episode.append(0.0)  # No training in this episode

        # Record metrics
        rewards_per_episode.append(total_reward)
        epsilon_values.append(agent.epsilon)
        step_counts.append(step_count)
        cumulative_reward += total_reward
        cumulative_rewards.append(cumulative_reward)

        # Calculate average reward and win rate
        win = 1 if total_reward > 0 else 0
        wins_per_episode.append(win)
        average_reward = np.mean(rewards_per_episode[-100:])  # Moving average over last 100 episodes
        average_rewards.append(average_reward)

        # Save model and metrics at checkpoints
        if e % 1000 == 0:
            # Save the model
            agent.save(results_dir / f"dqn_blackjack_model_{e}.keras")
            # Save the decision history
            agent.history_tracker.save_history(results_dir / f"decision_history_{e}.json")
            # Save accumulated metrics
            np.save(results_dir / f"rewards_per_episode_{e}.npy", np.array(rewards_per_episode))
            np.save(results_dir / f"epsilon_values_{e}.npy", np.array(epsilon_values))
            np.save(results_dir / f"losses_per_episode_{e}.npy", np.array(losses_per_episode))
            np.save(results_dir / f"step_counts_{e}.npy", np.array(step_counts))
            np.save(results_dir / f"cumulative_rewards_{e}.npy", np.array(cumulative_rewards))
            np.save(results_dir / f"average_rewards_{e}.npy", np.array(average_rewards))
            np.save(results_dir / f"wins_per_episode_{e}.npy", np.array(wins_per_episode))
            with open(results_dir / f"action_counts_{e}.json", "w") as f:
                json.dump(action_counts, f)
            logger.info(f"Checkpoint saved at episode {e}")

        # Log progress every 100 episodes
        if e % 100 == 0:
            elapsed_time = time.time() - start_time
            logger.info(f"Episode {e}/{episodes} - Avg Reward: {average_reward:.2f}, Avg Loss: {losses_per_episode[-1]:.4f}, Epsilon: {agent.epsilon:.4f}, Elapsed Time: {elapsed_time/60:.2f} minutes")

    # Save final decision history and metrics
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
    logger.info("Training completed and final data saved.")

if __name__ == "__main__":
    train_agent(episodes=50000, batch_size=1024)
