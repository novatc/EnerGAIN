import pandas as pd
import numpy as np

from dqn_agent import DQNAgent
from enviroment import EnergyMarkets


def train_dqn(agent, env, episodes, batch_size):
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, -1])
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, -1])

            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if (e + 1) % 10 == 0:  # Update target network every 10 episodes
            agent.update_target_model()
            print("Episode: {}/{}, Epsilon: {:.2f}".format(e + 1, episodes, agent.epsilon))


def evaluate_dqn(agent, env, episodes):
    total_rewards = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, -1])
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, -1])

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


if __name__ == '__main__':
    # Load and preprocess the data
    data = pd.read_csv("data/clean/dataset_01102018_01012023.csv")
    # ... Preprocess the data as shown in previous examples

    # Create the environment
    env = EnergyMarkets(data)

    # Create the DQN agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print("State size: {}, Action size: {}".format(state_size, action_size))
    agent = DQNAgent(state_size, action_size)

    # Train the agent
    train_episodes = 1000
    batch_size = 64
    train_dqn(agent, env, train_episodes, batch_size)

    # Evaluate the agent
    test_episodes = 100
    avg_reward = evaluate_dqn(agent, env, test_episodes)
    print("Average reward over {} test episodes: {:.2f}".format(test_episodes, avg_reward))
