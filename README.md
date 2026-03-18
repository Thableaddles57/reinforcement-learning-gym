
# Reinforcement Learning Gym

Implementations of various reinforcement learning algorithms (e.g., Q-learning, SARSA, DQN) using OpenAI Gym environments.

## Algorithms Implemented

- **Q-Learning**: A value-based off-policy reinforcement learning algorithm.
- **SARSA**: A value-based on-policy reinforcement learning algorithm.
- **DQN (Deep Q-Network)**: Combining Q-learning with deep neural networks.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Each algorithm is implemented in its own file within the `src/` directory. You can run them directly:

```bash
python src/q_learning_agent.py
```

## Example: Q-Learning on FrozenLake

```python
import gym
import numpy as np

# Initialize environment
env = gym.make('FrozenLake-v1', is_slippery=False)

# Q-Learning parameters
learning_rate = 0.9
discount_factor = 0.8
episodes = 1000

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(q_table[state, :] + np.random.randn(1, env.action_space.n) * (1./(episode + 1)))
        next_state, reward, done, _, _ = env.step(action)
        
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[next_state, :]) - q_table[state, action])
        state = next_state

print("Q-table trained!")
print(q_table)

# Test the agent
total_rewards = 0
for _ in range(100):
    state = env.reset()[0]
    done = False
    while not done:
        action = np.argmax(q_table[state, :])
        next_state, reward, done, _, _ = env.step(action)
        total_rewards += reward
        state = next_state

print(f"Average reward over 100 test episodes: {total_rewards / 100}")
env.close()
```

## Contributing

Feel free to add more algorithms or improve existing ones!
