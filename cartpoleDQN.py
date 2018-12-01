import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow import layers

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:

    def __init__(self, observation_space, action_space, epsilon = 1.0):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.env = gym.make('CartPole-v0')
        self.epsilon = epsilon
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = tf.keras.models.Sequential()
        self.model.add(layers.Dense(24, input_dim=observation_space, activation="relu"))
        self.model.add(layers.Dense(24, activation="relu"))
        self.model.add(layers.Dense(self.action_space, activation="linear"))
        self.model.compile(loss='mse',optimizer=tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    #
    # def get_exploration_rate(self, t):
    #     return max(EXPLORATION_MIN, min(1.0, 1.0 - np.math.log10(t / 25.0)))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)

        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole(goal_reward=195):
    env = gym.make('CartPole-v0')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    average = 0
    scores = deque(maxlen=100)
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                average += step
                scores.append(step)
                mean_scores = np.mean(scores)
                if mean_scores >= goal_reward and run >= 100:
                    print('Solved in {} episodes'.format(run - 100))
                    return run - 100
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                if run % 100 == 0:
                    print("100 episodes over - Average: " + str(average / 100))
                    average = 0
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()