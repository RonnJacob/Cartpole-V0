import gym
import numpy as np
import math
import sys
from collections import deque
import matplotlib.pyplot as plt


class QLearningCartPole:
    def __init__(self, buckets=(1, 1, 6, 12,), min_learning_rate=0.1, min_exploration_rate=0.1, discount=1.0):
        # Load the cart-pole environment.
        self.env = gym.make('CartPole-v0')

        # discrete values for each feature space dimension(position, velocity, angle, angular velocity)
        self.buckets = buckets
        self.min_learning_rate = min_learning_rate
        self.min_exploration_rate = min_exploration_rate
        self.discount = discount

        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))  # also include the action space

    def convert_to_discrete(self, state):

        # set manual upper bounds for velocity and angular velocity.
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]

        # set manual lower bounds for velocity and angular velocity.
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]

        # find the width for every dimension.
        width = [upper_bounds[i] - lower_bounds[i] for i in range(len(state))]

        # generate ratios for every dimension.
        ratios = [(state[i] + abs(lower_bounds[i])) / width[i] for i in range(len(state))]

        # discretize each dimension into one of the buckets
        discrete_state = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(state))]

        # making the range of buckets to 0 to bucket_length
        discrete_state = [min(self.buckets[i] - 1, max(0, discrete_state[i])) for i in range(len(state))]
        return tuple(discrete_state)

    def get_action(self, state, exploration_rate):
        if np.random.random() > exploration_rate:
            # action with highest q value
            return np.argmax(self.q_table[state])
        else:
            # choose a random action
            return self.env.action_space.sample()

    def update(self, current_state, action, reward, new_state, alpha):
        self.q_table[current_state][action] += alpha * (
                reward + self.discount * np.max(self.q_table[new_state]) - self.q_table[current_state][action])

    def get_exploration_rate(self, t):
        return max(self.min_exploration_rate, min(1.0, 1.0 - math.log10(t / 25.0)))

    def get_learning_rate(self, t):
        return max(self.min_learning_rate, min(1.0, 1.0 - math.log10(t / 25.0)))

    def run(self, episodes=1000, goal_reward=195):
        scores = deque(maxlen=100)

        for episode in range(1, episodes):
            current_state = self.convert_to_discrete(self.env.reset())
            alpha = self.get_learning_rate(episode)
            epsilon = self.get_exploration_rate(episode)
            r = 0
            done = False

            while not done:
                # self.env.render()
                action = self.get_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.convert_to_discrete(obs)
                self.update(current_state, action, reward, new_state, alpha)
                current_state = new_state
                r += reward

            scores.append(r)
            mean_score = np.mean(scores)

            if mean_score >= goal_reward and episode >= 100:
                print('Solved in {} episodes'.format(episode - 100))
                return episode - 100
            else:
                print("Episode {}, reward {}".format(episode, r))
        return episode


if __name__ == "__main__":
    solver = QLearningCartPole()
    x = []
    y = []
    for i in xrange(1, 11):
        x.append(i)
        y.append(solver.run())
        plt.plot(x, y, 'ro')
    plt.ylabel('No of Episodes to solve')
    plt.show()
    print('Min {}'.format(min(y)))
    print('Max {}'.format(max(y)))
    print('Mean {}'.format(np.mean(y)))
    print('SD {}'.format(np.std(y)))
    sys.exit()
