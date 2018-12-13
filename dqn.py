import gym
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

fig = plt.figure()
plt.plot(10, 3000)
fig.suptitle('Episodes to finish learning.')


class DQNAgent:

    # A naive neural network with 3 hidden layers and Rectified Linear Units as non-linear function.
    def __init__(self, actions=2, observations=4, gamma=0.9, init_epsilon=1.0, final_epsilon=0.00001,
                 epsilon_decay=0.95, epsilon_anneal_steps=10, alpha=0.0001, replay_size=2000, mini_batch=256,
                 l1=128, l2=128, l3=128):
        self.actions = actions
        self.observations = observations
        self.gamma = gamma
        self.init_epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_anneal_steps = epsilon_anneal_steps
        self.alpha = alpha
        self.replay_size = replay_size
        self.mini_batch = mini_batch
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.w1 = self.weight_variable([observations, l1])
        self.b1 = self.bias_variable([l1])
        self.w2 = self.weight_variable([l1, l2])
        self.b2 = self.bias_variable([l2])
        self.w3 = self.weight_variable([l2, l3])
        self.b3 = self.bias_variable([l3])
        self.w4 = self.weight_variable([l3, actions])
        self.b4 = self.bias_variable([actions])

    # Initializes weights using xavier initialization
    def weights_initializer(self, dimensions):
        dimensions_sum = np.sum(dimensions)
        if len(dimensions) == 1:
            dimensions_sum += 1
        bound = np.sqrt(self.actions + self.observations / dimensions_sum)
        return tf.random_uniform(dimensions, minval=-bound, maxval=bound)

    # Creates weight variables
    def weight_variable(self, dimensions):
        return tf.Variable(self.weights_initializer(dimensions))

    # function to create bias variables
    def bias_variable(self, dimensions):
        return tf.Variable(self.weights_initializer(dimensions))

    # Adds options to the graph
    def add_value(self):
        observation = tf.placeholder(tf.float32, [None, self.observations])
        h1 = tf.nn.relu(tf.matmul(observation, self.w1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.w2) + self.b2)
        h3 = tf.nn.relu(tf.matmul(h2, self.w3) + self.b3)
        q = tf.squeeze(tf.matmul(h3, self.w4) + self.b4)
        return observation, q

    # Samples actions with random rate epsilon
    def get_action(self, Q, feed, epsilon):
        action_values = Q.eval(feed_dict=feed)
        if random.random() <= epsilon:
            # action_index = env.action_space.sample()
            action_index = random.randrange(self.actions)
        else:
            action_index = np.argmax(action_values)
        action = np.zeros(self.actions)
        action[action_index] = 1
        return action


def run(env, i):
    agent = DQNAgent()
    session = tf.InteractiveSession()

    obs, q1 = agent.add_value()

    act = tf.placeholder(tf.float32, [None, agent.actions])
    rwd = tf.placeholder(tf.float32, [None, ])

    next_obs, q2 = agent.add_value()

    values1 = tf.reduce_sum(tf.multiply(q1, act), reduction_indices=1)
    values2 = rwd + agent.gamma * tf.reduce_max(q2, reduction_indices=1)
    loss = tf.reduce_mean(tf.square(values1 - values2))
    train_step = tf.train.AdamOptimizer(agent.alpha).minimize(loss)

    session.run(tf.initialize_all_variables())

    # saving and loading networks
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("checkpoints-cartpole")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(session, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    epsilon = agent.init_epsilon
    global_step = 0
    exp_pointer = 0
    learning_finished = False
    feed = {}

    # replay memory
    observation_queue = np.empty([agent.replay_size, agent.observations])
    action_queue = np.empty([agent.replay_size, agent.actions])
    reward_queue = np.empty([agent.replay_size])
    next_observation_queue = np.empty([agent.replay_size, agent.observations])

    scores = deque(maxlen=100)

    # The episode loop
    for i_episode in range(3000):

        observation = env.reset()
        done = False
        score = 0
        sum_loss_value = 0

        # The step loop
        while not done:
            global_step += 1
            if global_step % agent.epsilon_anneal_steps == 0 and epsilon > agent.final_epsilon:
                epsilon = epsilon * agent.epsilon_decay
            # env.render()

            observation_queue[exp_pointer] = observation
            action = agent.get_action(q1, {obs: np.reshape(observation, (1, -1))}, epsilon)
            action_queue[exp_pointer] = action
            observation, reward, done, _ = env.step(np.argmax(action))

            score += reward
            reward = score

            if done and score < 200:
                reward = -500
                observation = np.zeros_like(observation)

            reward_queue[exp_pointer] = reward
            next_observation_queue[exp_pointer] = observation

            exp_pointer += 1
            if exp_pointer == agent.replay_size:
                exp_pointer = 0

            if global_step >= agent.replay_size:
                random_index = np.random.choice(agent.replay_size, agent.mini_batch)
                feed.update({obs: observation_queue[random_index]})
                feed.update({act: action_queue[random_index]})
                feed.update({rwd: reward_queue[random_index]})
                feed.update({next_obs: next_observation_queue[random_index]})
                if not learning_finished:
                    step_loss_value, _ = session.run([loss, train_step], feed_dict=feed)
                else:
                    step_loss_value = session.run(loss, feed_dict=feed)
                # Use sum to calculate average loss of this episode
                sum_loss_value += step_loss_value

        print("====== Episode {} ended with score = {}, avg_loss = {} ======".format(i_episode + 1, score,
                                                                                     sum_loss_value / score))
        scores.append(score)

        if np.mean(scores) > 195:  # The threshold of being solved
            learning_finished = True
            plt.scatter(x=i + 1, y=i_episode)
        else:
            learning_finished = False
        if learning_finished:
            break
            # print("Testing !!!")

        # save progress every 100 episodes
        if learning_finished and i_episode % 100 == 0:
            saver.save(session, 'checkpoints-cartpole/' + 'CartPole-v0' + '-dqn', global_step=global_step)


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    for i in range(10):
        run(env, i)
    plt.show()
