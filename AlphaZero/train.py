from collections import deque
from ConnectX import ConnectX
from Utils import calculate_winrate
from Agent import Agent_random, Agent_DNN

import tensorflow as tf
import MCTS
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# game parameter
game_row = 6
game_col = 7
game_state_size = game_row * game_col
game_action_size = game_col

# trainning parameter
episodes = 200
iterations = 200
outcomes = []
losses = []

# agent
agent = Agent_DNN()

# optimizer
actor_optimizer = tf.keras.optimizers.Adam(1e-4)
critic_optimizer = tf.keras.optimizers.Adam(1e-4)

# loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mean_squared_error = tf.keras.losses.MeanSquaredError()


def train_all(episodes, view_winrate=False):
    for step in range(episodes):

        mytree = MCTS.Node(ConnectX())

        # training dataset
        states = []
        vs = []
        probs = []
        masks = []

        while mytree.winner is None:
            for _ in range(iterations):
                mytree.explore(agent)

            current_player = mytree.game.get_player()
            state = mytree.game.get_state()

            mytree, allowed_actions, (v, nn_v, p, nn_p) = mytree.next()
            mytree.detach_mother()

            states.append(state)
            probs.append(p)
            masks.append(allowed_actions)
            vs.append(v * current_player)

        actor_loss, critic_loss = train_step(states, probs, masks, vs)
        actor_loss = actor_loss.numpy()
        critic_loss = critic_loss.numpy()
        print("#{}.. actor loss = {}, critic loss = {} ".format(step+1, actor_loss, critic_loss))
        if view_winrate:
            winrate, _, _ = calculate_winrate(ConnectX(), agent, Agent_random())
            print("...winrate = {}".format(winrate))


@tf.function
def train_step(states, probs, masks, vs):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        actor_loss = 0
        critic_loss = 0

        for state, prob, mask, v in zip(states, probs, masks, vs):
            nn_prob = agent.actor(state)
            nn_v = agent.critic(state)
            nn_mask = np.full((game_action_size), False)
            nn_prob = tf.reshape(nn_prob, [game_action_size])
            for a in mask:
                nn_mask[a] = True

            nn_prob = tf.boolean_mask(nn_prob, nn_mask)

            actor_loss += cross_entropy(nn_prob, prob)
            critic_loss += mean_squared_error(nn_v, v)

        actor_grad = actor_tape.gradient(actor_loss, agent.actor.trainable_variables)
        critic_grad = critic_tape.gradient(critic_loss, agent.critic.trainable_variables)

        actor_optimizer.apply_gradients(zip(actor_grad, agent.actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_grad, agent.critic.trainable_variables))

        return actor_loss, critic_loss


if __name__ == '__main__':
    winrate, _, _ = calculate_winrate(ConnectX(), agent, Agent_random())
    print("initial winrate = {}".format(winrate))

    train_all(400, view_winrate=True)
