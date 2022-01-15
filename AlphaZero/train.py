from collections import deque
from ConnectX import ConnectX
from Utils import MCTS_all_viewer, MCTS_node_viewer, print_comparison
from Agent import Agent_random, Agent_expert, Agent_DNN

import tensorflow as tf
import MCTS
import numpy as np

np.set_printoptions(linewidth=np.inf)

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
actor_optimizer = tf.keras.optimizers.Adam(1e-3)
critic_optimizer = tf.keras.optimizers.Adam(1e-3)

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
        h = 1

        while mytree.winner is None:
            for _ in range(iterations):
                mytree.explore(agent)

            current_player = mytree.game.get_player()
            state = mytree.game.get_state()

            #MCTS_node_viewer(mytree, state_print=True)

            mytree, allowed_actions, (v, nn_v, p, nn_p) = mytree.next(temperature=1.0)
            #print(p)
            h+=1
            mytree.detach_mother()

            states.append(state)
            probs.append(p)
            masks.append(allowed_actions)
            # vs.append(v * current_player)

        vs = [float(mytree.winner) for _ in probs]
        actor_loss, critic_loss = train_step(states, probs, masks, vs)
        actor_loss = actor_loss.numpy()
        critic_loss = critic_loss.numpy()
        print("#{}.. actor loss = {}, critic loss = {} ".format(step + 1, actor_loss, critic_loss))
        if view_winrate:
            print_comparison(ConnectX(), agent, Agent_random(), 200)
            print_comparison(ConnectX(), agent, Agent_expert(), 2)
            nn_probs, nn_value = agent.result(ConnectX())
            print("first probs = ", nn_probs)


# @tf.function
def train_step(states, probs, masks, vs):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        actor_loss = 0
        critic_loss = 0
        current_player = 1

        for state, prob, mask, v in zip(states, probs, masks, vs):
            nn_prob = agent.actor(state)
            nn_v = agent.critic(state) * current_player
            current_player = -current_player
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
    print_comparison(ConnectX(), agent, Agent_random(), 200)
    print_comparison(ConnectX(), agent, Agent_expert(), 4)

    train_all(episodes, view_winrate=True)
