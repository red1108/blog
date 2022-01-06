# %%

from collections import deque
import MCTS
import numpy as np
from ConnectX import ConnectX
from Agent import Agent_random, Agent_DNN
import tensorflow as tf
import progressbar as pb

# game parameter
game_row = 6
game_col = 7
game_state_size = game_row * game_col
game_action_size = game_col

# trainning parameter
episodes = 400
outcomes = []
losses = []

# widget and timer
widget = ['training loop: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

# agent
agent = Agent_DNN()

# optimizer
actor_optimizer = tf.keras.optimizers.Adam(1e-3)
critic_optimizer = tf.keras.optimizers.Adam(1e-3)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mean_squared_error = tf.keras.losses.MeanSquaredError()


def train_all(episodes):
    for e in range(episodes):

        mytree = MCTS.Node(ConnectX())
        vterm = []
        logterm = []
        actor_loss = 0

        # training dataset
        states = []
        vs = []
        probs = []
        masks = []

        while mytree.winner is None:
            for _ in range(50):
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
        print("loss = ", actor_loss, critic_loss)

        timer.update(e + 1)

    timer.finish()

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


train_all(400)
