# %%

from collections import deque
import MCTS
import numpy as np
from ConnectX import ConnectX
from Agent import Agent_random, Agent_DNN
import tensorflow as tf

episodes = 400
outcomes = []
losses = []

agent = Agent_DNN()

import progressbar as pb

widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA()]
timer = pb.ProgressBar(widgets=widget, maxval=episodes).start()

actor_optimizer = tf.keras.optimizers.Adam(1e-3)
critic_optimizer = tf.keras.optimizers.Adam(1e-3)


def train_step(episodes):
    for e in range(episodes):

        mytree = MCTS.Node(ConnectX())
        vterm = []
        logterm = []
        actor_loss = 0

        while mytree.winner is None:
            for _ in range(50):
                mytree.explore(agent)

            current_player = mytree.game.get_player()
            mytree, (v, nn_v, p, nn_p) = mytree.next()
            mytree.detach_mother()


            agent.actor.fit()
            actor_loss += actor_loss_fn(p, nn_p)
            '''
            # compute prob * log pi
            nn_p = np.array(nn_p).astype(np.float64)
            p = np.array(p).astype(np.float64)
            v = np.array(v).astype(np.float64)
            nn_v = np.array(nn_v).astype(np.float64)
            loglist = np.log(nn_p) * p
            #print("nn_p = ", nn_p, "p = ", p, "loglist = ", loglist, "log = ")

            # constant term to make sure if policy result = MCTS result, loss = 0
            constant = np.where(p > 0, p * np.log(p), 0.)
            logterm.append(-np.sum(loglist - constant))
            
            '''
            vterm.append(nn_v * current_player)

        # we compute the "policy_loss" for computing gradient

        outcome = mytree.winner
        outcomes.append(outcome)
        #actor_loss = np.sum(np.stack(logterm))
        critic_loss = critic_loss_fn(vterm, outcome)
        #critic_loss = np.sum((np.stack(vterm) - outcome) ** 2)
        print("actor_loss = ", actor_loss, "critic loss = ", critic_loss)

        # loss = np.sum( (np.stack(vterm)-outcome)**2 + np.stack(logterm) )
        grad_actor = actor_tape.gradient(actor_loss, agent.actor.trainable_variables)
        grad_critic = critic_tape.gradient(critic_loss, agent.critic.trainable_variables)

        actor_optimizer.apply_gradients(zip(grad_actor, agent.actor.trainable_variables))
        critic_optimizer.apply_gradients(zip(grad_critic, agent.critic.trainable_variables))

        losses.append(float(loss))

        if (e + 1) % 50 == 0:
            print("game: ", e + 1, ", mean loss: {:3.2f}".format(np.mean(losses[-20:])),
                  ", recent outcomes: ", outcomes[-10:])
        del loss

        timer.update(e + 1)

    timer.finish()

train_step(400)