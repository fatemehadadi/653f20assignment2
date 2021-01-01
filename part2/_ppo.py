import sys
import numpy as np
import matplotlib as mpl

mpl.use("TKAgg")
import matplotlib.pyplot as plt
import gym
import torch

# personal import
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple
from random import shuffle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        self.action_head = nn.Linear(128, 2)

        self.value_head = nn.Linear(128, 1)

        self.saved_states = []
        self.saved_actions = []
        self.rewards = []
        self.dones = []

    def forward(self, x):
        x = func.relu(self.affine1(x))
        action_prob = func.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values


def random_sample(inds, minibatch_size):
    inds = np.random.permutation(inds)
    batches = inds[:len(inds) // minibatch_size * minibatch_size].reshape(-1, minibatch_size)
    for batch in batches:
        yield torch.from_numpy(batch).long()
    r = len(inds) % minibatch_size
    if r:
        yield torch.from_numpy(inds[-r:]).long()

def main():
    seed = 0 if len(sys.argv) == 1 else int(sys.argv[1])

    # Task setup block starts
    # Do not change
    env = gym.make('CartPole-v1')
    env.seed(seed)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    # Task setup block end

    # Learner setup block
    torch.manual_seed(seed)
    ####### Start
    policy = Policy()
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    eps = np.finfo(np.float32).eps.item()
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'action'])
    beta = 0.01
    cliprange = 0.1
    ####### End

    # Experiment block starts
    ret = 0
    rets = []
    avgrets = []
    o = env.reset()
    num_steps = 500000
    checkpoint = 10000
    for steps in range(num_steps):

        # Select an action
        ####### Start
        # Replace the following statement with your own code for
        # selecting an action
        policy.eval()
        if steps == 0:
            op = o
        op = torch.from_numpy(op).float().unsqueeze(0)
        probs, state_value = policy(op)
        m = Categorical(probs)
        action = m.sample()
        policy.saved_states.append(op)
        policy.saved_actions.append(SavedAction(m.log_prob(action), state_value, action))
        a = action.item()
        ####### End

        # Observe
        op, r, done, infos = env.step(a)

        # Learn
        ####### Start
        # Here goes your learning update
        policy.rewards.append(r)
        if done:
            policy.dones.append(1)
        else:
            policy.dones.append(0)
        torch.autograd.set_detect_anomaly(True)
        if (steps % 500 == 0 and steps > 0):
            R = 0
            saved_actions = policy.saved_actions
            saved_states = policy.saved_states
            policy_losses = []
            value_losses = []
            advantages = np.zeros_like(policy.rewards)
            returns = [0]*len(policy.rewards)

            for t in reversed(range(len(policy.rewards))):
                if t == len(policy.rewards)-1:
                    returns[t] = returns[t] + 0.99 * (1 - policy.dones[t]) * state_value
                else:
                    returns[t] = returns[t] + 0.99 * (1 - policy.dones[t]) * returns[t+1]
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)  # normalize the return


            for (log_prob, value, _), R in zip(saved_actions, returns):
                advantage = R - value.item()
                np.append(advantages, advantage)

            advantages = torch.from_numpy(advantages).float().to(device).view(-1, )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

            n_actions = len(saved_actions)
            size_minibatch = 100
            epochs = n_actions //size_minibatch
            if epochs == 0:
                size_minibatch = n_actions
                epochs = 1
            sample = [i for i in range(n_actions)]
            shuffle(sample)
            for e in range(epochs):
                batch_sample = sample[e:e+size_minibatch]
                for index in batch_sample:
                    minibatch_action = saved_actions[index]
                    minibatch_log_prob = minibatch_action[0]
                    minibatch_return = returns[index]
                    minibatch_advantage = advantages[index]
                    prob, v = policy(saved_states[index])
                    dist = Categorical(prob)
                    if minibatch_action[2] is None:
                        minibatch_action[2] = dist.sample()
                    log_prob = dist.log_prob(minibatch_action[2])
                    ratio = torch.exp(log_prob - minibatch_log_prob)
                    surr1 = ratio * minibatch_advantage
                    surr2 = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange) * minibatch_advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * (minibatch_return - v).pow(2).mean()
                    policy_losses.append(policy_loss)
                    value_losses.append(value_loss)
                    print(steps, index, e)
                    optimizer.zero_grad()

                optimizer.zero_grad()
                loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
                loss.backward(retain_graph=True)
                optimizer.step()
                policy_losses = []
                value_losses = []

            del policy.rewards[:]
            del policy.dones[:]
            del policy.saved_actions[:]
            del policy.saved_states[:]

        ####### End

        # Log
        ret += r
        if done:
            rets.append(ret)
            ret = 0
            o = env.reset()

        if (steps + 1) % checkpoint == 0:
            print(np.mean(rets))
            avgrets.append(np.mean(rets))
            rets = []
            plt.clf()
            plt.plot(range(checkpoint, (steps + 1) + checkpoint, checkpoint), avgrets)
            plt.pause(0.001)

    name = sys.argv[0].split('.')[-2].split('_')[-1]
    data = np.zeros((2, len(avgrets)))
    data[0] = range(checkpoint, num_steps + 1, checkpoint)
    data[1] = avgrets
    np.savetxt(name + str(seed) + ".txt", data)
    plt.show()


if __name__ == "__main__":
    main()
