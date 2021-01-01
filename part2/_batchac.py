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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)          # actor's layer
        self.value_head = nn.Linear(128, 1)        # critic's layer
        self.saved_actions = []         # action & reward buffer
        self.rewards = []

    def forward(self, x):
        x = func.relu(self.affine1(x))
        action_prob = func.softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        return action_prob, state_values

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
    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
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
        policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        a = action.item()
        ####### End

        # Observe
        op, r, done, infos = env.step(a)

        # Learn
        ####### Start
        # Here goes your learning update
        policy.rewards.append(r)

        if (steps % 1000 == 0 and steps > 0) or done:
            R = 0
            saved_actions = policy.saved_actions
            policy_losses = []  # list to save actor (policy) loss
            value_losses = []  # list to save critic (value) loss
            returns = []  # list to save the true values

            for rr in policy.rewards[::-1]:             # calculate the true value using rewards returned from the environment
                R = rr + 0.99 * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)  # normalize

            for (log_prob, value), R in zip(saved_actions, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                value_losses.append(func.smooth_l1_loss(value, torch.tensor([R])))

            optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            loss.backward()
            optimizer.step()

            del policy.rewards[:]
            del policy.saved_actions[:]

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
