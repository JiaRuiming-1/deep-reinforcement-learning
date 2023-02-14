import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

        self.fc1_units = fc1_units

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc1(state)
        tmp_shape = x.shape
        xa = x.view(-1, self.fc1_units)
        xa = F.relu(self.bn(xa).view(tmp_shape))
        xa = F.relu(self.fc2(xa))
        return torch.tanh(self.fc3(xa))



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=400, fc2_units=300):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn = nn.BatchNorm1d(fcs1_units)
        self.fc1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

        self.fcs1_units = fcs1_units

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Build a critic (value) network that maps
        (state, action) pairs -> Q-values
        :param state: tuple.
        :param action: tuple.
        """
        x = self.fc1(state)
        tmp_shape = x.shape
        xc = x.view(-1,self.fcs1_units)
        xc = F.relu(self.bn(xc).view(tmp_shape))
        x = F.relu(self.fc2(xc))
        return self.fc3(x)

# PPO continuous
class Policy_Gaussian(nn.Module):
    '''
    '''

    def __init__(self, state_size, action_size, seed, fc1=256, fc2=256):
        '''

        '''
        super(Policy_Gaussian, self).__init__()
        self.actor_body = Actor(state_size, action_size, seed, fc1_units=fc1, fc2_units=fc2)
        self.critic_body = Critic(state_size, action_size, seed, fcs1_units=fc1, fc2_units=fc2)
        self.std = nn.Parameter(torch.ones(1, action_size))

        self.actor_params = list(self.actor_body.parameters())
        self.actor_params.append(self.std)

    def forward(self, states, actions=None):

        estimated_values = self.critic_body(states)
        mean = self.actor_body(states)
        # pdb.set_trace()
        dist = torch.distributions.Normal(mean, self.std)
        if isinstance(actions, type(None)):
            actions = dist.sample()

        log_prob = dist.log_prob(actions).sum(-1, keepdim=True)
        entropy_loss = dist.entropy().sum(-1, keepdim=True)

        return {'action': actions,       #(n, 20, 4)
                'log_pi_a': log_prob,    #(n, 20, 1) sum
                'entropy': entropy_loss, #(n, 20, 1) sum
                'mean': mean,            #(n, 20, 4)
                'v': estimated_values}   #(n, 20, 1)


# test model
if __name__ == '__main__':
    state1 = torch.tensor([[[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037],[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037],[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037]],[[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037],[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037],[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037]]], dtype=torch.float32)

    state2 = torch.tensor([[[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037]],[[-0.0943, -3.9967,  0.1509,  0.9998, -0.0117,  0.0002,  0.0188,
         -0.7456, -0.0106, -0.4663, -1.8749,  0.0945,  2.9961, -0.0495,
         -9.9861,  0.0396,  0.9991,  0.0192,  0.0017, -0.0373,  1.4857,
         -0.0976,  0.7593,  0.1247,  0.5414, -1.3702,  3.5088, -1.0000,
         -7.1895,  0.0000,  1.0000,  0.0000, -0.0037]]], dtype=torch.float32)
    net = Policy_Gaussian(33, 4, 0, fc1=256, fc2=256)
    net.eval()
    action = torch.tensor([[[ 0.2394,  2.2476, -0.1988, -0.2694],[ 0.2394,  2.2476, -0.1988, -0.2694]],[[ 0.2394,  2.2476, -0.1988, -0.2694],[ 0.2394,  2.2476, -0.1988, -0.2694]]])
    pred = net(state1)
    net.train()
    print(pred)
    print(state1.shape)
    print(pred['entropy'].shape, pred['log_pi_a'].shape)