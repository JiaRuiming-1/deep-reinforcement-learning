import torch
import torch.nn as nn
import torch.nn.functional as F

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class FCBody(nn.Module):
    def __init__(self, state_dim, hidden_units=(64, 64), gate=F.relu):
        super(FCBody, self).__init__()
        dims = (state_dim,) + hidden_units

        self.layers = nn.ModuleList(
            [layer_init(nn.Linear(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        self.gate = gate
        self.feature_dim = dims[-1]

    def forward(self, x):
        for layer in self.layers:
            x = self.gate(layer(x))
        return x


class DDPGActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim,
                 actor_opt_fn, critic_opt_fn
                 ):
        super(DDPGActorCriticNet, self).__init__()

        self.actor_body = FCBody(state_dim, (256, 256), gate=F.relu)
        self.critic_body = FCBody(state_dim + action_dim, (256, 256), gate=F.relu)
        self.fc_action = layer_init(nn.Linear(self.actor_body.feature_dim, action_dim), 1e-3)
        self.fc_critic = layer_init(nn.Linear(self.critic_body.feature_dim, 1), 1e-3)

        self.actor_opt = actor_opt_fn(list(self.actor_body.parameters()) + list(self.fc_action.parameters()))
        self.critic_opt = critic_opt_fn(list(self.critic_body.parameters()) + list(self.fc_critic.parameters()))

    def forward(self, obs):
        action = torch.tanh(self.fc_action(self.actor_body(obs)))
        return action

    def critic(self, phi, a):
        return self.fc_critic(self.critic_body(torch.cat([phi, a], dim=1)))