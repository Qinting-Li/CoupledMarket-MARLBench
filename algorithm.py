"""MA-CETD3涓嶮ATD3绠楁硶瀹炵幇"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import deque
from config import ALGO_PARAMS, NUM_AGENTS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)


class AttentionCritic(nn.Module):
    """Critic network with attention over agent output features."""
    def __init__(self, state_dim, action_dim, num_agents=6, n_heads=2, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        self.num_agents = num_agents
        self.n_heads = n_heads
        agent_feat_dim = 1
        self.self_embed = nn.Linear(3 + action_dim, 32)
        self.other_embed = nn.Linear(agent_feat_dim, 32)
        self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=n_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(32)
        fused_dim = 32 + 32 + 2
        layers = []
        prev_dim = fused_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.value_net = nn.Sequential(*layers)

    def forward(self, states, actions):
        load = states[:, 0:1]
        agent_end = 1 + self.num_agents
        agent_outputs = states[:, 1:agent_end]
        carbon_green = states[:, agent_end:agent_end + 2]
        time_price = states[:, agent_end + 2:agent_end + 4]
        self_feat = torch.cat([load, time_price, actions], dim=-1)
        self_emb = F.relu(self.self_embed(self_feat)).unsqueeze(1)
        other_tokens = agent_outputs.unsqueeze(-1)
        other_emb = F.relu(self.other_embed(other_tokens))
        attn_out, _ = self.attn(self_emb, other_emb, other_emb)
        attn_out = self.attn_norm(attn_out.squeeze(1))
        fused = torch.cat([self_emb.squeeze(1), attn_out, carbon_green], dim=-1)
        return self.value_net(fused)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, rewards, next_state, done):
        self.buffer.append((state, actions, rewards, next_state, done))

    def sample(self, batch_size=128):
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).to(device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class MATD3Agent:
    def __init__(self, agent_id, state_dim, action_dim, num_agents=NUM_AGENTS):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=ALGO_PARAMS['lr_actor'])

        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=ALGO_PARAMS['lr_critic'])

        self.update_count = 0

    def select_action(self, state, noise_std=0.0):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()
        if noise_std > 0:
            action = action + np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action, -1, 1)
        return action

    def update(self, buffer, batch_size=128):
        if len(buffer) < batch_size:
            return
        states, actions_all, rewards, next_states, dones = buffer.sample(batch_size)

        my_actions = actions_all[:, self.agent_id * self.action_dim:
                                   (self.agent_id + 1) * self.action_dim]
        my_rewards = rewards[:, self.agent_id].unsqueeze(-1)

        with torch.no_grad():
            next_a = self.actor_target(next_states)
            noise = torch.clamp(torch.randn_like(next_a) * 0.2, -0.5, 0.5)
            next_a = torch.clamp(next_a + noise, -1, 1)
            q1_t = self.critic1_target(next_states, next_a)
            q2_t = self.critic2_target(next_states, next_a)
            target = my_rewards + 0.99 * (1 - dones.unsqueeze(-1)) * torch.min(q1_t, q2_t)

        q1 = self.critic1(states, my_actions)
        q2 = self.critic2(states, my_actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        if self.update_count % 2 == 0:
            my_a = self.actor(states)
            actor_loss = -self.critic1(states, my_a).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update()

    def _soft_update(self, tau=0.005):
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)
        for tp, p in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


class MACETD3Agent(MATD3Agent):
    def __init__(self, agent_id, state_dim, action_dim, num_agents=NUM_AGENTS):
        super().__init__(agent_id, state_dim, action_dim, num_agents)
        self.critic1 = AttentionCritic(state_dim, action_dim, num_agents=num_agents).to(device)
        self.critic2 = AttentionCritic(state_dim, action_dim, num_agents=num_agents).to(device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=ALGO_PARAMS['lr_critic'])
        self.cem_pop_size = 10
        self.cem_elite_num = 3
        self.cem_beta = ALGO_PARAMS['cem_beta']
        self.cem_sigma_min = ALGO_PARAMS['cem_sigma_min']
        self.lambda_cem = ALGO_PARAMS['lambda_init']
        self.lambda_decay = ALGO_PARAMS['lambda_decay']
        self.reward_history = deque(maxlen=200)
        self.adaptive_lambda_floor = 0.05
        self._init_cem_params()

    def _init_cem_params(self):
        self.cem_mean = self._get_actor_params()
        self.cem_var = np.ones_like(self.cem_mean) * 0.1

    def _get_actor_params(self):
        params = []
        for p in self.actor.parameters():
            params.append(p.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def _set_actor_params(self, flat_params):
        idx = 0
        for p in self.actor.parameters():
            size = p.data.numel()
            p.data.copy_(torch.FloatTensor(
                flat_params[idx:idx+size]).reshape(p.data.shape).to(device))
            idx += size

    def update(self, buffer, batch_size=128):
        if len(buffer) < batch_size:
            return
        states, actions_all, rewards, next_states, dones = buffer.sample(batch_size)

        my_actions = actions_all[:, self.agent_id * self.action_dim:
                                   (self.agent_id + 1) * self.action_dim]
        my_rewards = rewards[:, self.agent_id].unsqueeze(-1)

        with torch.no_grad():
            next_a = self.actor_target(next_states)
            noise = torch.clamp(torch.randn_like(next_a) * 0.2, -0.5, 0.5)
            next_a = torch.clamp(next_a + noise, -1, 1)
            q1_t = self.critic1_target(next_states, next_a)
            q2_t = self.critic2_target(next_states, next_a)
            target = my_rewards + 0.99 * (1 - dones.unsqueeze(-1)) * torch.min(q1_t, q2_t)

        q1 = self.critic1(states, my_actions)
        q2 = self.critic2(states, my_actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.update_count += 1
        if self.update_count % 2 == 0:
            my_a = self.actor(states)
            td3_loss = -self.critic1(states, my_a).mean()

            hybrid_loss = td3_loss
            self.actor_optimizer.zero_grad()
            hybrid_loss.backward()
            self.actor_optimizer.step()
            self._soft_update()
            self._adaptive_lambda_update()
            self.lambda_cem = max(self.lambda_cem * self.lambda_decay, self.adaptive_lambda_floor)

        cem_interval = self._adaptive_cem_interval()
        if self.update_count > 0 and self.update_count % cem_interval == 0 and self.lambda_cem > 0.08:
            self._cem_step(buffer)

    def _adaptive_lambda_update(self):
        """Adjust the CEM floor from recent reward dispersion."""
        if len(self.reward_history) < 50:
            return
        recent = np.array(list(self.reward_history)[-50:])
        reward_std = np.std(recent)
        reward_mean = np.mean(np.abs(recent)) + 1e-8
        cv = reward_std / reward_mean
        if cv > 0.3:
            self.adaptive_lambda_floor = min(0.15, self.adaptive_lambda_floor + 0.005)
        elif cv < 0.1:
            self.adaptive_lambda_floor = max(0.02, self.adaptive_lambda_floor - 0.005)

    def _adaptive_cem_interval(self):
        """Choose the CEM interval from recent reward dispersion."""
        if len(self.reward_history) < 50:
            return 500
        recent = np.array(list(self.reward_history)[-50:])
        cv = np.std(recent) / (np.mean(np.abs(recent)) + 1e-8)
        if cv > 0.3:
            return 300
        elif cv > 0.15:
            return 500
        else:
            return 800

    def record_reward(self, reward):
        self.reward_history.append(reward)

    def _cem_step(self, buffer):
        if len(buffer) < 128:
            return
        current_params = self._get_actor_params()
        self.cem_mean = current_params.copy()

        eval_states, _, _, _, _ = buffer.sample(128)
        with torch.no_grad():
            curr_a = self.actor(eval_states)
            curr_q = torch.min(self.critic1(eval_states, curr_a),
                               self.critic2(eval_states, curr_a)).mean().item()

        pop_rewards = []
        pop_params = []
        for _ in range(self.cem_pop_size):
            noise = np.random.randn(*self.cem_mean.shape) * np.sqrt(np.abs(self.cem_var))
            candidate = self.cem_mean + noise * self.lambda_cem * 0.1
            pop_params.append(candidate)
            self._set_actor_params(candidate)

            with torch.no_grad():
                a = self.actor(eval_states)
                q = torch.min(self.critic1(eval_states, a),
                              self.critic2(eval_states, a)).mean().item()
            pop_rewards.append(q)

        pop_rewards = np.array(pop_rewards)
        elite_idx = np.argsort(pop_rewards)[-self.cem_elite_num:]
        best_q = pop_rewards[elite_idx[-1]]

        if best_q > curr_q:
            elite_params = np.array([pop_params[i] for i in elite_idx])
            new_mean = np.mean(elite_params, axis=0)
            alpha = min(self.lambda_cem, 0.3)
            blended = alpha * new_mean + (1 - alpha) * current_params
            self._set_actor_params(blended)
            self.cem_mean = blended.copy()

            new_var = np.var(elite_params, axis=0)
            self.cem_var = (1 - self.cem_beta) * new_var + self.cem_beta * self.cem_sigma_min**2
        else:
            self._set_actor_params(current_params)


class MultiAgentSystem:
    def __init__(self, state_dim, action_dim, num_agents, algorithm='MA-CETD3'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.algorithm = algorithm
        self.buffer = ReplayBuffer(capacity=50000)

        AgentClass = MACETD3Agent if algorithm == 'MA-CETD3' else MATD3Agent
        self.agents = [AgentClass(i, state_dim, action_dim, num_agents)
                       for i in range(num_agents)]

        self.noise_std = ALGO_PARAMS['noise_std']
        self.episode_count = 0

    def select_actions(self, states, explore=True):
        actions = {}
        for i, agent in enumerate(self.agents):
            noise = self.noise_std if explore else 0.0
            actions[i] = agent.select_action(states, noise)
        return actions

    def store_transition(self, state, actions, rewards, next_state, done):
        flat_actions = np.concatenate([actions[i] for i in range(self.num_agents)])
        self.buffer.push(state, flat_actions, rewards, next_state, float(done))

    def update(self):
        if len(self.buffer) < 128:
            return
        for agent in self.agents:
            agent.update(self.buffer)

    def decay_noise(self):
        self.noise_std *= ALGO_PARAMS['noise_decay']
        self.noise_std = max(self.noise_std, 0.02)
        self.episode_count += 1
