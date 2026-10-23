# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_NAME = "hyperjet"
RUN_TAG = "hyperjet_50"
NUM_USERS = 50
CHECKPOINT_INTERVAL = 50

current_dir = os.path.dirname(os.path.abspath(__file__))
methods_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(methods_dir)
env_dir = os.path.join(project_root, '02_environment')
utilities_dir = os.path.join(project_root, '05_analysis_tools', 'utilities')
results_root = os.path.join(project_root, '06_results_archive')

for path in (project_root, env_dir, utilities_dir):
    if path not in sys.path:
        sys.path.append(path)

log_dir = os.path.join(results_root, 'logs', METHOD_NAME)
model_dir = os.path.join(results_root, 'models', METHOD_NAME)
figure_dir = os.path.join(results_root, 'figures', 'training_curves', METHOD_NAME)
summary_dir = os.path.join(results_root, 'summaries')
for directory in (log_dir, model_dir, figure_dir, summary_dir):
    os.makedirs(directory, exist_ok=True)

log_file = os.path.join(log_dir, f'train_{RUN_TAG}.log')
checkpoint_path = os.path.join(model_dir, f'checkpoint_{RUN_TAG}.pth')
best_model_path = os.path.join(model_dir, f'best_model_{RUN_TAG}.pth')
curve_path = os.path.join(figure_dir, f'training_curve_{RUN_TAG}.png')
epoch_metrics_file = os.path.join(summary_dir, METHOD_NAME, f'epoch_metrics_{RUN_TAG}.csv')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a',
    force=True,
)

from env_hierarchical import HierarchicalDistributedSystemEnv
from experiment_record_utils import (
    build_epoch_records,
    build_epoch_stats,
    write_epoch_metrics_csv,
    write_experiment_record,
)


class Config:
    LR = 3e-4
    GAMMA = 0.99
    EPOCHS = 1000
    HIDDEN_DIM = 128
    SEED = 42
    ENTROPY_BETA_START = 0.01
    ENTROPY_BETA_MIN = 0.001
    ENTROPY_DECAY = 0.999
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.5
    HGNN_LAYERS = 2
    HGNN_HIDDEN_DIM = 64
    PPO_EPOCHS = 4
    CLIP_EPS = 0.2
    GAE_LAMBDA = 0.95
    MINI_BATCH_SIZE = 64
    UPDATE_TIMESTEP = 200


random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class HypergraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(HypergraphConv, self).__init__()
        self.theta = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, H):
        d_v = H.sum(dim=1)
        d_e = H.sum(dim=0)
        d_v_inv_sqrt = d_v.pow(-0.5)
        d_v_inv_sqrt[d_v_inv_sqrt == float('inf')] = 0
        d_e_inv = d_e.pow(-1.0)
        d_e_inv[d_e_inv == float('inf')] = 0
        x_trans = self.theta(x)
        x_trans = x_trans * d_v_inv_sqrt.unsqueeze(1)
        msg_edges = torch.mm(H.t(), x_trans)
        msg_edges = msg_edges * d_e_inv.unsqueeze(1)
        msg_nodes = torch.mm(H, msg_edges)
        out = msg_nodes * d_v_inv_sqrt.unsqueeze(1)
        return F.relu(out + self.bias)


class HGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers=2):
        super(HGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(HypergraphConv(input_dim, hidden_dim))
        for _ in range(layers - 1):
            self.convs.append(HypergraphConv(hidden_dim, hidden_dim))

    def forward(self, x, H):
        for conv in self.convs:
            x = conv(x, H)
        return x


class Actor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.output(x).squeeze(-1)


class PPOMemory:
    def __init__(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.tier_indices = []
        self.masks = []
        self.task_indices = []

    def store(self, state, action, log_prob, reward, next_state, done, tier_idx, mask, task_idx):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.tier_indices.append(tier_idx)
        self.masks.append(mask)
        self.task_indices.append(task_idx)

    def clear(self):
        self.states = []
        self.next_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.tier_indices = []
        self.masks = []
        self.task_indices = []

    def get_tensors(self, device):
        max_len = 0
        if self.masks:
            max_len = max(len(m) for m in self.masks)

        padded_masks = []
        for m in self.masks:
            if len(m) < max_len:
                pad = np.zeros(max_len - len(m), dtype=bool)
                padded_masks.append(np.concatenate([m, pad]))
            else:
                padded_masks.append(m)

        return (
            torch.FloatTensor(np.array(self.states)).to(device),
            torch.LongTensor(np.array(self.actions)).to(device),
            torch.FloatTensor(np.array(self.log_probs)).to(device),
            torch.FloatTensor(np.array(self.rewards)).to(device),
            torch.FloatTensor(np.array(self.next_states)).to(device),
            torch.FloatTensor(np.array(self.dones)).to(device),
            torch.LongTensor(np.array(self.tier_indices)).to(device),
            torch.BoolTensor(np.array(padded_masks)).to(device),
            torch.LongTensor(np.array(self.task_indices)).to(device),
        )


def build_advanced_hypergraph(env, device):
    tasks = list(env.task_map.values())
    tasks.sort(key=lambda t: t.unique_id)
    task_to_idx = {t.unique_id: i for i, t in enumerate(tasks)}
    N = len(tasks)

    features = []
    for t in tasks:
        features.append([
            t.wf_runtime_sec / 100.0,
            t.core_count / 4.0,
            t.req_mem / 8.0,
            float(t.privacy_level) / 3.0,
            t.req_gpu_mem / 1.0,
            t.data_input_size / 1e7,
            t.data_output_size / 1e7,
            t.static_remaining_time / 100.0,
        ])
    X = torch.FloatTensor(features).to(device)

    hyperedges = []
    for t in tasks:
        for p_name in t.parents:
            if p_name in env.task_map:
                p_id = env.task_map[p_name].unique_id
                if p_id in task_to_idx:
                    hyperedges.append([task_to_idx[p_id], task_to_idx[t.unique_id]])

    privacy_groups = defaultdict(list)
    for i, t in enumerate(tasks):
        privacy_groups[t.privacy_level].append(i)
    for indices in privacy_groups.values():
        if len(indices) > 1:
            hyperedges.append(indices)

    data_groups = defaultdict(list)
    for i, t in enumerate(tasks):
        size_mb = t.data_input_size / (1024 * 1024)
        if size_mb < 1:
            bin_idx = 0
        elif size_mb < 10:
            bin_idx = 1
        else:
            bin_idx = 2
        data_groups[bin_idx].append(i)
    for indices in data_groups.values():
        if len(indices) > 1:
            hyperedges.append(indices)

    M = len(hyperedges)
    H = torch.zeros(N, M).to(device)
    for m, edge_indices in enumerate(hyperedges):
        for n in edge_indices:
            H[n, m] = 1.0

    return X, H, task_to_idx


def build_env():
    env_kwargs = {'time_interval': 10.0}
    if NUM_USERS is not None:
        env_kwargs['num_users'] = NUM_USERS
    return HierarchicalDistributedSystemEnv(**env_kwargs)


def get_effective_num_users():
    return 100 if NUM_USERS is None else NUM_USERS


def build_run_record(best_epoch_idx, epoch_rewards, epoch_r_latency, epoch_r_ddl, epoch_r_slack,
                     epoch_avg_wf_time, epoch_ddl_rate, epoch_avg_path_qoe):
    config_dict = {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
    return {
        'method': METHOD_NAME,
        'script_name': os.path.basename(__file__),
        'run_tag': RUN_TAG,
        'load': {
            'label': f'{get_effective_num_users()}_users',
            'num_users': get_effective_num_users(),
        },
        'training': {
            'seed': Config.SEED,
            'epochs': Config.EPOCHS,
            'learning_rate': Config.LR,
            'key_hyperparameters': config_dict,
        },
        'best_model': build_epoch_stats(
            best_epoch_idx,
            epoch_rewards,
            epoch_r_latency,
            epoch_r_ddl,
            epoch_r_slack,
            epoch_avg_wf_time,
            epoch_ddl_rate,
            epoch_avg_path_qoe,
        ),
        'artifacts': {
            'log_file': log_file,
            'checkpoint_file': checkpoint_path,
            'best_model_file': best_model_path,
            'epoch_metrics_file': epoch_metrics_file,
            'figure_files': [curve_path],
        },
    }


def save_curve(epoch_rewards):
    plt.figure()
    plt.plot(epoch_rewards)
    plt.title('HyperJet-MAPPO Training Reward')
    plt.savefig(curve_path)
    plt.close()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    print(f'Using device: {device}')
    print(f'Results root: {results_root}')

    env = build_env()
    state_dim = 121
    hgnn_dim = Config.HGNN_HIDDEN_DIM
    input_dim = state_dim + hgnn_dim

    hgnn = HGNN(8, hgnn_dim, Config.HGNN_LAYERS).to(device)
    agent_local = Actor(input_dim, 100, Config.HIDDEN_DIM).to(device)
    agent_edge = Actor(input_dim, 10, Config.HIDDEN_DIM).to(device)
    agent_cloud = Actor(input_dim, 2, Config.HIDDEN_DIM).to(device)
    agents = [agent_local, agent_edge, agent_cloud]
    critic = Critic(input_dim, Config.HIDDEN_DIM).to(device)

    all_params = list(agent_local.parameters()) + list(agent_edge.parameters()) + list(agent_cloud.parameters()) + list(critic.parameters()) + list(hgnn.parameters())
    opt_local = optim.Adam(agent_local.parameters(), lr=Config.LR)
    opt_edge = optim.Adam(agent_edge.parameters(), lr=Config.LR)
    opt_cloud = optim.Adam(agent_cloud.parameters(), lr=Config.LR)
    opt_critic = optim.Adam(list(critic.parameters()) + list(hgnn.parameters()), lr=Config.LR)
    sched_local = optim.lr_scheduler.StepLR(opt_local, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    sched_edge = optim.lr_scheduler.StepLR(opt_edge, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    sched_cloud = optim.lr_scheduler.StepLR(opt_cloud, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    sched_critic = optim.lr_scheduler.StepLR(opt_critic, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    schedulers = [sched_local, sched_edge, sched_cloud]

    print('Starting HyperJet-MAPPO Training...')

    memory = PPOMemory()
    start_epoch = 0
    epoch_rewards = []
    epoch_r_latency = []
    epoch_r_ddl = []
    epoch_r_slack = []
    epoch_avg_wf_time = []
    epoch_ddl_rate = []
    epoch_avg_path_qoe = []
    epoch_steps = []
    best_reward = -float('inf')
    best_epoch_idx = -1
    entropy_beta = Config.ENTROPY_BETA_START

    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent_local.load_state_dict(checkpoint['agent_local'])
        agent_edge.load_state_dict(checkpoint['agent_edge'])
        agent_cloud.load_state_dict(checkpoint['agent_cloud'])
        critic.load_state_dict(checkpoint['critic'])
        hgnn.load_state_dict(checkpoint['hgnn'])
        opt_local.load_state_dict(checkpoint['opt_local'])
        opt_edge.load_state_dict(checkpoint['opt_edge'])
        opt_cloud.load_state_dict(checkpoint['opt_cloud'])
        opt_critic.load_state_dict(checkpoint['opt_critic'])
        sched_local.load_state_dict(checkpoint['sched_local'])
        sched_edge.load_state_dict(checkpoint['sched_edge'])
        sched_cloud.load_state_dict(checkpoint['sched_cloud'])
        sched_critic.load_state_dict(checkpoint['sched_critic'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_reward = checkpoint.get('best_reward', best_reward)
        entropy_beta = checkpoint.get('entropy_beta', entropy_beta)
        epoch_rewards = checkpoint.get('epoch_rewards', epoch_rewards)
        epoch_r_latency = checkpoint.get('epoch_r_latency', epoch_r_latency)
        epoch_r_ddl = checkpoint.get('epoch_r_ddl', epoch_r_ddl)
        epoch_r_slack = checkpoint.get('epoch_r_slack', epoch_r_slack)
        epoch_avg_wf_time = checkpoint.get('epoch_avg_wf_time', epoch_avg_wf_time)
        epoch_ddl_rate = checkpoint.get('epoch_ddl_rate', epoch_ddl_rate)
        epoch_avg_path_qoe = checkpoint.get('epoch_avg_path_qoe', epoch_avg_path_qoe)
        epoch_steps = checkpoint.get('epoch_steps', epoch_steps)
        best_epoch_idx = checkpoint.get('best_epoch_idx', best_epoch_idx)
        print(f'Resuming from epoch {start_epoch}, Best Reward: {best_reward:.4f}')

    for epoch in range(start_epoch, Config.EPOCHS):
        state = env.reset()
        masks = env._get_action_mask()
        X, H, task_to_idx = build_advanced_hypergraph(env, device)

        with torch.no_grad():
            hgnn_embs = hgnn(X, H)

        done = False
        total_reward = 0.0
        steps = 0
        total_r_latency = 0.0
        total_r_ddl = 0.0
        total_r_slack = 0.0
        final_metrics = {}

        while not done:
            curr_task = env.current_task
            if curr_task and curr_task.unique_id in task_to_idx:
                t_idx = task_to_idx[curr_task.unique_id]
                curr_emb = hgnn_embs[t_idx].unsqueeze(0)
            else:
                t_idx = -1
                curr_emb = torch.zeros(1, hgnn_dim).to(device)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            enhanced_state = torch.cat([state_tensor, curr_emb], dim=1)

            logits_local = agent_local(enhanced_state)
            mask_local_t = torch.BoolTensor(masks['local']).unsqueeze(0).to(device)
            logits_local = logits_local.masked_fill(~mask_local_t, -1e9)
            dist_local = torch.distributions.Categorical(logits=logits_local)
            prob_local = dist_local.probs.max()

            logits_edge = agent_edge(enhanced_state)
            mask_edge_t = torch.BoolTensor(masks['edge']).unsqueeze(0).to(device)
            logits_edge = logits_edge.masked_fill(~mask_edge_t, -1e9)
            dist_edge = torch.distributions.Categorical(logits=logits_edge)
            prob_edge = dist_edge.probs.max()

            logits_cloud = agent_cloud(enhanced_state)
            mask_cloud_t = torch.BoolTensor(masks['cloud']).unsqueeze(0).to(device)
            logits_cloud = logits_cloud.masked_fill(~mask_cloud_t, -1e9)
            dist_cloud = torch.distributions.Categorical(logits=logits_cloud)
            prob_cloud = dist_cloud.probs.max()

            epsilon = max(0.05, 1.0 - epoch / 200.0)
            if random.random() < epsilon:
                tier_idx = random.randint(0, 2)
            else:
                probs = [prob_local.item(), prob_edge.item(), prob_cloud.item()]
                tier_idx = int(np.argmax(probs))

            if tier_idx == 0:
                dist = dist_local
                mask_np = masks['local']
            elif tier_idx == 1:
                dist = dist_edge
                mask_np = masks['edge']
            else:
                dist = dist_cloud
                mask_np = masks['cloud']

            action = dist.sample()
            log_prob = dist.log_prob(action)
            node_idx = action.item()

            next_state, reward, done, next_masks, info = env.step(tier_idx, node_idx)
            memory.store(state, node_idx, log_prob.item(), reward, next_state, done, tier_idx, mask_np, t_idx)

            state = next_state
            masks = next_masks
            total_reward += reward
            total_r_latency += info.get('r_latency', 0.0)
            total_r_ddl += info.get('r_ddl', 0.0)
            total_r_slack += info.get('r_slack', 0.0)
            final_metrics = info
            steps += 1

        if len(memory.states) > 0:
            states, actions, old_log_probs, rewards, next_states, dones, tier_indices, batch_masks, task_indices = memory.get_tensors(device)
            hgnn_embs_grad = hgnn(X, H)

            valid_idx_mask = (task_indices != -1)
            safe_indices = task_indices.clone()
            safe_indices[~valid_idx_mask] = 0
            gathered_embs = hgnn_embs_grad[safe_indices]
            gathered_embs[~valid_idx_mask] = 0
            enhanced_states = torch.cat([states, gathered_embs], dim=1)
            next_gathered_embs = torch.cat([gathered_embs[1:], torch.zeros(1, hgnn_dim).to(device)], dim=0)
            enhanced_next_states = torch.cat([next_states, next_gathered_embs], dim=1)

            with torch.no_grad():
                values = critic(enhanced_states)
                next_values = critic(enhanced_next_states)

            advantages = torch.zeros_like(rewards).to(device)
            last_gae_lam = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_val = next_values[t]
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_val = values[t + 1]

                delta = rewards[t] + Config.GAMMA * next_val * next_non_terminal - values[t]
                advantages[t] = last_gae_lam = delta + Config.GAMMA * Config.GAE_LAMBDA * next_non_terminal * last_gae_lam

            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            dataset_size = len(states)
            indices = np.arange(dataset_size)
            for _ in range(Config.PPO_EPOCHS):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, Config.MINI_BATCH_SIZE):
                    hgnn_embs_batch = hgnn(X, H)
                    end = start + Config.MINI_BATCH_SIZE
                    mb_idx = indices[start:end]

                    mb_states = states[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_log_probs = old_log_probs[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_advantages = advantages[mb_idx]
                    mb_tiers = tier_indices[mb_idx]
                    mb_masks = batch_masks[mb_idx]
                    mb_task_indices = task_indices[mb_idx]

                    valid_mask_b = (mb_task_indices != -1)
                    safe_indices_b = mb_task_indices.clone()
                    safe_indices_b[~valid_mask_b] = 0
                    mb_embs = hgnn_embs_batch[safe_indices_b]
                    mb_embs[~valid_mask_b] = 0
                    mb_enhanced_states = torch.cat([mb_states, mb_embs], dim=1)

                    opt_local.zero_grad()
                    opt_edge.zero_grad()
                    opt_cloud.zero_grad()
                    opt_critic.zero_grad()

                    new_values = critic(mb_enhanced_states)
                    critic_loss = F.mse_loss(new_values, mb_returns)
                    critic_loss.backward(retain_graph=True)

                    tier_batches = [idx for idx in range(3) if (mb_tiers == idx).sum() > 0]
                    for pos, t_idx in enumerate(tier_batches):
                        tier_mask = (mb_tiers == t_idx)
                        t_states = mb_enhanced_states[tier_mask]
                        t_actions = mb_actions[tier_mask]
                        t_old_log_probs = mb_old_log_probs[tier_mask]
                        t_advantages = mb_advantages[tier_mask]
                        t_masks = mb_masks[tier_mask]

                        agent = agents[t_idx]
                        logits = agent(t_states)
                        valid_len = logits.shape[1]
                        t_masks_sliced = t_masks[:, :valid_len]
                        logits = logits.masked_fill(~t_masks_sliced, -1e9)
                        dist = torch.distributions.Categorical(logits=logits)
                        new_log_probs = dist.log_prob(t_actions)
                        entropy = dist.entropy().mean()
                        ratio = torch.exp(new_log_probs - t_old_log_probs)
                        surr1 = ratio * t_advantages
                        surr2 = torch.clamp(ratio, 1.0 - Config.CLIP_EPS, 1.0 + Config.CLIP_EPS) * t_advantages
                        actor_loss = -torch.min(surr1, surr2).mean() - entropy_beta * entropy

                        if pos == len(tier_batches) - 1:
                            actor_loss.backward()
                        else:
                            actor_loss.backward(retain_graph=True)

                    torch.nn.utils.clip_grad_norm_(all_params, 0.5)
                    opt_local.step()
                    opt_edge.step()
                    opt_cloud.step()
                    opt_critic.step()

            memory.clear()

        avg_reward = total_reward / steps if steps > 0 else 0.0
        avg_r_latency = total_r_latency / steps if steps > 0 else 0.0
        avg_r_ddl = total_r_ddl / steps if steps > 0 else 0.0
        avg_r_slack = total_r_slack / steps if steps > 0 else 0.0
        epoch_rewards.append(avg_reward)
        epoch_r_latency.append(avg_r_latency)
        epoch_r_ddl.append(avg_r_ddl)
        epoch_r_slack.append(avg_r_slack)
        epoch_avg_wf_time.append(final_metrics.get('avg_wf_time', 0.0))
        epoch_ddl_rate.append(final_metrics.get('ddl_rate', 0.0))
        epoch_avg_path_qoe.append(final_metrics.get('avg_path_qoe', 0.0))
        epoch_steps.append(steps)
        write_epoch_metrics_csv(
            summary_dir,
            METHOD_NAME,
            RUN_TAG,
            build_epoch_records(
                epoch_rewards,
                epoch_r_latency,
                epoch_r_ddl,
                epoch_r_slack,
                epoch_avg_wf_time,
                epoch_ddl_rate,
                epoch_avg_path_qoe,
            ),
        )

        for scheduler in schedulers:
            scheduler.step()
        sched_critic.step()
        entropy_beta = max(Config.ENTROPY_BETA_MIN, entropy_beta * Config.ENTROPY_DECAY)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_epoch_idx = epoch
            torch.save({
                'agent_local': agent_local.state_dict(),
                'agent_edge': agent_edge.state_dict(),
                'agent_cloud': agent_cloud.state_dict(),
                'critic': critic.state_dict(),
                'hgnn': hgnn.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
                'best_epoch_idx': best_epoch_idx,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')},
            }, best_model_path)
            logging.info(f'New Best Model Saved! Reward: {best_reward:.4f}')

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == Config.EPOCHS:
            torch.save({
                'epoch': epoch,
                'agent_local': agent_local.state_dict(),
                'agent_edge': agent_edge.state_dict(),
                'agent_cloud': agent_cloud.state_dict(),
                'critic': critic.state_dict(),
                'hgnn': hgnn.state_dict(),
                'opt_local': opt_local.state_dict(),
                'opt_edge': opt_edge.state_dict(),
                'opt_cloud': opt_cloud.state_dict(),
                'opt_critic': opt_critic.state_dict(),
                'sched_local': sched_local.state_dict(),
                'sched_edge': sched_edge.state_dict(),
                'sched_cloud': sched_cloud.state_dict(),
                'sched_critic': sched_critic.state_dict(),
                'best_reward': best_reward,
                'best_epoch_idx': best_epoch_idx,
                'entropy_beta': entropy_beta,
                'epoch_rewards': epoch_rewards,
                'epoch_r_latency': epoch_r_latency,
                'epoch_r_ddl': epoch_r_ddl,
                'epoch_r_slack': epoch_r_slack,
                'epoch_avg_wf_time': epoch_avg_wf_time,
                'epoch_ddl_rate': epoch_ddl_rate,
                'epoch_avg_path_qoe': epoch_avg_path_qoe,
                'epoch_steps': epoch_steps,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')},
            }, checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch + 1}')
            logging.info(f'Checkpoint saved at epoch {epoch + 1}')

        print(
            f'Epoch {epoch + 1}/{Config.EPOCHS}, Reward: {avg_reward:.4f}, Avg Latency: {avg_r_latency:.4f}, '
            f'Avg DDL: {avg_r_ddl:.4f}, Avg Slack: {avg_r_slack:.4f}, Steps: {steps}, '
            f'Avg WF Time: {final_metrics.get("avg_wf_time", 0.0):.4f}, LR: {sched_local.get_last_lr()[0]:.6f}'
        )
        logging.info(
            f'Epoch {epoch + 1}, Avg Reward: {avg_reward:.4f}, Avg Latency: {avg_r_latency:.4f}, '
            f'Avg DDL: {avg_r_ddl:.4f}, Avg Slack: {avg_r_slack:.4f}, Steps: {steps}, Avg WF Time: {final_metrics.get("avg_wf_time", 0.0):.4f}'
        )

    save_curve(epoch_rewards)
    record = build_run_record(
        best_epoch_idx,
        epoch_rewards,
        epoch_r_latency,
        epoch_r_ddl,
        epoch_r_slack,
        epoch_avg_wf_time,
        epoch_ddl_rate,
        epoch_avg_path_qoe,
    )
    summary_json_path, summary_csv_path = write_experiment_record(summary_dir, METHOD_NAME, RUN_TAG, record)
    logging.info(f'Experiment summary saved to {summary_json_path}')
    logging.info(f'Experiment summary index updated at {summary_csv_path}')
    print(f'Experiment summary saved to {summary_json_path}')
    print('Training Complete.')


if __name__ == '__main__':
    try:
        train()
    except Exception:
        import traceback
        traceback.print_exc()
        logging.error('Fatal error', exc_info=True)



