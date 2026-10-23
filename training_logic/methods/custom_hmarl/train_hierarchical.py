import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHOD_NAME = "custom_hmarl"
BASE_RUN_TAG = "hierarchical"
RUN_TAG_SUFFIX = os.environ.get("TRAIN_RUN_TAG_SUFFIX", "").strip()
RUN_TAG = f"{BASE_RUN_TAG}{RUN_TAG_SUFFIX}" if RUN_TAG_SUFFIX else BASE_RUN_TAG
DEFAULT_NUM_USERS = None
TRAIN_NUM_USERS = os.environ.get("TRAIN_NUM_USERS")
NUM_USERS = int(TRAIN_NUM_USERS) if TRAIN_NUM_USERS else DEFAULT_NUM_USERS
CHECKPOINT_INTERVAL = 10

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
reward_plot_path = os.path.join(figure_dir, f'reward_total_{RUN_TAG}.png')
component_plot_path = os.path.join(figure_dir, f'reward_components_{RUN_TAG}.png')
metric_plot_path = os.path.join(figure_dir, f'metrics_plot_{RUN_TAG}.png')
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
    LR = 1e-3
    GAMMA = 0.99
    EPOCHS = 1000
    HIDDEN_DIM = 128
    SEED = 42
    ENTROPY_BETA_START = 0.05
    ENTROPY_BETA_MIN = 0.001
    ENTROPY_DECAY = 0.999
    LR_STEP_SIZE = 200
    LR_GAMMA = 0.5


SEED_OVERRIDE = os.environ.get("TRAIN_SEED")
if SEED_OVERRIDE:
    Config.SEED = int(SEED_OVERRIDE)


random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class ManagerAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(ManagerAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


class WorkerAgent(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(WorkerAgent, self).__init__()
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
            'figure_files': [reward_plot_path, component_plot_path, metric_plot_path],
        },
    }


def save_plots(epoch_rewards, epoch_r_latency, epoch_r_ddl, epoch_r_slack,
               epoch_avg_wf_time, epoch_ddl_rate, epoch_avg_path_qoe):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_rewards) + 1), epoch_rewards, marker='o', linestyle='-', color='b')
    plt.title('Total Reward per Epoch (Hierarchical)')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(reward_plot_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_r_latency) + 1), epoch_r_latency, label='Latency Reward')
    plt.plot(range(1, len(epoch_r_ddl) + 1), epoch_r_ddl, label='DDL Penalty')
    plt.plot(range(1, len(epoch_r_slack) + 1), epoch_r_slack, label='Slack Reward')
    plt.title('Sub-Rewards per Epoch (Hierarchical)')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(component_plot_path)
    plt.close()

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(range(1, len(epoch_avg_wf_time) + 1), epoch_avg_wf_time, color='r')
    axs[0].set_title('Avg Workflow Completion Time')
    axs[0].set_ylabel('Time (s)')
    axs[0].grid(True)

    axs[1].plot(range(1, len(epoch_ddl_rate) + 1), epoch_ddl_rate, color='g')
    axs[1].set_title('DDL Satisfaction Rate')
    axs[1].set_ylabel('Rate (0-1)')
    axs[1].grid(True)

    axs[2].plot(range(1, len(epoch_avg_path_qoe) + 1), epoch_avg_path_qoe, color='m')
    axs[2].set_title('Avg Path QoE (Slack)')
    axs[2].set_ylabel('Score')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(metric_plot_path)
    plt.close()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    print(f'Using device: {device}')
    print(f'Results root: {results_root}')

    env = build_env()
    input_dim = 121

    manager = ManagerAgent(input_dim, Config.HIDDEN_DIM).to(device)
    local_worker = WorkerAgent(input_dim, 100, Config.HIDDEN_DIM).to(device)
    edge_worker = WorkerAgent(input_dim, 10, Config.HIDDEN_DIM).to(device)
    cloud_worker = WorkerAgent(input_dim, 2, Config.HIDDEN_DIM).to(device)
    critic = Critic(input_dim, Config.HIDDEN_DIM).to(device)

    manager_opt = optim.Adam(manager.parameters(), lr=Config.LR)
    local_opt = optim.Adam(local_worker.parameters(), lr=Config.LR)
    edge_opt = optim.Adam(edge_worker.parameters(), lr=Config.LR)
    cloud_opt = optim.Adam(cloud_worker.parameters(), lr=Config.LR)
    critic_opt = optim.Adam(critic.parameters(), lr=Config.LR)

    manager_scheduler = optim.lr_scheduler.StepLR(manager_opt, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    local_scheduler = optim.lr_scheduler.StepLR(local_opt, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    edge_scheduler = optim.lr_scheduler.StepLR(edge_opt, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    cloud_scheduler = optim.lr_scheduler.StepLR(cloud_opt, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    critic_scheduler = optim.lr_scheduler.StepLR(critic_opt, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)

    print('Starting Hierarchical Training...')

    start_epoch = 0
    epoch_rewards = []
    epoch_r_latency = []
    epoch_r_ddl = []
    epoch_r_slack = []
    epoch_avg_wf_time = []
    epoch_ddl_rate = []
    epoch_avg_path_qoe = []
    best_reward = -float('inf')
    best_epoch_idx = -1
    entropy_beta = Config.ENTROPY_BETA_START

    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        manager.load_state_dict(checkpoint['manager'])
        local_worker.load_state_dict(checkpoint['local_worker'])
        edge_worker.load_state_dict(checkpoint['edge_worker'])
        cloud_worker.load_state_dict(checkpoint['cloud_worker'])
        critic.load_state_dict(checkpoint['critic'])
        manager_opt.load_state_dict(checkpoint['manager_opt'])
        local_opt.load_state_dict(checkpoint['local_opt'])
        edge_opt.load_state_dict(checkpoint['edge_opt'])
        cloud_opt.load_state_dict(checkpoint['cloud_opt'])
        critic_opt.load_state_dict(checkpoint['critic_opt'])
        manager_scheduler.load_state_dict(checkpoint['manager_scheduler'])
        local_scheduler.load_state_dict(checkpoint['local_scheduler'])
        edge_scheduler.load_state_dict(checkpoint['edge_scheduler'])
        cloud_scheduler.load_state_dict(checkpoint['cloud_scheduler'])
        critic_scheduler.load_state_dict(checkpoint['critic_scheduler'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_reward = checkpoint.get('best_reward', best_reward)
        best_epoch_idx = checkpoint.get('best_epoch_idx', best_epoch_idx)
        entropy_beta = checkpoint.get('entropy_beta', entropy_beta)
        epoch_rewards = checkpoint.get('epoch_rewards', epoch_rewards)
        epoch_r_latency = checkpoint.get('epoch_r_latency', epoch_r_latency)
        epoch_r_ddl = checkpoint.get('epoch_r_ddl', epoch_r_ddl)
        epoch_r_slack = checkpoint.get('epoch_r_slack', epoch_r_slack)
        epoch_avg_wf_time = checkpoint.get('epoch_avg_wf_time', epoch_avg_wf_time)
        epoch_ddl_rate = checkpoint.get('epoch_ddl_rate', epoch_ddl_rate)
        epoch_avg_path_qoe = checkpoint.get('epoch_avg_path_qoe', epoch_avg_path_qoe)
        print(f'Resuming training from epoch {start_epoch}')

    for epoch in range(start_epoch, Config.EPOCHS):
        state = env.reset()
        masks = env._get_action_mask()

        done = False
        total_reward = 0.0
        total_wait = 0.0
        total_run = 0.0
        total_comm = 0.0
        total_r_latency = 0.0
        total_r_ddl = 0.0
        total_r_slack = 0.0
        final_metrics = {}
        steps = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            manager_logits = manager(state_tensor)
            manager_mask = torch.BoolTensor(masks['manager']).unsqueeze(0).to(device)
            manager_logits = manager_logits.masked_fill(~manager_mask, -1e9)
            manager_probs = F.softmax(manager_logits, dim=-1)
            manager_dist = torch.distributions.Categorical(manager_probs)
            tier_action = manager_dist.sample()

            tier = tier_action.item()
            if tier == 0:
                worker = local_worker
                worker_mask = torch.BoolTensor(masks['local']).unsqueeze(0).to(device)
            elif tier == 1:
                worker = edge_worker
                worker_mask = torch.BoolTensor(masks['edge']).unsqueeze(0).to(device)
            else:
                worker = cloud_worker
                worker_mask = torch.BoolTensor(masks['cloud']).unsqueeze(0).to(device)

            worker_logits = worker(state_tensor)
            worker_logits = worker_logits.masked_fill(~worker_mask, -1e9)
            worker_probs = F.softmax(worker_logits, dim=-1)
            worker_dist = torch.distributions.Categorical(worker_probs)
            node_action = worker_dist.sample()

            value = critic(state_tensor)
            next_state, reward, done, next_masks, info = env.step(tier, node_action.item())

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                next_value = critic(next_state_tensor)

            target = reward + Config.GAMMA * next_value * (1 - int(done))
            advantage = target - value

            manager_entropy = manager_dist.entropy().mean()
            manager_loss = -manager_dist.log_prob(tier_action) * advantage.detach() - entropy_beta * manager_entropy

            worker_entropy = worker_dist.entropy().mean()
            worker_loss = -worker_dist.log_prob(node_action) * advantage.detach() - entropy_beta * worker_entropy

            critic_loss = F.mse_loss(value, target.detach())

            manager_opt.zero_grad()
            manager_loss.backward()
            torch.nn.utils.clip_grad_norm_(manager.parameters(), 1.0)
            manager_opt.step()

            if tier == 0:
                opt = local_opt
            elif tier == 1:
                opt = edge_opt
            else:
                opt = cloud_opt

            opt.zero_grad()
            worker_loss.backward()
            torch.nn.utils.clip_grad_norm_(worker.parameters(), 1.0)
            opt.step()

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_opt.step()

            state = next_state
            masks = next_masks
            total_reward += reward
            total_wait += info.get('wait', 0.0)
            total_run += info.get('dur', 0.0)
            total_comm += info.get('comm', 0.0)
            total_r_latency += info.get('r_latency', 0.0)
            total_r_ddl += info.get('r_ddl', 0.0)
            total_r_slack += info.get('r_slack', 0.0)
            final_metrics = info
            steps += 1

        avg_reward = total_reward / steps if steps > 0 else 0.0
        avg_wait = total_wait / steps if steps > 0 else 0.0
        avg_run = total_run / steps if steps > 0 else 0.0
        avg_comm = total_comm / steps if steps > 0 else 0.0
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

        manager_scheduler.step()
        local_scheduler.step()
        edge_scheduler.step()
        cloud_scheduler.step()
        critic_scheduler.step()
        entropy_beta = max(Config.ENTROPY_BETA_MIN, entropy_beta * Config.ENTROPY_DECAY)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_epoch_idx = epoch
            torch.save({
                'manager': manager.state_dict(),
                'local_worker': local_worker.state_dict(),
                'edge_worker': edge_worker.state_dict(),
                'cloud_worker': cloud_worker.state_dict(),
                'critic': critic.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
                'best_epoch_idx': best_epoch_idx,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')},
            }, best_model_path)
            logging.info(f'New Best Model Saved! Reward: {best_reward:.2f}')

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == Config.EPOCHS:
            torch.save({
                'manager': manager.state_dict(),
                'local_worker': local_worker.state_dict(),
                'edge_worker': edge_worker.state_dict(),
                'cloud_worker': cloud_worker.state_dict(),
                'critic': critic.state_dict(),
                'manager_opt': manager_opt.state_dict(),
                'local_opt': local_opt.state_dict(),
                'edge_opt': edge_opt.state_dict(),
                'cloud_opt': cloud_opt.state_dict(),
                'critic_opt': critic_opt.state_dict(),
                'manager_scheduler': manager_scheduler.state_dict(),
                'local_scheduler': local_scheduler.state_dict(),
                'edge_scheduler': edge_scheduler.state_dict(),
                'cloud_scheduler': cloud_scheduler.state_dict(),
                'critic_scheduler': critic_scheduler.state_dict(),
                'epoch': epoch,
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
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')},
            }, checkpoint_path)
            logging.info(f'Checkpoint saved at epoch {epoch + 1}')

        logging.info(
            f'Epoch {epoch + 1}/{Config.EPOCHS}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, '
            f'Avg Wait: {avg_wait:.2f}, Avg Run: {avg_run:.2f}, Avg Comm: {avg_comm:.2f}, '
            f'Avg Latency: {avg_r_latency:.2f}, Avg DDL: {avg_r_ddl:.2f}, Avg Slack: {avg_r_slack:.2f}, '
            f'LR: {manager_scheduler.get_last_lr()[0]:.6f}, Entropy: {entropy_beta:.4f}'
        )
        print(
            f'Epoch {epoch + 1}/{Config.EPOCHS}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, '
            f'Avg Latency: {avg_r_latency:.2f}, Avg DDL: {avg_r_ddl:.2f}, Avg Slack: {avg_r_slack:.2f}, '
            f'LR: {manager_scheduler.get_last_lr()[0]:.6f}, Entropy: {entropy_beta:.4f}'
        )

    save_plots(
        epoch_rewards,
        epoch_r_latency,
        epoch_r_ddl,
        epoch_r_slack,
        epoch_avg_wf_time,
        epoch_ddl_rate,
        epoch_avg_path_qoe,
    )
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


if __name__ == '__main__':
    try:
        train()
    except Exception:
        import traceback
        traceback.print_exc()
        logging.error('Fatal error', exc_info=True)

