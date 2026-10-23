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

METHOD_NAME = "maa2c"
RUN_TAG = "maa2c_50"
NUM_USERS = 50
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


random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


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
    plt.title('Total Reward per Epoch (MAA2C)')
    plt.xlabel('Epoch')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(reward_plot_path)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_r_latency) + 1), epoch_r_latency, label='Latency Reward')
    plt.plot(range(1, len(epoch_r_ddl) + 1), epoch_r_ddl, label='DDL Penalty')
    plt.plot(range(1, len(epoch_r_slack) + 1), epoch_r_slack, label='Slack Reward')
    plt.title('Sub-Rewards per Epoch (MAA2C)')
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

    agent_local = Actor(input_dim, 100, Config.HIDDEN_DIM).to(device)
    agent_edge = Actor(input_dim, 10, Config.HIDDEN_DIM).to(device)
    agent_cloud = Actor(input_dim, 2, Config.HIDDEN_DIM).to(device)
    critic = Critic(input_dim, Config.HIDDEN_DIM).to(device)

    opt_local = optim.Adam(agent_local.parameters(), lr=Config.LR)
    opt_edge = optim.Adam(agent_edge.parameters(), lr=Config.LR)
    opt_cloud = optim.Adam(agent_cloud.parameters(), lr=Config.LR)
    opt_critic = optim.Adam(critic.parameters(), lr=Config.LR)

    sched_local = optim.lr_scheduler.StepLR(opt_local, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    sched_edge = optim.lr_scheduler.StepLR(opt_edge, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    sched_cloud = optim.lr_scheduler.StepLR(opt_cloud, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)
    sched_critic = optim.lr_scheduler.StepLR(opt_critic, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA)

    print('Starting MAA2C (3 Agents - No Manager) Training...')

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
        agent_local.load_state_dict(checkpoint['local'])
        agent_edge.load_state_dict(checkpoint['edge'])
        agent_cloud.load_state_dict(checkpoint['cloud'])
        critic.load_state_dict(checkpoint['critic'])
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

            logits_local = agent_local(state_tensor)
            mask_local = torch.BoolTensor(masks['local']).unsqueeze(0).to(device)
            logits_local = logits_local.masked_fill(~mask_local, -1e9)
            dist_local = torch.distributions.Categorical(logits=logits_local)
            action_local = dist_local.sample()
            prob_local = dist_local.probs.max()

            logits_edge = agent_edge(state_tensor)
            mask_edge = torch.BoolTensor(masks['edge']).unsqueeze(0).to(device)
            logits_edge = logits_edge.masked_fill(~mask_edge, -1e9)
            dist_edge = torch.distributions.Categorical(logits=logits_edge)
            action_edge = dist_edge.sample()
            prob_edge = dist_edge.probs.max()

            logits_cloud = agent_cloud(state_tensor)
            mask_cloud = torch.BoolTensor(masks['cloud']).unsqueeze(0).to(device)
            logits_cloud = logits_cloud.masked_fill(~mask_cloud, -1e9)
            dist_cloud = torch.distributions.Categorical(logits=logits_cloud)
            action_cloud = dist_cloud.sample()
            prob_cloud = dist_cloud.probs.max()

            epsilon = max(0.05, 1.0 - epoch / 200.0)
            if random.random() < epsilon:
                tier_idx = random.randint(0, 2)
            else:
                probs = [prob_local.item(), prob_edge.item(), prob_cloud.item()]
                tier_idx = int(np.argmax(probs))

            if tier_idx == 0:
                node_idx = action_local.item()
            elif tier_idx == 1:
                node_idx = action_edge.item()
            else:
                node_idx = action_cloud.item()

            value = critic(state_tensor)
            next_state, reward, done, next_masks, info = env.step(tier_idx, node_idx)

            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            with torch.no_grad():
                next_value = critic(next_state_tensor)

            target = reward + Config.GAMMA * next_value * (1 - int(done))
            advantage = target - value

            loss_local = torch.tensor(0.0, device=device)
            loss_edge = torch.tensor(0.0, device=device)
            loss_cloud = torch.tensor(0.0, device=device)

            if tier_idx == 0:
                entropy_local = dist_local.entropy().mean()
                loss_local = -dist_local.log_prob(action_local) * advantage.detach() - entropy_beta * entropy_local
            elif tier_idx == 1:
                entropy_edge = dist_edge.entropy().mean()
                loss_edge = -dist_edge.log_prob(action_edge) * advantage.detach() - entropy_beta * entropy_edge
            else:
                entropy_cloud = dist_cloud.entropy().mean()
                loss_cloud = -dist_cloud.log_prob(action_cloud) * advantage.detach() - entropy_beta * entropy_cloud

            loss_critic = F.mse_loss(value, target.detach())

            if tier_idx == 0:
                opt_local.zero_grad()
                loss_local.backward()
                torch.nn.utils.clip_grad_norm_(agent_local.parameters(), 1.0)
                opt_local.step()
            elif tier_idx == 1:
                opt_edge.zero_grad()
                loss_edge.backward()
                torch.nn.utils.clip_grad_norm_(agent_edge.parameters(), 1.0)
                opt_edge.step()
            else:
                opt_cloud.zero_grad()
                loss_cloud.backward()
                torch.nn.utils.clip_grad_norm_(agent_cloud.parameters(), 1.0)
                opt_cloud.step()

            opt_critic.zero_grad()
            loss_critic.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_critic.step()

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

        sched_local.step()
        sched_edge.step()
        sched_cloud.step()
        sched_critic.step()
        entropy_beta = max(Config.ENTROPY_BETA_MIN, entropy_beta * Config.ENTROPY_DECAY)

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_epoch_idx = epoch
            torch.save({
                'local': agent_local.state_dict(),
                'edge': agent_edge.state_dict(),
                'cloud': agent_cloud.state_dict(),
                'critic': critic.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
                'best_epoch_idx': best_epoch_idx,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')},
            }, best_model_path)
            logging.info(f'New Best Model Saved! Reward: {best_reward:.2f}')

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == Config.EPOCHS:
            torch.save({
                'local': agent_local.state_dict(),
                'edge': agent_edge.state_dict(),
                'cloud': agent_cloud.state_dict(),
                'critic': critic.state_dict(),
                'opt_local': opt_local.state_dict(),
                'opt_edge': opt_edge.state_dict(),
                'opt_cloud': opt_cloud.state_dict(),
                'opt_critic': opt_critic.state_dict(),
                'sched_local': sched_local.state_dict(),
                'sched_edge': sched_edge.state_dict(),
                'sched_cloud': sched_cloud.state_dict(),
                'sched_critic': sched_critic.state_dict(),
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
            f'LR: {sched_local.get_last_lr()[0]:.6f}, Entropy: {entropy_beta:.4f}'
        )
        print(
            f'Epoch {epoch + 1}/{Config.EPOCHS}, Total Reward: {total_reward:.2f}, Avg Reward: {avg_reward:.2f}, '
            f'Avg Latency: {avg_r_latency:.2f}, Avg DDL: {avg_r_ddl:.2f}, Avg Slack: {avg_r_slack:.2f}, '
            f'LR: {sched_local.get_last_lr()[0]:.6f}, Entropy: {entropy_beta:.4f}'
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



