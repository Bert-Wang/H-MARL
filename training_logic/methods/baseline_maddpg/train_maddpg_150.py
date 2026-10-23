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

METHOD_NAME = "maddpg"
RUN_TAG = "maddpg_150"
NUM_USERS = 150
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
figure_path = os.path.join(figure_dir, f'training_curves_{RUN_TAG}.png')
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
    BATCH_SIZE = 64
    BUFFER_SIZE = 50000
    TAU = 0.01
    GUMBEL_TEMP = 1.0
    START_STEPS = 1000


random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done, tier):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done, tier)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, tier = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done, tier

    def __len__(self):
        return len(self.buffer)


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
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


class DDPGAgent:
    def __init__(self, input_dim, action_dim, device, agent_id):
        self.agent_id = agent_id
        self.actor = Actor(input_dim, action_dim, Config.HIDDEN_DIM).to(device)
        self.actor_target = Actor(input_dim, action_dim, Config.HIDDEN_DIM).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=Config.LR)

        self.critic = Critic(input_dim, action_dim, Config.HIDDEN_DIM).to(device)
        self.critic_target = Critic(input_dim, action_dim, Config.HIDDEN_DIM).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=Config.LR)

        self.action_dim = action_dim
        self.device = device
        self.input_dim = input_dim

    def select_action(self, state, evaluate=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor(state_t)

        if evaluate:
            probs = F.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs, dim=-1).item()
            action_onehot = F.one_hot(torch.tensor([action_idx]), num_classes=self.action_dim).float().to(self.device)
        else:
            action_onehot = F.gumbel_softmax(logits, tau=Config.GUMBEL_TEMP, hard=True)
            action_idx = torch.argmax(action_onehot, dim=-1).item()

        q_value = self.critic(state_t, action_onehot)
        return action_idx, q_value.item(), action_onehot

    def update(self, states, actions_idx, rewards, next_states, dones):
        if len(states) == 0:
            return

        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions_idx).to(self.device)
        reward_batch = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        action_onehot = F.one_hot(action_batch, num_classes=self.action_dim).float()

        with torch.no_grad():
            next_logits = self.actor_target(next_state_batch)
            next_action_onehot = F.gumbel_softmax(next_logits, tau=Config.GUMBEL_TEMP, hard=True)
            target_q = self.critic_target(next_state_batch, next_action_onehot)
            target_q = reward_batch + (1 - done_batch) * Config.GAMMA * target_q

        current_q = self.critic(state_batch, action_onehot)
        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        current_logits = self.actor(state_batch)
        current_action_soft = F.gumbel_softmax(current_logits, tau=Config.GUMBEL_TEMP, hard=False)
        actor_loss = -self.critic(state_batch, current_action_soft).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update(self.critic, self.critic_target, Config.TAU)
        self._soft_update(self.actor, self.actor_target, Config.TAU)

    def _soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def checkpoint_state(self):
        return {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'action_dim': self.action_dim,
            'agent_id': self.agent_id,
        }

    def load_checkpoint_state(self, state):
        self.actor.load_state_dict(state['actor'])
        self.actor_target.load_state_dict(state['actor_target'])
        self.critic.load_state_dict(state['critic'])
        self.critic_target.load_state_dict(state['critic_target'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])


def build_env():
    env_kwargs = {'time_interval': 10.0}
    if NUM_USERS is not None:
        env_kwargs['num_users'] = NUM_USERS
    return HierarchicalDistributedSystemEnv(**env_kwargs)


def get_effective_num_users():
    return 100 if NUM_USERS is None else NUM_USERS


def build_run_record(best_epoch_idx, metrics):
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
            metrics['reward'],
            metrics['r_latency'],
            metrics['r_ddl'],
            metrics['r_slack'],
            metrics['avg_wf_time'],
            metrics['ddl_rate'],
            metrics['avg_path_qoe'],
        ),
        'artifacts': {
            'log_file': log_file,
            'checkpoint_file': checkpoint_path,
            'best_model_file': best_model_path,
            'epoch_metrics_file': epoch_metrics_file,
            'figure_files': [figure_path],
        },
    }


def save_training_curves(metrics):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(metrics['reward'])
    plt.title('Total Reward')
    plt.subplot(2, 2, 2)
    plt.plot(metrics['r_latency'])
    plt.title('Avg Latency Reward')
    plt.subplot(2, 2, 3)
    plt.plot(metrics['r_ddl'])
    plt.title('Avg DDL Reward')
    plt.subplot(2, 2, 4)
    plt.plot(metrics['r_slack'])
    plt.title('Avg Slack Reward')
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    print(f'Using device: {device}')
    print(f'Results root: {results_root}')

    env = build_env()
    input_dim = 121

    agent_local = DDPGAgent(input_dim, 100, device, agent_id=0)
    agent_edge = DDPGAgent(input_dim, 10, device, agent_id=1)
    agent_cloud = DDPGAgent(input_dim, 2, device, agent_id=2)
    agents = [agent_local, agent_edge, agent_cloud]
    replay_buffer = ReplayBuffer(Config.BUFFER_SIZE)

    metrics = {
        'reward': [],
        'r_latency': [],
        'r_ddl': [],
        'r_slack': [],
        'avg_wf_time': [],
        'ddl_rate': [],
        'avg_path_qoe': [],
    }
    start_epoch = 0
    best_reward = -float('inf')
    best_epoch_idx = -1

    if os.path.exists(checkpoint_path):
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for agent, state in zip(agents, checkpoint['agents']):
            agent.load_checkpoint_state(state)
        replay_buffer.buffer = checkpoint.get('replay_buffer', [])
        replay_buffer.position = checkpoint.get('replay_position', 0)
        metrics = checkpoint.get('metrics', metrics)
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_reward = checkpoint.get('best_reward', best_reward)
        best_epoch_idx = checkpoint.get('best_epoch_idx', best_epoch_idx)
        print(f'Resuming training from epoch {start_epoch}')

    print('Starting MADDPG (Single Layer) Training...')

    for epoch in range(start_epoch, Config.EPOCHS):
        state = env.reset()
        done = False
        total_reward = 0.0
        total_r_latency = 0.0
        total_r_ddl = 0.0
        total_r_slack = 0.0
        final_metrics = {}
        steps = 0

        while not done:
            agent_outputs = []
            for agent in agents:
                if len(replay_buffer) < Config.START_STEPS:
                    action_idx = random.randint(0, agent.action_dim - 1)
                    q_val = 0.0
                else:
                    action_idx, q_val, _ = agent.select_action(state, evaluate=False)
                agent_outputs.append((action_idx, q_val))

            epsilon = max(0.05, 1.0 - epoch / 200.0)
            if len(replay_buffer) < Config.START_STEPS:
                tier_idx = random.randint(0, 2)
            elif random.random() < epsilon:
                tier_idx = random.randint(0, 2)
            else:
                q_values = [out[1] for out in agent_outputs]
                tier_idx = int(np.argmax(q_values))

            selected_action_idx = agent_outputs[tier_idx][0]
            next_state, reward, done, _mask, info = env.step(tier_idx, selected_action_idx)
            replay_buffer.push(state, selected_action_idx, reward, next_state, done, tier_idx)

            state = next_state
            total_reward += reward
            total_r_latency += info.get('r_latency', 0.0)
            total_r_ddl += info.get('r_ddl', 0.0)
            total_r_slack += info.get('r_slack', 0.0)
            final_metrics = info
            steps += 1

            if len(replay_buffer) > Config.BATCH_SIZE:
                b_state, b_action, b_reward, b_next_state, b_done, b_tier = replay_buffer.sample(Config.BATCH_SIZE)
                for tier_id in range(3):
                    indices = np.where(b_tier == tier_id)[0]
                    if len(indices) > 0:
                        agents[tier_id].update(
                            b_state[indices],
                            b_action[indices],
                            b_reward[indices],
                            b_next_state[indices],
                            b_done[indices],
                        )

        avg_reward = total_reward / max(1, steps)
        avg_lat = total_r_latency / max(1, steps)
        avg_ddl = total_r_ddl / max(1, steps)
        avg_slack = total_r_slack / max(1, steps)

        metrics['reward'].append(total_reward)
        metrics['r_latency'].append(avg_lat)
        metrics['r_ddl'].append(avg_ddl)
        metrics['r_slack'].append(avg_slack)
        metrics['avg_wf_time'].append(final_metrics.get('avg_wf_time', 0.0))
        metrics['ddl_rate'].append(final_metrics.get('ddl_rate', 0.0))
        metrics['avg_path_qoe'].append(final_metrics.get('avg_path_qoe', 0.0))
        write_epoch_metrics_csv(
            summary_dir,
            METHOD_NAME,
            RUN_TAG,
            build_epoch_records(
                metrics['reward'],
                metrics['r_latency'],
                metrics['r_ddl'],
                metrics['r_slack'],
                metrics['avg_wf_time'],
                metrics['ddl_rate'],
                metrics['avg_path_qoe'],
            ),
        )

        logging.info(
            f'Epoch {epoch}: Reward={total_reward:.2f}, AvgReward={avg_reward:.2f}, '
            f'Lat={avg_lat:.2f}, DDL={avg_ddl:.2f}, Slack={avg_slack:.2f}, Steps={steps}'
        )
        print(f'Epoch {epoch}: Reward={total_reward:.2f} (Avg:{avg_reward:.2f}, Lat:{avg_lat:.2f}, DDL:{avg_ddl:.2f}, Slack:{avg_slack:.2f})')

        if total_reward > best_reward:
            best_reward = total_reward
            best_epoch_idx = epoch
            torch.save({
                'agents': [agent.checkpoint_state() for agent in agents],
                'epoch': epoch,
                'best_reward': best_reward,
                'best_epoch_idx': best_epoch_idx,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')},
            }, best_model_path)
            logging.info(f'New Best Model Saved! Reward: {best_reward:.2f}')

        if (epoch + 1) % CHECKPOINT_INTERVAL == 0 or (epoch + 1) == Config.EPOCHS:
            torch.save({
                'agents': [agent.checkpoint_state() for agent in agents],
                'replay_buffer': replay_buffer.buffer,
                'replay_position': replay_buffer.position,
                'metrics': metrics,
                'epoch': epoch,
                'best_reward': best_reward,
                'best_epoch_idx': best_epoch_idx,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')},
            }, checkpoint_path)
            logging.info(f'Checkpoint saved at epoch {epoch + 1}')

        if (epoch + 1) % 50 == 0 or (epoch + 1) == Config.EPOCHS:
            save_training_curves(metrics)

    save_training_curves(metrics)
    record = build_run_record(best_epoch_idx, metrics)
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



