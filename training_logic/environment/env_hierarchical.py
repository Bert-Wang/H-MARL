import os
import sys
import json
import random
import math
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import logging

# Add parent directory to path to allow importing from dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Configure logging
log_dir = os.path.join(current_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'env_hierarchical.log'),
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)


def resolve_dataset_csv_path():
    candidate_paths = [
        os.path.join(parent_dir, '01_data', 'raw', '100_users', 'global_100users_with_privacy.csv'),
        os.path.join(os.path.dirname(parent_dir), 'dataset', '100_users', 'global_100users_with_privacy.csv'),
    ]

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Dataset CSV not found. Checked: " + "; ".join(candidate_paths)
    )


class WorkflowTask:

    def __init__(self, id, name, parents, children, input_files, output_files, wf_runtime_sec, core_count, req_mem=0.0, req_gpu_mem=0.0, data_input_size=0.0, data_output_size=0.0, privacy_level=0, home_pi=None, allowed_nodes=None, workflow_id=None):
        self.id = id
        self.parents = parents
        self.children = children
        self.input_files = input_files
        self.output_files = output_files
        self.wf_runtime_sec = wf_runtime_sec
        self.core_count = core_count
        self.req_mem = req_mem
        self.req_gpu_mem = req_gpu_mem
        self.data_input_size = data_input_size
        self.data_output_size = data_output_size
        self.privacy_level = privacy_level
        self.home_pi = home_pi
        self.allowed_nodes = allowed_nodes if allowed_nodes is not None else []
        self.max_latency = self.wf_runtime_sec * 5.0 # QoS Constraint
        
        self.assigned_node = None
        self.unique_id = id
        self.is_local = False
        
        # New attributes for Slack modeling
        self.workflow_id = workflow_id
        self.static_remaining_time = 0.0  # CP_tail: Critical Path length from this task to sink
        self.deadline = 0.0               # Path/Workflow deadline


class HierarchicalDistributedSystemEnv:
    def __init__(
        self,
        time_interval=1.0,
        num_users=100,
        disable_slack_feature=False,
        disable_action_mask=False,
        reward_weight_ddl=1.5,
        reward_weight_slack=4.0,
    ):
        self.num_users = num_users
        self.disable_slack_feature = disable_slack_feature
        self.disable_action_mask = disable_action_mask
        self.reward_weight_ddl = reward_weight_ddl
        self.reward_weight_slack = reward_weight_slack
        # Topology Definition
        # 100 Pis (Local), 10 Jetsons (Edge), 2 Cloud (Cloud)
        self.pis = [f"pi_{i}" for i in range(100)]
        self.jetsons = [f"jetson_{i}" for i in range(10)]
        self.cloud = [f"cloud_{i}" for i in range(2)]
        
        self.compute_nodes = self.pis + self.jetsons + self.cloud
        self.all_nodes = self.compute_nodes
        
        # Node Resources
        self.node_specs = {}
        for n in self.pis: 
            self.node_specs[n] = {'cpu': 4, 'mem': 8.0, 'gpu_mem': 1.0, 'type': 'pi', 'queue': deque(), 'free_time': 0.0, 'speed': 1.0}
        for n in self.jetsons: 
            self.node_specs[n] = {'cpu': 8, 'mem': 16.0, 'gpu_mem': 8.0, 'type': 'edge', 'queue': deque(), 'free_time': 0.0, 'speed': 2.0}
        for n in self.cloud: 
            self.node_specs[n] = {'cpu': 64, 'mem': 128.0, 'gpu_mem': 24.0, 'type': 'cloud', 'queue': deque(), 'free_time': 0.0, 'speed': 4.0}
            
        # Bandwidth Matrix
        # Total nodes = 100 + 10 + 2 = 112
        self.bw_matrix = np.zeros((112, 112))
        self._init_bw_matrix()
        
        self.current_time = 0.0
        self.time_interval = time_interval
        
        self.completed_tasks = set()
        self.running_tasks = [] 
        self.task_infos = {} 
        self.ready_queue = []
        self.batch_queue = [] 
        self.current_task = None
        self.task_map = {}
        self.all_tasks = []

    def _init_bw_matrix(self):
        def get_idx(name):
            if 'pi_' in name: return int(name.split('_')[1])
            if 'jetson_' in name: return int(name.split('_')[1]) + 100
            if 'cloud_' in name: return int(name.split('_')[1]) + 110
            return -1
            
        for s in self.all_nodes:
            for d in self.all_nodes:
                s_idx = get_idx(s)
                d_idx = get_idx(d)
                
                if s == d: bw = float('inf')
                # Intra-tier BW
                elif 'pi' in s and 'pi' in d: bw = 12.5 * 1024 * 1024
                elif 'jetson' in s and 'jetson' in d: bw = 125.0 * 1024 * 1024
                elif 'cloud' in s and 'cloud' in d: bw = 1250.0 * 1024 * 1024
                
                # Inter-tier BW
                elif ('pi' in s and 'jetson' in d) or ('jetson' in s and 'pi' in d): bw = 12.5 * 1024 * 1024
                elif ('pi' in s and 'cloud' in d) or ('cloud' in s and 'pi' in d): bw = 2.5 * 1024 * 1024
                elif ('jetson' in s and 'cloud' in d) or ('cloud' in s and 'jetson' in d): bw = 6.25 * 1024 * 1024
                else: bw = 1.0
                
                self.bw_matrix[s_idx][d_idx] = bw

    def _get_node_name(self, index):
        if 0 <= index <= 99:
            return f"pi_{index}"
        elif 100 <= index <= 109:
            return f"jetson_{index-100}"
        elif 110 <= index <= 111:
            return f"cloud_{index-110}"
        return None

    def _get_node_idx(self, name):
        if 'pi_' in name: return int(name.split('_')[1])
        if 'jetson_' in name: return int(name.split('_')[1]) + 100
        if 'cloud_' in name: return int(name.split('_')[1]) + 110
        return -1


    def reset(self):
        self.current_time = 0.0
        self.completed_tasks = set()
        self.running_tasks = []
        self.task_infos = {}
        self.task_map = {} 
        self.ready_queue = []
        self.batch_queue = []
        self.current_task = None
        self.all_tasks = []
        self.local_pending_tasks = [] # Tasks that are local but waiting for dependencies
        
        # Workflow Stats Tracking
        self.workflow_progress = defaultdict(lambda: {'tasks_done': 0, 'start_time': float('inf'), 'end_time': 0.0})
        self.completed_workflows = []
        self.total_slack_score = 0.0
        self.total_assignments = 0
        
        for n in self.all_nodes:
            self.node_specs[n]['queue'].clear()
            self.node_specs[n]['remaining_cpu'] = self.node_specs[n]['cpu']
            self.node_specs[n]['remaining_mem'] = self.node_specs[n]['mem']
            self.node_specs[n]['remaining_gpu_mem'] = self.node_specs[n]['gpu_mem']
            self.node_specs[n]['free_time'] = 0.0
            
        self._load_tasks()
        
        self._update_ready_queue()
        self._next_batch_task()
        
        return self._get_state()

    def _load_tasks(self):
        self.dag_adj = defaultdict(list)
        self.dag_in_degree = defaultdict(int)

        csv_path = resolve_dataset_csv_path()
        logging.warning(f"Using dataset CSV: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} tasks from CSV.")
            
            # Apply load factor modifications
            if getattr(self, 'num_users', 100) != 100:
                new_rows = []
                if self.num_users < 100:
                    allowed_users = [f"user_{i}" for i in range(self.num_users)]
                    df = df[df['workflow_label'].isin(allowed_users)].copy()
                elif self.num_users > 100:
                    num_to_duplicate = self.num_users - 100
                    for i in range(num_to_duplicate):
                        orig_user = f"user_{i}"
                        new_user = f"user_{i+100}"
                        user_df = df[df['workflow_label'] == orig_user].copy()
                        
                        user_df['workflow_label'] = new_user
                        user_df['global_id'] = user_df['global_id'].str.replace(f"{orig_user}:", f"{new_user}:")
                        if 'global_parents' in user_df.columns:
                            user_df['global_parents'] = user_df['global_parents'].str.replace(f"{orig_user}:", f"{new_user}:")
                        if 'global_children' in user_df.columns:
                            user_df['global_children'] = user_df['global_children'].str.replace(f"{orig_user}:", f"{new_user}:")
                        
                        new_rows.append(user_df)
                        
                    if new_rows:
                        df = pd.concat([df] + new_rows, ignore_index=True)
                logging.info(f"Adjusted for {self.num_users} users. Now processing {len(df)} tasks.")
            
            for index, row in df.iterrows():
                tid = row['global_id']
                name = row['task_name']
                
                parents_str = str(row['global_parents']) if not pd.isna(row['global_parents']) else ""
                parents = [p.strip() for p in parents_str.split(';')] if parents_str else []
                
                children_str = str(row['global_children']) if not pd.isna(row['global_children']) else ""
                children = [c.strip() for c in children_str.split(';')] if children_str else []
                
                # Use trace_comp_time (ms) -> convert to seconds
                runtime_ms = float(row['trace_comp_time']) if 'trace_comp_time' in row and not pd.isna(row['trace_comp_time']) else 1000.0
                runtime = max(0.01, runtime_ms / 1000.0)
                
                # Use trace_cpu_usage -> cores
                cores = float(row['trace_cpu_usage']) if 'trace_cpu_usage' in row and not pd.isna(row['trace_cpu_usage']) else 1.0
                
                # Use trace_max_mem (likely GB based on values like 3.14, 14.58)
                req_mem = float(row['trace_max_mem']) if 'trace_max_mem' in row and not pd.isna(row['trace_max_mem']) else 0.1
                
                # Use trace_max_gpu_wrk_mem (likely GB)
                req_gpu_mem = float(row['trace_max_gpu_wrk_mem']) if 'trace_max_gpu_wrk_mem' in row and not pd.isna(row['trace_max_gpu_wrk_mem']) else 0.0
                
                data_in = float(row['trace_read']) if 'trace_read' in row and not pd.isna(row['trace_read']) else 0.0
                data_out = float(row['trace_write']) if 'trace_write' in row and not pd.isna(row['trace_write']) else 0.0
                
                privacy = int(row['privacy_level']) if not pd.isna(row['privacy_level']) else 0
                home_pi_idx = int(row['home_pi']) if not pd.isna(row['home_pi']) else 0
                workflow_label = row['workflow_label'] if 'workflow_label' in row and not pd.isna(row['workflow_label']) else "unknown"
                
                # Parse allowed_nodes and map to new topology
                allowed_indices_str = str(row['allowed_nodes']) if not pd.isna(row['allowed_nodes']) else ""
                allowed_indices_str = allowed_indices_str.replace('"', '').replace("'", "")
                original_indices = [int(x) for x in allowed_indices_str.split(',')] if allowed_indices_str else []
                
                allowed_node_names = []
                for idx in original_indices:
                    node_name = self._get_node_name(idx)
                    if node_name and node_name not in allowed_node_names:
                        allowed_node_names.append(node_name)
                
                # Ensure home_pi is allowed if valid
                home_pi_name = self._get_node_name(home_pi_idx)
                if home_pi_name and home_pi_name not in allowed_node_names:
                     allowed_node_names.append(home_pi_name)
                
                # Linear Mapping for Resources (Normalize to Range [0.1, 4.0], [0, 8.0], [0, 1.0])
                # CPU: [0, 76.2] -> [0.1, 4.0]
                # MEM: [0, 15] -> [0, 8.0]
                # GPU: [0, 7.5] -> [0, 1.0]
                
                raw_cpu = float(cores)
                raw_mem = float(req_mem)
                raw_gpu = float(req_gpu_mem)
                
                # Linear Scaling
                cores = 0.1 + (raw_cpu / 76.2) * (4.0 - 0.1)
                req_mem = (raw_mem / 15.0) * 8.0
                req_gpu_mem = (raw_gpu / 7.5) * 1.0
                
                cores = max(0.1, min(cores, 4.0))
                req_mem = max(0.0, min(req_mem, 8.0))
                req_gpu_mem = max(0.0, min(req_gpu_mem, 1.0))
                
                t = WorkflowTask(tid, name, parents, children, [], [], runtime, cores, req_mem, req_gpu_mem, data_in, data_out, workflow_id=workflow_label)
                t.unique_id = tid
                t.privacy_level = privacy
                t.home_pi = home_pi_name
                t.allowed_nodes = allowed_node_names
                
                self.task_map[t.unique_id] = t
                
                # Local Task Logic:
                # If no parents (Source Task), force execution on home_pi (no agent scheduling)
                # Else (Intermediate/Sink), allow scheduling on any node (clear privacy restrictions)
                if not parents:
                    t.is_local = True
                    t.assigned_node = t.home_pi
                    # Assign immediately as source tasks have no dependencies
                    if t.assigned_node:
                        self._assign_task(t, t.assigned_node)
                    else:
                        logging.warning(f"Local Source Task {t.unique_id} has no assigned node")
                else:
                    t.is_local = False
                    # Allow scheduling to any node (ignore CSV restrictions for non-source tasks)
                    t.allowed_nodes = None 
                    self.all_tasks.append(t)
                
                # DAG Construction (for all tasks)
                self.dag_in_degree[t.unique_id] = len(parents)
                for p in parents:
                    self.dag_adj[p].append(t.unique_id)

        except Exception as e:
            logging.error(f"Error parsing CSV: {e}")
            raise RuntimeError(f"Failed to parse dataset CSV: {csv_path}") from e

        # Initialize Slack parameters
        self._init_slack_parameters()

    def _init_slack_parameters(self):
        # 1. Compute Static Remaining Time (CP_tail) for all tasks
        # Use memoization
        memo_cp = {}

        def get_cp_tail(tid):
            if tid in memo_cp:
                return memo_cp[tid]
            
            if tid not in self.task_map:
                return 0.0
            
            task = self.task_map[tid]
            
            children_ids = self.dag_adj.get(tid, [])
            
            max_child_cp = 0.0
            for child_id in children_ids:
                max_child_cp = max(max_child_cp, get_cp_tail(child_id))
            
            # CP_tail = runtime + max(child_CP_tail)
            cp_val = task.wf_runtime_sec + max_child_cp
            memo_cp[tid] = cp_val
            return cp_val

        # Compute for all tasks
        for tid in self.task_map:
            get_cp_tail(tid)

        # 2. Compute Workflow Deadlines
        # Group by workflow_id
        wf_makespans = defaultdict(float)
        self.workflow_meta = defaultdict(lambda: {'total_tasks': 0, 'deadline': 0.0})
        
        for tid, task in self.task_map.items():
            wf_id = task.workflow_id
            # The makespan of a workflow is the max CP_tail among all its tasks (specifically sources)
            if memo_cp[tid] > wf_makespans[wf_id]:
                wf_makespans[wf_id] = memo_cp[tid]
            
            self.workflow_meta[wf_id]['total_tasks'] += 1
        
        # 3. Set Deadline and Static Remaining Time
        deadline_factor = 1.5
        for tid, task in self.task_map.items():
            task.static_remaining_time = memo_cp[tid]
            wf_span = wf_makespans[task.workflow_id]
            task.deadline = wf_span * deadline_factor
            
            # Store deadline in meta
            self.workflow_meta[task.workflow_id]['deadline'] = task.deadline
            
        logging.info("Slack parameters initialized for all tasks.")

    def _consume_resources(self, t, node_id):
        req_cpu = t.core_count if t.core_count else 1
        req_mem = t.req_mem if t.req_mem else 0.1
        req_gpu = t.req_gpu_mem if t.req_gpu_mem else 0.0
        
        self.node_specs[node_id]['remaining_cpu'] -= req_cpu
        self.node_specs[node_id]['remaining_mem'] -= req_mem
        self.node_specs[node_id]['remaining_gpu_mem'] -= req_gpu

    def _release_resources(self, t, node_id):
        req_cpu = t.core_count if t.core_count else 1
        req_mem = t.req_mem if t.req_mem else 0.1
        req_gpu = t.req_gpu_mem if t.req_gpu_mem else 0.0
        
        self.node_specs[node_id]['remaining_cpu'] += req_cpu
        self.node_specs[node_id]['remaining_mem'] += req_mem
        self.node_specs[node_id]['remaining_gpu_mem'] += req_gpu
        
        self.node_specs[node_id]['remaining_cpu'] = min(self.node_specs[node_id]['remaining_cpu'], self.node_specs[node_id]['cpu'])
        self.node_specs[node_id]['remaining_mem'] = min(self.node_specs[node_id]['remaining_mem'], self.node_specs[node_id]['mem'])
        self.node_specs[node_id]['remaining_gpu_mem'] = min(self.node_specs[node_id]['remaining_gpu_mem'], self.node_specs[node_id]['gpu_mem'])

    def _update_ready_queue(self):
        # 1. Update Agent-Managed Tasks
        new_ready = []
        for t in self.all_tasks:
            if t.unique_id not in self.completed_tasks and \
               t not in self.ready_queue and \
               t not in self.batch_queue and \
               t != self.current_task and \
               not any(rt[0].unique_id == t.unique_id for rt in self.running_tasks) and \
               self.dag_in_degree[t.unique_id] == 0:
                new_ready.append(t)
        
        # Task Sorting Strategy:
        # 1. Critical Path (CP_tail) - Descending (Longest path first)
        # 2. Number of Children - Descending (Unlock more tasks)
        # 3. Deadline - Ascending (Earliest deadline first)
        self.ready_queue.extend(new_ready)
        self.ready_queue.sort(key=lambda x: (x.static_remaining_time, len(self.dag_adj[x.unique_id]), -x.deadline), reverse=True)
        
        # 2. Update Local Pending Tasks (Auto-Schedule) - DEPRECATED
        # With new logic (Source tasks only), local tasks are auto-started in _load_tasks
        # So local_pending_tasks is unused. Kept empty for safety.
        self.local_pending_tasks = []

    def _advance_time_step(self):
        self.current_time += self.time_interval

        # # Debug info for remaining tasks
        # if int(self.current_time) % 10 == 0:
        #     remaining_scheduled = sum(1 for t in self.all_tasks if t.unique_id not in self.completed_tasks)
        #     print(f"[DEBUG] Time: {self.current_time:.2f}s | Remaining Scheduled: {remaining_scheduled}/{len(self.all_tasks)} | Running: {len(self.running_tasks)} | Total Completed: {len(self.completed_tasks)}")
        
        still_running = []
        for t, end_t, node_id in self.running_tasks:
            if end_t <= self.current_time + 1e-6:
                self.completed_tasks.add(t.unique_id)
                try:
                    self.node_specs[node_id]['queue'].remove(t)
                except ValueError:
                    pass
                if t.unique_id in self.dag_adj:
                    for child_id in self.dag_adj[t.unique_id]:
                        self.dag_in_degree[child_id] -= 1
                
                # Update Workflow Stats
                wf_id = t.workflow_id
                self.workflow_progress[wf_id]['tasks_done'] += 1
                self.workflow_progress[wf_id]['end_time'] = max(self.workflow_progress[wf_id]['end_time'], end_t)
                
                # Check if Workflow is complete
                if self.workflow_progress[wf_id]['tasks_done'] >= self.workflow_meta[wf_id]['total_tasks']:
                    # Only add if not already added (to avoid duplicates if tasks finish same step)
                    # But tasks_done is incremented, so it should hit equal only once if exact match
                    # Use >= just in case. But to be safe, check if we already marked it?
                    # Since we iterate tasks, and modify state, let's just check if it was already completed?
                    # We can use a set for completed workflows?
                    # Or simpler:
                    if wf_id not in [cw['id'] for cw in self.completed_workflows]:
                         # Workflow Finished
                         start_time = self.workflow_progress[wf_id]['start_time']
                         end_time = self.workflow_progress[wf_id]['end_time']
                         deadline = self.workflow_meta[wf_id]['deadline']
                         
                         self.completed_workflows.append({
                             'id': wf_id,
                             'completion_time': end_time - start_time,
                             'met_ddl': end_time <= deadline,
                             'deadline': deadline,
                             'makespan': end_time - start_time
                         })
                         # logging.info(f"Workflow {wf_id} completed. Time: {end_time - start_time:.2f}s. Met DDL: {end_time <= deadline}")

                # logging.info(f"[Time {self.current_time:.2f}] Task {t.unique_id} finished on {node_id}.")
            else:
                still_running.append((t, end_t, node_id))
        self.running_tasks = still_running

    def _next_batch_task(self):
        if self.current_task: return
        
        while not self.batch_queue:
            if self._is_done():
                self.current_task = None
                return

            if not self.ready_queue:
                self._advance_time_step()
                self._update_ready_queue()
                continue
            
            self.batch_queue = list(self.ready_queue)
            self.ready_queue = [] 
            
        if self.batch_queue:
            self.current_task = self.batch_queue.pop(0)
        else:
            self.current_task = None

    def _is_done(self):
        return not self.running_tasks and not self.ready_queue and not self.batch_queue and self.current_task is None

    def _get_state(self):
        # State Dimension: 
        # 112 queues (100 Pi + 10 Jetson + 2 Cloud)
        # 1 ready_count
        # 8 task features (added slack)
        # Total = 112 + 1 + 8 = 121
        
        total_tasks = max(1, len(self.all_tasks) + len(self.running_tasks) + len(self.completed_tasks))

        queues = []
        for n in self.all_nodes:
            queues.append(len(self.node_specs[n]['queue']) / float(total_tasks))
            
        ready_count = (len(self.batch_queue) + len(self.ready_queue) + (1 if self.current_task else 0)) / float(total_tasks)
        
        if self.current_task:
            t = self.current_task
            
            # Calculate Slack
            # slack(p,t) = Dp - (t + R(p,t))
            # R(p,t) = t.static_remaining_time (approx)
            slack = t.deadline - self.current_time - t.static_remaining_time
            if self.disable_slack_feature:
                slack = 0.0
            
            task_features = [
                t.wf_runtime_sec / 100.0,
                t.core_count / 4.0,
                t.req_mem / 8.0,
                float(t.privacy_level) / 3.0,
                t.req_gpu_mem / 1.0,
                t.data_input_size / 1e7,
                t.data_output_size / 1e7,
                slack / 100.0,
            ]
        else:
            task_features = [0.0] * 8
            
        state = queues + [ready_count] + task_features
        return np.array(state, dtype=np.float32)

    def _assign_task(self, t, node_id):
        t.assigned_node = node_id
        
        max_dependency_ready_time = 0.0
        total_comm_delay = 0.0
        
        for p_name in t.parents:
            if p_name in self.task_map:
                parent_task = self.task_map[p_name]
                if parent_task.unique_id in self.task_infos:
                    p_info = self.task_infos[parent_task.unique_id]
                    p_end = p_info['end']
                    p_node_id = parent_task.assigned_node
                    
                    transfer_time = 0.0
                    if p_node_id != node_id:
                        s_idx = self._get_node_idx(p_node_id)
                        d_idx = self._get_node_idx(node_id)
                        bw = self.bw_matrix[s_idx][d_idx]
                        if bw > 0:
                            transfer_time = parent_task.data_output_size / bw
                            
                    total_comm_delay += transfer_time
                    ready_at = p_end + transfer_time
                    if ready_at > max_dependency_ready_time:
                        max_dependency_ready_time = ready_at
                        
        task_ready_time = max(self.current_time, max_dependency_ready_time)
        req_cpu = float(t.core_count) if t.core_count else 1.0
        req_mem = float(t.req_mem) if t.req_mem else 0.1
        req_gpu = float(t.req_gpu_mem) if t.req_gpu_mem else 0.0
        
        cap_cpu = float(self.node_specs[node_id].get('cpu', 0.0))
        cap_mem = float(self.node_specs[node_id].get('mem', 0.0))
        cap_gpu = float(self.node_specs[node_id].get('gpu_mem', 0.0))
        
        start_t = task_ready_time
        
        node_tasks = []
        for rt, rt_end, rt_node in self.running_tasks:
            if rt_node != node_id:
                continue
            rt_start = self.task_infos.get(rt.unique_id, {}).get('start', None)
            if rt_start is None:
                continue
            node_tasks.append((rt, float(rt_start), float(rt_end)))
        
        candidate_times = {task_ready_time}
        for _, _, rt_end in node_tasks:
            # Only consider times >= task_ready_time
            if rt_end >= task_ready_time - 1e-9:
                candidate_times.add(rt_end)
                
        # Optimization: Sort and check earliest feasible slot
        # To avoid infinite future search, we can also check intervals
        
        start_t = task_ready_time # Default to ready time if feasible
        found_slot = False
        
        # Check ready time first
        used_cpu = 0.0
        used_mem = 0.0
        used_gpu = 0.0
        for rt, rt_start, rt_end in node_tasks:
             if rt_start <= task_ready_time < rt_end:
                 used_cpu += float(rt.core_count) if rt.core_count else 1.0
                 used_mem += float(rt.req_mem) if rt.req_mem else 0.1
                 used_gpu += float(rt.req_gpu_mem) if rt.req_gpu_mem else 0.0
        
        if (used_cpu + req_cpu <= cap_cpu + 1e-9) and (used_mem + req_mem <= cap_mem + 1e-9) and (used_gpu + req_gpu <= cap_gpu + 1e-9):
             start_t = task_ready_time
             found_slot = True
        else:
             # Check candidate completion times
             for cand in sorted(candidate_times):
                 if cand < task_ready_time: continue
                 
                 used_cpu = 0.0
                 used_mem = 0.0
                 used_gpu = 0.0
                 for rt, rt_start, rt_end in node_tasks:
                     if rt_start <= cand < rt_end:
                         used_cpu += float(rt.core_count) if rt.core_count else 1.0
                         used_mem += float(rt.req_mem) if rt.req_mem else 0.1
                         used_gpu += float(rt.req_gpu_mem) if rt.req_gpu_mem else 0.0
                 
                 if (used_cpu + req_cpu <= cap_cpu + 1e-9) and (used_mem + req_mem <= cap_mem + 1e-9) and (used_gpu + req_gpu <= cap_gpu + 1e-9):
                     start_t = cand
                     found_slot = True
                     break
        
        if not found_slot:
             # Fallback: Append to the very end of all tasks on this node
             # This is a worst-case scenario to guarantee execution
             max_end = task_ready_time
             for _, _, rt_end in node_tasks:
                 max_end = max(max_end, rt_end)
             start_t = max_end
        
        speed_factor = self.node_specs[node_id].get('speed', 1.0)
        base_duration = t.wf_runtime_sec if t.wf_runtime_sec else 1.0
        duration = base_duration / speed_factor
        
        end_t = start_t + duration
        comp_queue_wait = max(0.0, start_t - task_ready_time)
        
        self.node_specs[node_id]['queue'].append(t)
        
        self.running_tasks.append((t, end_t, node_id))
        self.task_infos[t.unique_id] = {'start': start_t, 'end': end_t}
        
        return comp_queue_wait, total_comm_delay, duration

    def _get_action_mask(self):
        """
        Hierarchical Masking helper.
        Returns a dict with masks for different levels/agents.
        """
        if self.disable_action_mask:
            return {
                'manager': np.ones(3, dtype=np.bool_),
                'local': np.ones(100, dtype=np.bool_),
                'edge': np.ones(10, dtype=np.bool_),
                'cloud': np.ones(2, dtype=np.bool_),
            }

        if self.current_task is None:
            return {
                'manager': np.ones(3, dtype=np.bool_), # 0: Local, 1: Edge, 2: Cloud
                'local': np.ones(100, dtype=np.bool_),
                'edge': np.ones(10, dtype=np.bool_),
                'cloud': np.ones(2, dtype=np.bool_)
            }
            
        task = self.current_task
        
        # 1. Node Feasibility Mask (All Nodes 112)
        node_mask = np.ones(112, dtype=np.bool_)
        
        # Privacy & Resource Check
        for i, node in enumerate(self.all_nodes):
            # Privacy
            if task.allowed_nodes and node not in task.allowed_nodes:
                node_mask[i] = 0
                continue
            
            req_cpu = float(task.core_count) if task.core_count else 1.0
            req_mem = float(task.req_mem) if task.req_mem else 0.1
            req_gpu = float(task.req_gpu_mem) if task.req_gpu_mem else 0.0
            
            if float(self.node_specs[node].get('cpu', 0.0)) + 1e-9 < req_cpu:
                node_mask[i] = 0
                continue
            if float(self.node_specs[node].get('mem', 0.0)) + 1e-9 < req_mem:
                node_mask[i] = 0
                continue
            if req_gpu > 0 and float(self.node_specs[node].get('gpu_mem', 0.0)) + 1e-9 < req_gpu:
                node_mask[i] = 0
                continue

        # Fallback if no nodes feasible
        if node_mask.sum() == 0:
            logging.warning(f"Task {task.unique_id} has no feasible nodes! Relaxing resources.")
            # Relax resources, keep privacy
            for i, node in enumerate(self.all_nodes):
                if task.allowed_nodes and node not in task.allowed_nodes:
                    node_mask[i] = 0
                else:
                    node_mask[i] = 1
        
        # 2. Construct Tier Masks
        # Local (0-99), Edge (100-109), Cloud (110-111)
        local_mask = node_mask[0:100]
        edge_mask = node_mask[100:110]
        cloud_mask = node_mask[110:112]
        
        # Manager Mask: If a tier has ANY valid node, it is a valid action
        manager_mask = np.zeros(3, dtype=np.bool_)
        if local_mask.sum() > 0: manager_mask[0] = 1
        if edge_mask.sum() > 0: manager_mask[1] = 1
        if cloud_mask.sum() > 0: manager_mask[2] = 1
        
        # Safety: If manager has no options (shouldn't happen due to fallback), enable all
        if manager_mask.sum() == 0:
            manager_mask[:] = 1
            local_mask[:] = 1 # Force enable
        
        return {
            'manager': manager_mask,
            'local': local_mask,
            'edge': edge_mask,
            'cloud': cloud_mask
        }

    def step(self, tier_action, node_action):
        """
        tier_action: 0 (Local), 1 (Edge), 2 (Cloud)
        node_action: index within the tier
        """
        if self.current_task is None:
             return self._get_state(), 0, True, self._get_action_mask(), {}
             
        task = self.current_task
        
        # Map actions to global node index
        target_node = None
        if tier_action == 0: # Local
            idx = min(node_action, 99)
            target_node = self.pis[idx]
        elif tier_action == 1: # Edge
            idx = min(node_action, 9)
            target_node = self.jetsons[idx]
        elif tier_action == 2: # Cloud
            idx = min(node_action, 1)
            target_node = self.cloud[idx]
            
        invalid_action_penalty = 0.0
        if task.allowed_nodes and target_node not in task.allowed_nodes:
            invalid_action_penalty = -10.0
            target_node = task.home_pi if task.home_pi else self.pis[0]

        wait, comm, dur = self._assign_task(task, target_node)

        # --- Reward Calculation ---
        # 1. Latency (Weighted Time)
        w_wait = 1.0
        w_dur = 5.0
        w_comm = 2.0

        weighted_time = (w_wait * wait) + (w_comm * comm) + (w_dur * dur)
        r_latency = - weighted_time / 100.0

        # 2. DDL Penalty
        finish_time = self.current_time + wait + comm + dur
        r_ddl = 0.0
        if finish_time > task.deadline:
            # Penalty: -1.0 fixed + delay scaled
            r_ddl = -1.0 - (finish_time - task.deadline) / 100.0

        r_slack = (task.wf_runtime_sec - (wait + comm + dur)) / 100.0

        # Composite Reward
        W_lat = 8.0
        W_ddl = self.reward_weight_ddl
        W_slack = self.reward_weight_slack

        reward = W_lat * r_latency + W_ddl * r_ddl + W_slack * r_slack + invalid_action_penalty

        # Update Global Stats
        self.workflow_progress[task.workflow_id]['start_time'] = min(self.workflow_progress[task.workflow_id]['start_time'], self.current_time)

        # Track QoE (Slack)
        # Remaining path time (approx)
        rem_path_time = max(0.0, task.static_remaining_time - task.wf_runtime_sec)
        slack_val = task.deadline - finish_time - rem_path_time
        self.total_slack_score += slack_val
        self.total_assignments += 1

        # logging.info(f"Step {task.unique_id}: Tier={tier_action}, Node={node_action} ({target_node}), R={reward:.2f}")
        
        self.current_task = None
        self._next_batch_task()
        
        done = self._is_done()
        mask = self._get_action_mask()
        
        # Calculate Metrics
        avg_wf_time = 0.0
        ddl_rate = 0.0
        if self.completed_workflows:
            avg_wf_time = sum(w['completion_time'] for w in self.completed_workflows) / len(self.completed_workflows)
            ddl_rate = sum(1 for w in self.completed_workflows if w['met_ddl']) / len(self.completed_workflows)
            
        avg_decision_path_slack = 0.0
        if self.total_assignments > 0:
            avg_decision_path_slack = self.total_slack_score / self.total_assignments
        
        info = {
            'wait': wait, 
            'comm': comm, 
            'dur': dur,
            'r_latency': r_latency,
            'r_ddl': r_ddl,
            'r_slack': r_slack,
            'avg_wf_time': avg_wf_time,
            'ddl_rate': ddl_rate,
            # Explicit semantic name for the task-decision-weighted online
            # path-slack diagnostic used by the legacy load-sweep logs.
            'avg_decision_path_slack': avg_decision_path_slack,
            # Backward-compatible field consumed by existing training and
            # plotting scripts; this is not an independent QoE score.
            'avg_path_qoe': avg_decision_path_slack
        }
        return self._get_state(), reward, done, mask, info
