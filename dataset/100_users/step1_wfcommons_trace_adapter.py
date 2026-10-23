#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd


# =========================
#   数据结构定义
# =========================

@dataclass
class WorkflowTask:
    """内部统一的任务表示"""
    id: str
    name: str
    parents: List[str]
    children: List[str]
    input_files: List[str]
    output_files: List[str]

    # 来自原始 WfCommons execution 部分
    wf_runtime_sec: Optional[float] = None
    core_count: Optional[float] = None

    # 来自 Trace 对接
    trace_row_idx: Optional[int] = None
    trace_runtime_sec: Optional[float] = None


# =========================
#   解析 WfCommons JSON
# =========================

def load_wfcommons_instance(json_path: str) -> Tuple[List[WorkflowTask], Dict[str, Any]]:
    """
    解析 WfCommons / WfFormat 实例 JSON，构造内部 WorkflowTask 列表，同时返回原始 JSON 便于后续修改。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    workflow = data["workflow"]
    spec = workflow["specification"]
    exec_info = workflow.get("execution", {})

    spec_tasks = {t["id"]: t for t in spec.get("tasks", [])}
    exec_tasks = {t["id"]: t for t in exec_info.get("tasks", [])}

    tasks: List[WorkflowTask] = []

    for tid, t_spec in spec_tasks.items():
        t_exec = exec_tasks.get(tid, {})
        wt = WorkflowTask(
            id=tid,
            name=t_spec.get("name", tid),
            parents=t_spec.get("parents", []),
            children=t_spec.get("children", []),
            input_files=t_spec.get("inputFiles", []),
            output_files=t_spec.get("outputFiles", []),
            wf_runtime_sec=t_exec.get("runtimeInSeconds"),
            core_count=t_exec.get("coreCount"),
        )
        tasks.append(wt)

    return tasks, data


# =========================
#   加载 Trace CSV
# =========================

def load_trace_csv(
    trace_csv: str,
    runtime_col: str = "runtime_sec",
) -> pd.DataFrame:
    """
    加载预处理后的 Google Cluster Trace CSV。

    要求：
      - 至少存在一列 runtime_col（默认 runtime_sec）
      - 其他列（job_id, task_index, cpu_rate, mem_usage...）会一起被带入到输出。
    """
    df = pd.read_csv(trace_csv)
    if runtime_col not in df.columns:
        raise ValueError(
            f"Trace CSV 中找不到运行时间列 '{runtime_col}'，当前列有: {list(df.columns)}"
        )

    # 删除 runtime 缺失的行
    df = df.dropna(subset=[runtime_col]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Trace CSV 中所有行的 runtime 都是 NaN，请检查输入。")

    # =========================================================
    # MODIFICATION: Clamp resource requirements to fit Edge Nodes
    # =========================================================
    # Pi: 8GB, Jetson: 16GB. Cloud: 128GB.
    # To ensure tasks are feasible on at least Jetson (Level 1),
    # we clamp Memory to 15.0 GB and GPU Mem to 7.5 GB.
    
    if 'max_mem' in df.columns:
        df['max_mem'] = df['max_mem'].clip(upper=15.0)
        
    if 'max_gpu_wrk_mem' in df.columns:
        df['max_gpu_wrk_mem'] = df['max_gpu_wrk_mem'].clip(upper=7.5)
    # =========================================================

    return df


# =========================
#   将 Trace 行映射到任务
# =========================

# 修改 map_tasks_to_trace 函数，添加按 start_time 排序的功能
def map_tasks_to_trace(
        tasks: List[WorkflowTask],
        trace_df: pd.DataFrame,
        runtime_col: str = "runtime_sec",
        start_time_col: str = "start_time",
        shuffle: bool = True,
        seed: int = 42,
) -> Tuple[List[WorkflowTask], pd.DataFrame]:
    """
    将 Trace 记录映射到每个 DAG 任务上，并按开始时间排序：
      - 若 trace_df 行数 >= 任务数：前 N 行先按 start_time 排序，再分配给 N 个任务。
      - 若 trace_df 行数 < 任务数：循环使用 trace_df 行（可视为样本复用）。
    """
    # 检查是否存在 start_time 列
    if start_time_col not in trace_df.columns:
        raise ValueError(
            f"Trace CSV 中找不到开始时间列 '{start_time_col}'，当前列有: {list(trace_df.columns)}"
        )

    if shuffle:
        trace_df = trace_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_tasks = len(tasks)
    n_rows = len(trace_df)

    if n_rows < n_tasks:
        reps = math.ceil(n_tasks / n_rows)
        trace_df_rep = pd.concat([trace_df] * reps, ignore_index=True)
    else:
        trace_df_rep = trace_df.copy()

    # 按照 start_time 排序
    trace_df_rep = trace_df_rep.sort_values(by=start_time_col).reset_index(drop=True)

    # 确保我们有足够的行数分配给所有任务
    if len(trace_df_rep) < n_tasks:
        # 如果仍然不够，再次复制直到满足需求
        reps = math.ceil(n_tasks / len(trace_df_rep))
        trace_df_rep = pd.concat([trace_df_rep] * reps, ignore_index=True)

    # 给每个任务挂上 Trace 信息
    for i, t in enumerate(tasks):
        row = trace_df_rep.iloc[i]
        t.trace_row_idx = int(i)
        t.trace_runtime_sec = float(row[runtime_col])

    return tasks, trace_df_rep


# =========================
#   输出：CSV + 更新后的 JSON
# =========================

def export_tasks_with_trace_csv(
    tasks: List[WorkflowTask],
    trace_df_rep: pd.DataFrame,
    output_csv: str,
) -> None:
    """
    导出一个 CSV：每行一个 DAG 任务，同时拼上选中的 Trace 记录（前缀 trace_）。
    """
    rows = []
    for t in tasks:
        base = {
            "task_id": t.id,
            "task_name": t.name,
            "parents": ";".join(t.parents),
            "children": ";".join(t.children),
            "input_files": ";".join(t.input_files),
            "output_files": ";".join(t.output_files),
            "wf_runtime_sec": t.wf_runtime_sec,
            "core_count": t.core_count,
            "trace_runtime_sec": t.trace_runtime_sec,
        }

        # 如果需要，也可以把完整 trace 行拼进来
        if t.trace_row_idx is not None:
            trace_row = trace_df_rep.iloc[t.trace_row_idx]
            for col in trace_df_rep.columns:
                base[f"trace_{col}"] = trace_row[col]

        rows.append(base)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 任务 + Trace 对齐结果已导出到: {output_csv}")


def update_wfcommons_runtime_and_export(
    original_json: Dict[str, Any],
    tasks: List[WorkflowTask],
    output_json: str,
) -> None:
    """
    用 Trace 的 runtime 替换 WfCommons execution.tasks 里的 runtimeInSeconds。
    """
    exec_tasks = original_json["workflow"].get("execution", {}).get("tasks", [])
    exec_by_id = {t["id"]: t for t in exec_tasks}

    for t in tasks:
        if t.trace_runtime_sec is None:
            continue
        if t.id in exec_by_id:
            exec_by_id[t.id]["runtimeInSeconds"] = float(t.trace_runtime_sec)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(original_json, f, indent=2)

    print(f"[INFO] 已输出更新 runtime 后的 WfCommons JSON: {output_json}")


# =========================
#   多个 workflow 共用一个 trace（入口）
# =========================

# 在主程序中修改以下部分，添加 start_time_col 参数
if __name__ == "__main__":
    # === 0) 基础路径设定 ===
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "tasks")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录: {output_dir}")

    # === 1) 模板 workflow JSON 路径 ===
    # 假设这些文件就在当前脚本同级目录下
    template_paths = [
        os.path.join(base_dir, f"bwa-chameleon{i}--workflow.json")
        for i in range(1, 6)
    ]
    
    # 设定生成的用户数量
    NUM_USERS = 100
    print(f"[INFO] 计划生成 {NUM_USERS} 个用户的任务数据...")

    # === 2) 公用的 trace CSV 路径 ===
    # 假设在上一级目录
    trace_csv_path = os.path.join(base_dir, "..", "pai_sensor_table_final.csv")
    runtime_col = "comp_time"  # 运行时间列
    start_time_col = "start_time"  # 开始时间列

    # === 3) 一次性加载 trace，并全局打乱 ===
    if not os.path.exists(trace_csv_path):
        # 尝试绝对路径或其他位置（如果不在上一级）
        # 这里保留原始逻辑作为一个 fallback 或者报错
        print(f"[WARN] 在 {trace_csv_path} 没找到 trace 文件，尝试使用当前目录...")
        trace_csv_path = os.path.join(base_dir, "pai_sensor_table_final.csv")

    trace_df_full = load_trace_csv(trace_csv_path, runtime_col=runtime_col)
    print(f"[INFO] 原始 trace 样本数: {len(trace_df_full)}")

    # 构造 100 个用户的任务列表
    # workflow_tasks_list 结构: List[Dict]
    workflow_tasks_list = []
    total_tasks = 0

    for user_id in range(NUM_USERS):
        # 轮询使用模板
        template_path = template_paths[user_id % len(template_paths)]
        
        # 加载解析
        tasks, wf_json = load_wfcommons_instance(template_path)
        
        workflow_tasks_list.append({
            "user_id": user_id,
            "template_path": template_path,
            "tasks": tasks,
            "wf_json": wf_json
        })
        total_tasks += len(tasks)

    print(f"[INFO] {NUM_USERS} 个用户总任务数: {total_tasks}")

    if len(trace_df_full) < total_tasks:
        print("[WARN] trace 行数少于总任务数，将会发生样本复用（不可避免）。")

    # 无放回抽样 min(total_tasks, len(trace_df_full)) 行
    trace_sample = trace_df_full.sample(
        n=min(total_tasks, len(trace_df_full)),
        random_state=42
    ).reset_index(drop=True)

    # 如果 trace 自身样本少于总任务数，则重复扩展
    if len(trace_sample) < total_tasks:
        reps = math.ceil(total_tasks / len(trace_sample))
        trace_sample = pd.concat([trace_sample] * reps, ignore_index=True)

    # 按照 start_time 排序
    if start_time_col in trace_sample.columns:
        trace_sample = trace_sample.sort_values(by=start_time_col).reset_index(drop=True)
    else:
        print(f"[WARN] Trace CSV 中找不到开始时间列 '{start_time_col}'，将不按时间排序")

    assert len(trace_sample) >= total_tasks

    # === 4) 按 workflow 顺序切片分配 trace 行 ===
    cursor = 0
    for item in workflow_tasks_list:
        user_id = item["user_id"]
        template_path = item["template_path"]
        tasks = item["tasks"]
        wf_json = item["wf_json"]

        n_tasks = len(tasks)
        slice_df = trace_sample.iloc[cursor:cursor + n_tasks].reset_index(drop=True)
        cursor += n_tasks

        # 映射 Trace
        tasks, trace_df_rep = map_tasks_to_trace(
            tasks,
            slice_df,
            runtime_col=runtime_col,
            start_time_col=start_time_col,
            shuffle=False,
            seed=42,
        )

        # 构造输出前缀: tasks/user_{id}_{template_name}
        template_name = os.path.splitext(os.path.basename(template_path))[0]
        # output_prefix 示例: .../dataset/100_users/tasks/user_0_bwa-chameleon1
        output_prefix = os.path.join(
            output_dir,
            f"user_{user_id}_{template_name}"
        )

        # 导出 CSV
        export_tasks_with_trace_csv(
            tasks,
            trace_df_rep,
            output_prefix + "_tasks.csv",
        )

        # 回写 JSON
        update_wfcommons_runtime_and_export(
            wf_json,
            tasks,
            output_prefix + "_wfcommons.json",
        )

        if user_id % 10 == 0:
            print(f"[INFO] User {user_id} ({template_name}) 处理完成")
            
    print("[INFO] 所有 100 个用户的任务生成完毕！")
