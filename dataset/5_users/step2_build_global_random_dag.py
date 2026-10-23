#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于已有的 5 个 *_google_tasks.csv，构造一个带随机跨-workflow依赖的大 DAG。

输入：
    - 每个 workflow 一个 *_google_tasks.csv
      （列里至少包含：task_id, parents, children, wf_runtime_sec, trace_runtime_sec, trace_* 等）

输出：
    1) 一个全局 CSV：global_5wf_with_random_edges.csv
       - 每行一个任务（5 个 workflow 合在一起）
       - 新增列：
         * workflow_label   : wf1 ~ wf5
         * global_id        : 比如 "wf1:ID0000001"
         * global_parents   : 使用 global_id 表示的父任务（包含本地 + 跨workflow）
         * global_children  : 使用 global_id 表示的子任务（包含本地 + 跨workflow）
    2) 一个 cross_edges.csv（可选）：保存随机生成的跨-workflow边列表
"""

import os
import random
from typing import List, Tuple, Dict

import pandas as pd


def load_workflow_tasks(csv_path: str, workflow_label: str) -> pd.DataFrame:
    """
    读取单个 *_google_tasks.csv，增加 workflow_label / global_id / global_parents / global_children，
    并将原有的 parents/children 转成全局版本（仅限本 workflow 内）。
    """
    df = pd.read_csv(csv_path)

    # 确保 task_id 是字符串
    df["task_id"] = df["task_id"].astype(str)

    # workflow 标记 & 全局 ID（用于后面跨-workflow依赖）
    df["workflow_label"] = workflow_label
    df["global_id"] = df["workflow_label"] + ":" + df["task_id"]

    # 初始化全局父子依赖列
    df["global_parents"] = ""
    df["global_children"] = ""

    # 构造本 workflow 内 local_id -> global_id / 行索引 的映射
    local_ids = df["task_id"].tolist()
    global_ids = df["global_id"].tolist()
    lid_to_gid: Dict[str, str] = dict(zip(local_ids, global_ids))
    lid_to_idx: Dict[str, int] = {tid: idx for idx, tid in enumerate(local_ids)}
    gid_to_idx: Dict[str, int] = {gid: idx for idx, gid in enumerate(global_ids)}

    # 先用 parents 列生成 global_parents
    for idx, row in df.iterrows():
        parents_str = row.get("parents", "")
        if pd.isna(parents_str):
            parents_str = ""
        parents_str = str(parents_str).strip()
        if not parents_str:
            continue

        local_parents = [p for p in parents_str.split(";") if p]
        global_parents = [lid_to_gid[p] for p in local_parents if p in lid_to_gid]
        df.at[idx, "global_parents"] = ";".join(global_parents)

    # 再根据 global_parents 生成 global_children（只在本 workflow 内）
    for idx, row in df.iterrows():
        gid = row["global_id"]
        gp_str = row.get("global_parents", "")
        if pd.isna(gp_str):
            gp_str = ""
        gp_str = str(gp_str).strip()
        if not gp_str:
            continue

        parents_global = [pg for pg in gp_str.split(";") if pg]
        for pg in parents_global:
            # 只处理本 workflow 内的父节点
            if pg in gid_to_idx:
                p_idx = gid_to_idx[pg]
                cur_children = df.at[p_idx, "global_children"]
                if pd.isna(cur_children):
                    cur_children = ""
                cur_children = str(cur_children).strip()
                if not cur_children:
                    df.at[p_idx, "global_children"] = gid
                else:
                    children_set = set(cur_children.split(";"))
                    if gid not in children_set:
                        df.at[p_idx, "global_children"] = cur_children + ";" + gid

    return df


def add_random_cross_edges_between_adjacent_workflows(
    dfs: List[pd.DataFrame],
    workflow_labels: List[str],
    edges_per_pair: int = 10,
    seed: int = 42,
) -> Tuple[List[pd.DataFrame], List[Tuple[str, str]]]:
    """
    在相邻的 workflow 对之间随机生成跨-workflow依赖边：
        wf1 -> wf2, wf2 -> wf3, wf3 -> wf4, wf4 -> wf5

    每对之间生成 edges_per_pair 条边（如果任务数太少，可能略少一点），
    边的形式为 (src_global_id, dst_global_id)。

    为了保证仍是 DAG，方向固定为 “前一个 workflow 指向后一个 workflow”，
    就不会引入环。
    """
    rng = random.Random(seed)

    # 为了快速定位 global_id -> (wf_idx, row_idx)
    gid_to_loc: Dict[str, Tuple[int, int]] = {}
    for wf_idx, df in enumerate(dfs):
        for row_idx, row in df.iterrows():
            gid_to_loc[row["global_id"]] = (wf_idx, row_idx)

    all_cross_edges: List[Tuple[str, str]] = []

    # 只在相邻 workflow 之间连边：wf[i] -> wf[i+1]
    for i in range(len(dfs) - 1):
        src_df = dfs[i]
        dst_df = dfs[i + 1]

        src_ids = src_df["global_id"].tolist()
        dst_ids = dst_df["global_id"].tolist()

        if not src_ids or not dst_ids:
            continue

        created = 0
        attempts = 0
        max_attempts = edges_per_pair * 20

        # 用集合避免重复 (src, dst)
        existing_pairs = set()

        while created < edges_per_pair and attempts < max_attempts:
            src_gid = rng.choice(src_ids)
            dst_gid = rng.choice(dst_ids)
            pair = (src_gid, dst_gid)

            attempts += 1
            if pair in existing_pairs:
                continue

            existing_pairs.add(pair)
            all_cross_edges.append(pair)
            created += 1

            # 更新 dst 的 global_parents
            dst_wf_idx, dst_row_idx = gid_to_loc[dst_gid]
            src_wf_idx, src_row_idx = gid_to_loc[src_gid]

            # 取出原来的 global_parents
            parents_str = dfs[dst_wf_idx].at[dst_row_idx, "global_parents"]
            if pd.isna(parents_str):
                parents_str = ""
            parents_str = str(parents_str).strip()
            parents_set = set(p for p in parents_str.split(";") if p)

            if src_gid not in parents_set:
                parents_set.add(src_gid)
                dfs[dst_wf_idx].at[dst_row_idx, "global_parents"] = ";".join(sorted(parents_set))

            # 更新 src 的 global_children
            children_str = dfs[src_wf_idx].at[src_row_idx, "global_children"]
            if pd.isna(children_str):
                children_str = ""
            children_str = str(children_str).strip()
            children_set = set(c for c in children_str.split(";") if c)

            if dst_gid not in children_set:
                children_set.add(dst_gid)
                dfs[src_wf_idx].at[src_row_idx, "global_children"] = ";".join(sorted(children_set))

        print(
            f"[INFO] 在 {workflow_labels[i]} -> {workflow_labels[i+1]} 之间生成跨依赖边 {created} 条 "
            f"(目标 {edges_per_pair} 条，尝试次数 {attempts})"
        )

    return dfs, all_cross_edges


if __name__ == "__main__":
    # ========== 1. 配置 5 个 *_google_tasks.csv 路径和对应的 label ==========
    # 把下面的路径改成你自己生成的 5 个 _google_tasks.csv
    workflow_files = [
        (r"C:\Users\15188\Desktop\第二篇论文代码\dataset\bwa-chameleon1--workflow_google_tasks.csv",  "wf1"),
        (r"C:\Users\15188\Desktop\第二篇论文代码\dataset\bwa-chameleon2--workflow_google_tasks.csv",        "wf2"),
        (r"C:\Users\15188\Desktop\第二篇论文代码\dataset\bwa-chameleon3--workflow_google_tasks.csv",     "wf3"),
        (r"C:\Users\15188\Desktop\第二篇论文代码\dataset\bwa-chameleon4--workflow_google_tasks.csv",         "wf4"),
        (r"C:\Users\15188\Desktop\第二篇论文代码\dataset\bwa-chameleon5--workflow_google_tasks.csv",          "wf5"),
    ]
    # 上面只是示例文件名，请按你实际的 *_google_tasks.csv 文件名改

    # ========== 2. 读入 5 个 workflow 的 CSV 并构建本地全局依赖 ==========
    dfs: List[pd.DataFrame] = []
    labels: List[str] = []

    for csv_path, label in workflow_files:
        print(f"[INFO] 读取 {label} from {csv_path}")
        df = load_workflow_tasks(csv_path, label)
        dfs.append(df)
        labels.append(label)
        print(f"[INFO]   {label} 任务数: {len(df)}")

    # ========== 3. 随机生成跨-workflow依赖（相邻 workflow 之间） ==========
    # 每对相邻 workflow 生成多少条跨边，可以自己调
    EDGES_PER_PAIR = 30
    RANDOM_SEED = 42

    dfs, cross_edges = add_random_cross_edges_between_adjacent_workflows(
        dfs,
        workflow_labels=labels,
        edges_per_pair=EDGES_PER_PAIR,
        seed=RANDOM_SEED,
    )

    # ========== 4. 导出全局合并后的任务表 ==========
    global_df = pd.concat(dfs, ignore_index=True)

    # 你可以自己改输出路径
    global_output_csv = r"C:\Users\15188\Desktop\第二篇论文代码\dataset\global_5wf_with_random_edges.csv"
    global_df.to_csv(global_output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 全局任务表已写入: {global_output_csv}")

    # ========== 5. 可选：把随机跨边单独导出一个列表 ==========
    cross_edges_csv = r"C:\Users\15188\Desktop\第二篇论文代码\dataset\global_5wf_cross_edges.csv"
    if cross_edges:
        df_edges = pd.DataFrame(cross_edges, columns=["src_global_id", "dst_global_id"])
        df_edges.to_csv(cross_edges_csv, index=False, encoding="utf-8-sig")
        print(f"[INFO] 跨-workflow随机边列表已写入: {cross_edges_csv}")
    else:
        print("[INFO] 没有生成任何跨-workflow边（可能某些 workflow 任务数为 0）")
