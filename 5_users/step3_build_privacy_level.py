#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from collections import defaultdict, deque
import math
import os

# ===== 1. 计算每个 workflow 内部的拓扑深度 =====

def compute_topo_depths(df: pd.DataFrame) -> dict:
    """
    只根据同一个 workflow 内的依赖关系，计算每个 global_id 的拓扑深度。
    忽略跨-workflow的父节点（即 workflow_label 不同的 global_parents）。
    """
    depths = {}

    for wf, group in df.groupby("workflow_label"):
        ids = group["global_id"].tolist()
        indeg = {gid: 0 for gid in ids}
        children = defaultdict(list)

        # 先把本 workflow 内的边收集起来
        id_set = set(ids)
        for _, row in group.iterrows():
            gid = row["global_id"]
            parents_str = str(row.get("global_parents", "") or "").strip()
            if parents_str:
                parents = [p for p in parents_str.split(";") if p]
                for p in parents:
                    if p in id_set:  # 只考虑本 workflow 内的父节点
                        indeg[gid] += 1
                        children[p].append(gid)

        # Kahn 拓扑 + 分层
        q = deque([gid for gid, d in indeg.items() if d == 0])
        for gid in q:
            depths[gid] = 0

        while q:
            u = q.popleft()
            du = depths[u]
            for v in children.get(u, []):
                indeg[v] -= 1
                if indeg[v] == 0:
                    depths[v] = du + 1
                    q.append(v)

    return depths


# ===== 2. 根据深度分配隐私等级 =====

def assign_privacy_levels(df: pd.DataFrame) -> pd.DataFrame:
    depths = compute_topo_depths(df)
    df = df.copy()
    df["topo_depth"] = df["global_id"].map(depths).fillna(0).astype(int)

    privacy_levels = {}

    for wf, group in df.groupby("workflow_label"):
        max_d = group["topo_depth"].max()
        if max_d <= 0:
            # 所有任务都视为 depth=0，统一给中等隐私
            for idx in group.index:
                privacy_levels[idx] = 1
            continue

        for idx, row in group.iterrows():
            d = row["topo_depth"]
            norm_d = d / max_d
            
            # Get resource requirements (handle missing columns gracefully)
            req_mem = row.get("trace_max_mem", 0.0)
            if pd.isna(req_mem): req_mem = 0.0
            
            if norm_d <= 0.3:
                level = 0   # 强隐私，只能本地
                # Constraint: If task > 7.5GB, cannot run on Pi (8GB). Force offload to Edge.
                if req_mem > 7.5:
                    level = 1
                # Constraint: If task > 15.0GB, cannot run on Jetson (16GB). Force offload to Cloud.
                if req_mem > 15.0:
                    level = 2
            elif norm_d <= 0.7:
                level = 1   # 中等隐私，端+边
                # Constraint: If task > 15.0GB, cannot run on Jetson. Force offload to Cloud.
                if req_mem > 15.0:
                    level = 2
            else:
                level = 2   # 低隐私，端+边+云
            privacy_levels[idx] = level

    df["privacy_level"] = df.index.map(privacy_levels).astype(int)
    return df


# ===== 3. 映射到 home_pi 和 allowed_nodes =====

def add_home_and_allowed_nodes(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_privacy_levels(df)

    # workflow -> home Pi 编号映射，可按需修改
    workflow_to_home_pi = {
        "wf1": 0,
        "wf2": 1,
        "wf3": 2,
        "wf4": 3,
        "wf5": 4,
    }

    jetsons = [5, 6, 7, 8]
    server = [9]

    home_pi_list = []
    allowed_nodes_list = []

    for _, row in df.iterrows():
        wf = row["workflow_label"]
        level = int(row["privacy_level"])

        home_pi = workflow_to_home_pi.get(wf, 0)
        home_pi_list.append(home_pi)

        if level == 0:
            allowed = [home_pi]
        elif level == 1:
            allowed = jetsons
        else:  # level == 2
            allowed = jetsons + server

        allowed_nodes_list.append(",".join(str(x) for x in allowed))

    df["home_pi"] = home_pi_list
    df["allowed_nodes"] = allowed_nodes_list

    return df


if __name__ == "__main__":
    # ==== 输入 & 输出路径，按你实际情况修改 ====
    input_csv = r"C:\Users\15188\Desktop\第二篇论文代码\dataset\global_5wf_with_random_edges.csv"
    output_csv = r"C:\Users\15188\Desktop\第二篇论文代码\dataset\global_5wf_with_privacy.csv"

    df = pd.read_csv(input_csv)
    df_out = add_home_and_allowed_nodes(df)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[INFO] 已生成带隐私属性的数据集：{output_csv}")
