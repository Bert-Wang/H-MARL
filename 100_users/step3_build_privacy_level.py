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

    # 100个用户，映射到 100 个 Local Nodes (0-99)
    # workflow_label 格式: "user_0", "user_1", ...
    
    # 定义节点 ID 范围
    # 100 Locals: 0-99
    # 10 Jetsons: 100-109
    # 2 Cloud: 110-111
    locals = list(range(100))
    jetsons = list(range(100, 110))  # 100..109 (10 nodes)
    cloud = list(range(110, 112))    # 110..111 (2 nodes)

    home_pi_list = []
    allowed_nodes_list = []

    for _, row in df.iterrows():
        wf = str(row["workflow_label"])
        level = int(row["privacy_level"])
        
        # 提取 user_id
        # 格式 "user_123" -> 123
        try:
            user_id = int(wf.split("_")[-1])
        except ValueError:
            user_id = 0
            
        # 确保 user_id 在 0-99 范围内，直接对应 local node id
        if 0 <= user_id < 100:
            home_pi = locals[user_id]
        else:
            # Fallback (should not happen if filenames are correct)
            home_pi = locals[user_id % 100]
            
        home_pi_list.append(home_pi)

        if level == 0:
            # 强隐私：只能在本地 (Local)
            allowed = [home_pi]
        elif level == 1:
            # 中等隐私：本地 + 边缘 (Local + Edge)
            allowed = [home_pi] + jetsons
        else:  # level == 2
            # 低隐私：本地 + 边缘 + 云 (Local + Edge + Cloud)
            allowed = [home_pi] + jetsons + cloud

        allowed_nodes_list.append(",".join(str(x) for x in allowed))

    df["home_pi"] = home_pi_list
    df["allowed_nodes"] = allowed_nodes_list

    return df


if __name__ == "__main__":
    # Base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ==== 输入 & 输出路径 ====
    input_csv = os.path.join(base_dir, "global_100users_with_random_edges.csv")
    output_csv = os.path.join(base_dir, "global_100users_with_privacy.csv")

    if not os.path.exists(input_csv):
        print(f"[ERROR] 输入文件不存在: {input_csv}")
        print("请先运行 step2_build_global_random_dag.py 生成该文件。")
    else:
        df = pd.read_csv(input_csv)
        df_out = add_home_and_allowed_nodes(df)
        df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

        print(f"[INFO] 已生成带隐私属性的数据集：{output_csv}")
        print(f"[INFO] 本地节点(Locals): 0-99 (100个)")
        print(f"[INFO] 边缘节点(Jetsons): 100-109 (10个)")
        print(f"[INFO] 云节点(Cloud): 110-111 (2个)")
