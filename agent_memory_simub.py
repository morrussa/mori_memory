#!/usr/bin/env python3
"""Simplified long-run simulator for the current agent-memory design.

Core behaviors aligned with the Lua project:
- memory merge threshold (`merge_limit`)
- cluster-first retrieval with hot-memory fallback
- heat updates, neighbor heating, and global normalization
- optional topic-bucket retrieval (disabled by default)

Primary evaluation focus:
- heat distribution deterioration (`heat_gini` trend)
- whether needed memories can be recalled on demand
"""

from __future__ import annotations

import argparse
import bisect
import csv
import heapq
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm <= 0:
        return v
    return v / norm


@dataclass
class SimParams:
    turns: int = 20000
    dim: int = 256
    seed: int = 42
    report_every: int = 1000

    # Topic stream generator
    topic_groups: int = 20
    topics_per_group: int = 10
    topic_variant_mix: float = 0.35
    turn_noise_mix: float = 0.18
    switch_prob: float = 0.12
    near_switch_prob: float = 0.55

    # Query schedule / evaluation
    warmup_turns: int = 20
    query_prob: float = 0.45
    relevant_window: int = 40
    query_current_intent_prob: float = 0.60
    query_long_term_min_age: int = 120
    query_noise_mix: float = 0.14
    learning_curve_enabled: bool = True
    learning_warmup_turns: int = 500
    learning_full_turns: int = 12000
    learning_query_noise_extra: float = 0.18
    learning_min_sim_gate_start: float = 0.42
    learning_power_suppress_start: float = 1.15
    learning_topic_cross_quota_start: float = 0.48
    learning_max_memory_start: int = 3
    learning_max_turns_start: int = 14
    learning_keyword_weight_start: float = 0.78
    learning_super_topn_query_start: int = 2
    refinement_enabled: bool = True
    refinement_start_turn: int = 200
    refinement_sample_mem_topk: int = 48
    refinement_route_lr: float = 0.10
    refinement_gate_lr: float = 0.08
    refinement_merge_lr: float = 0.05
    refinement_route_bias_scale: float = 0.08
    refinement_probe_clusters_start: int = 8
    refinement_probe_clusters_end: int = 2
    refinement_probe_per_cluster_limit: int = 12
    persistent_explore_enabled: bool = True
    persistent_explore_epsilon: float = 0.01
    persistent_explore_period_turns: int = 0
    persistent_explore_extra_clusters: int = 1
    persistent_explore_candidate_cap: int = 32

    # Project-aligned config defaults (from module/config.lua)
    merge_limit: float = 0.95
    cluster_sim: float = 0.72
    hot_cluster_ratio: float = 0.65
    cluster_heat_cap: int = 180000
    max_neighbors: int = 5
    new_memory_heat: int = 43000
    neighbors_heat: int = 26000
    total_heat: int = 10000000
    softmax: bool = True
    tolerance: int = 500
    cold_neighbor_multiplier: float = 2.2

    loss_turn: int = 50
    time_boost: float = 0.2

    max_memory: int = 5
    max_turns: int = 10
    min_sim_gate: float = 0.58
    power_suppress: float = 1.80
    keyword_weight: float = 0.55
    keyword_queries: int = 2
    keyword_noise_mix: float = 0.20
    topic_sim_threshold: float = 0.70
    topic_cross_quota_ratio: float = 0.25
    use_topic_buckets: bool = False

    # Search pool size for add-memory merge check
    fast_scan_topk: int = 64
    hierarchical_cluster_enabled: bool = True
    supercluster_min_clusters: int = 64
    supercluster_target_size: int = 64
    supercluster_sim: float = 0.52
    supercluster_max_size_mult: float = 1.8
    supercluster_topn_add: int = 3
    supercluster_topn_query: int = 4
    supercluster_topn_scale: float = 0.20
    supercluster_rebuild_every: int = 600

    # Cold-memory rescue simulation (delayed salvage)
    maintenance_task: int = 75
    cold_rescue_delay_min: int = 24
    cold_rescue_delay_max: int = 120
    cold_rescue_topn: int = 3
    cold_rescue_batch: int = 24
    cold_rescue_on_empty_only: bool = False
    cold_wake_multiplier: float = 1.8
    cold_extra_neighbor_heat: int = 18000
    cold_rescue_max_queue: int = 50000

    # Topic-shift window to simulate natural miss periods
    shift_probe_turns: int = 12
    shift_query_prob_boost: float = 0.12
    shift_target_prev_prob: float = 0.55
    shift_query_noise_boost: float = 0.06

    # Topic-flow drift simulation + stable-topic random lift
    topic_flow_drift: float = 0.05
    topic_flow_anchor_mix: float = 0.22
    topic_flow_switch_jolt: float = 0.08
    stable_warmup_turns: int = 6
    stable_min_pair_sim: float = 0.72
    topic_random_lift_interval: int = 3
    topic_random_lift_count: int = 2
    topic_random_lift_prob: float = 0.85
    topic_random_lift_only_cold: bool = True
    topic_cache_weight: float = 1.02


@dataclass
class ClusterState:
    members: List[int] = field(default_factory=list)
    hot_count: int = 0
    cold_count: int = 0
    is_hot_cluster: bool = False


class TopicStream:
    def __init__(self, p: SimParams, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.topic_vectors, self.topic_groups = self._build_topics()
        self.num_topics = self.topic_vectors.shape[0]
        self.group_to_topics: List[List[int]] = [[] for _ in range(p.topic_groups)]
        for tid, gid in enumerate(self.topic_groups):
            self.group_to_topics[gid].append(tid)
        self.topic_sim = self.topic_vectors @ self.topic_vectors.T
        self._active_topic: Optional[int] = None
        self._flow_anchor: Optional[np.ndarray] = None

    def _rand_unit(self, dim: int) -> np.ndarray:
        return unit(self.rng.normal(size=dim).astype(np.float32))

    def _build_topics(self) -> Tuple[np.ndarray, np.ndarray]:
        p = self.p
        total_topics = p.topic_groups * p.topics_per_group
        vecs = np.zeros((total_topics, p.dim), dtype=np.float32)
        groups = np.zeros(total_topics, dtype=np.int32)
        base = np.zeros((p.topic_groups, p.dim), dtype=np.float32)
        for gid in range(p.topic_groups):
            base[gid] = self._rand_unit(p.dim)

        tid = 0
        for gid in range(p.topic_groups):
            for _ in range(p.topics_per_group):
                noise = self._rand_unit(p.dim)
                vec = unit((1.0 - p.topic_variant_mix) * base[gid] + p.topic_variant_mix * noise)
                vecs[tid] = vec.astype(np.float32)
                groups[tid] = gid
                tid += 1
        return vecs, groups

    def initial_topic(self) -> int:
        return int(self.rng.integers(0, self.num_topics))

    def next_topic(self, current_topic: int) -> int:
        if self.rng.random() >= self.p.switch_prob:
            return current_topic

        cur_group = int(self.topic_groups[current_topic])
        if self.rng.random() < self.p.near_switch_prob:
            candidates = [t for t in self.group_to_topics[cur_group] if t != current_topic]
            if candidates:
                return int(candidates[int(self.rng.integers(0, len(candidates)))])

        nxt = int(self.rng.integers(0, self.num_topics))
        if nxt == current_topic:
            nxt = (nxt + 1) % self.num_topics
        return nxt

    def turn_vector(self, topic_id: int) -> np.ndarray:
        base = self.topic_vectors[topic_id]
        drift = min(0.8, max(0.0, float(self.p.topic_flow_drift)))
        anchor_mix = min(0.8, max(0.0, float(self.p.topic_flow_anchor_mix)))
        switch_jolt = min(0.9, max(0.0, float(self.p.topic_flow_switch_jolt)))
        noise_mix = min(0.95, max(0.01, float(self.p.turn_noise_mix)))

        if self._active_topic != topic_id or self._flow_anchor is None:
            self._active_topic = topic_id
            j = self._rand_unit(self.p.dim)
            self._flow_anchor = unit((1.0 - switch_jolt) * base + switch_jolt * j)
        else:
            d = self._rand_unit(self.p.dim)
            self._flow_anchor = unit((1.0 - drift) * self._flow_anchor + drift * d)

        noise = self._rand_unit(self.p.dim)
        base_w = max(0.01, 1.0 - noise_mix - anchor_mix)
        vec = unit(base_w * base + anchor_mix * self._flow_anchor + noise_mix * noise)
        return vec.astype(np.float32)

    def query_vector(self, topic_id: int, noise_mix: float) -> np.ndarray:
        noise = self._rand_unit(self.p.dim)
        nm = min(0.95, max(0.01, float(noise_mix)))
        vec = unit((1.0 - nm) * self.topic_vectors[topic_id] + nm * noise)
        return vec.astype(np.float32)

    def topic_similarity(self, a: int, b: int) -> float:
        return float(self.topic_sim[a, b])


class AgentMemorySim:
    def __init__(self, p: SimParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.topic = TopicStream(p, self.rng)

        self.max_memories = p.turns
        self.vec_pool = np.zeros((self.max_memories, p.dim), dtype=np.float32)
        self.heat_pool = np.zeros(self.max_memories, dtype=np.float64)
        self.cluster_of = np.full(self.max_memories, -1, dtype=np.int32)
        self.mem_turns: List[List[int]] = []
        self.mem_count = 0

        self.clusters: List[ClusterState] = []
        self.cluster_centroids = np.zeros((self.max_memories, p.dim), dtype=np.float32)
        self.super_members: List[List[int]] = []
        self.super_centroids = np.zeros((self.max_memories, p.dim), dtype=np.float32)
        self.super_of_cluster = np.full(self.max_memories, -1, dtype=np.int32)
        self.super_indexed_clusters = 0
        self.super_last_rebuild_clusters = 0
        self.super_rebuild_count = 0
        self.cluster_route_score = np.zeros(self.max_memories, dtype=np.float32)
        self.cluster_route_seen = np.zeros(self.max_memories, dtype=np.float32)

        self.turn_topics: List[int] = []
        self.turns_by_topic: Dict[int, List[int]] = {}
        self.topic_to_mem: Dict[int, set] = {}
        self.mem_topic_counts: List[Dict[int, int]] = []
        self.mem_useful_score = np.zeros(self.max_memories, dtype=np.float32)
        self.mem_redundant_score = np.zeros(self.max_memories, dtype=np.float32)
        self.total_heat = 0.0
        self.last_switch_turn = 1
        self.prev_topic_for_shift: Optional[int] = None
        self.last_turn_topic: Optional[int] = None
        self.prev_turn_vec: Optional[np.ndarray] = None
        self.same_topic_streak = 0
        self.streak_sim_sum = 0.0
        self.streak_sim_count = 0
        self.topic_cache_topic: Optional[int] = None
        self.topic_cache_mem: set = set()

        self.new_count = 0
        self.merge_count = 0
        self.normalize_events = 0
        self.query_turns = 0
        self.sim_ops_add_total = 0
        self.sim_ops_query_total = 0
        self.turn_sim_ops: List[int] = []

        self.returned_turns_sum = 0
        self.eval_count = 0

        self.target_precision_sum = 0.0
        self.target_recall_recent_sum = 0.0
        self.target_recall_all_sum = 0.0
        self.target_hit_sum = 0.0
        self.target_recent_hit_sum = 0.0
        self.target_mrr_sum = 0.0
        self.empty_query_count = 0
        self.empty_target_query_count = 0

        self.cold_rescue_queue: List[Tuple[int, int]] = []
        self.cold_rescue_pending = set()
        self.cold_rescue_enqueued = 0
        self.cold_rescue_executed = 0
        self.topic_lift_attempted = 0
        self.topic_lift_executed = 0
        self.topic_cache_unload_count = 0
        self.topic_cache_selected_turns_total = 0
        self.learned_min_gate = float(
            min(
                self.p.learning_min_sim_gate_start * 0.78,
                self.p.min_sim_gate * 0.62,
            )
        )
        self.online_merge_limit = float(self.p.merge_limit)
        self.refinement_events = 0
        self.persistent_explore_events = 0
        self.persistent_explore_cluster_probes = 0
        self.persistent_explore_turn_hits = 0

        self.snapshots: List[Dict[str, float]] = []

    def _refresh_hot_flag(self, cid: int) -> None:
        c = self.clusters[cid]
        total = c.hot_count + c.cold_count
        ratio = (c.hot_count / total) if total > 0 else 0.0
        c.is_hot_cluster = ratio >= self.p.hot_cluster_ratio

    def _set_heat(self, mem_idx: int, new_heat: float) -> None:
        new_heat = max(0.0, float(new_heat))
        old_heat = float(self.heat_pool[mem_idx])
        if old_heat == new_heat:
            return

        self.heat_pool[mem_idx] = new_heat
        self.total_heat += new_heat - old_heat

        old_hot = old_heat > 0.0
        new_hot = new_heat > 0.0
        if old_hot == new_hot:
            return

        cid = int(self.cluster_of[mem_idx])
        if cid < 0 or cid >= len(self.clusters):
            return

        clu = self.clusters[cid]
        if new_hot:
            clu.hot_count += 1
            clu.cold_count = max(0, clu.cold_count - 1)
        else:
            clu.cold_count += 1
            clu.hot_count = max(0, clu.hot_count - 1)
        self._refresh_hot_flag(cid)

    def _use_superclusters(self) -> bool:
        return bool(
            self.p.hierarchical_cluster_enabled
            and len(self.clusters) >= max(1, int(self.p.supercluster_min_clusters))
        )

    def _effective_super_topn(self, base: int) -> int:
        snum = len(self.super_members)
        if snum <= 0:
            return 0
        base_topn = max(1, int(base))
        cnum = max(1, len(self.clusters))
        min_c = max(2, int(self.p.supercluster_min_clusters))
        ratio = max(1.0, float(cnum) / float(min_c))
        dyn = 1.0 + max(0.0, float(self.p.supercluster_topn_scale)) * math.log2(ratio)
        topn = int(math.ceil(base_topn * dyn))
        return max(1, min(snum, topn))

    def _rebuild_superclusters(self) -> None:
        cnum = len(self.clusters)
        self.super_members = []
        self.super_of_cluster[: max(0, cnum)] = -1
        if cnum <= 0:
            self.super_indexed_clusters = 0
            self.super_last_rebuild_clusters = 0
            return

        target_size = max(8, int(self.p.supercluster_target_size))
        snum = max(1, int(math.ceil(cnum / target_size)))
        cmat = self.cluster_centroids[:cnum]

        if snum == 1:
            members = [list(range(cnum))]
        else:
            seed_ids = np.linspace(0, cnum - 1, num=snum, dtype=np.int32)
            seed_mat = cmat[seed_ids]
            sims = cmat @ seed_mat.T
            assign = np.argmax(sims, axis=1).astype(np.int32)
            bins: List[List[int]] = [[] for _ in range(snum)]
            for cid, sid in enumerate(assign.tolist()):
                bins[int(sid)].append(cid)
            members = [lst for lst in bins if lst]
            if not members:
                members = [list(range(cnum))]

        self.super_members = members
        for sid, lst in enumerate(self.super_members):
            idx = np.asarray(lst, dtype=np.int32)
            self.super_of_cluster[idx] = sid
            cen = np.mean(cmat[idx], axis=0).astype(np.float32)
            self.super_centroids[sid] = unit(cen)

        self.super_indexed_clusters = cnum
        self.super_last_rebuild_clusters = cnum
        self.super_rebuild_count += 1

    def _attach_cluster_to_super(self, cid: int) -> int:
        cvec = self.cluster_centroids[cid]
        snum = len(self.super_members)
        if snum <= 0:
            self.super_members = [[cid]]
            self.super_centroids[0] = cvec
            self.super_of_cluster[cid] = 0
            return 0

        sims = self.super_centroids[:snum] @ cvec
        best = int(np.argmax(sims))
        best_sim = float(sims[best])
        max_size = max(
            8,
            int(
                math.ceil(
                    float(self.p.supercluster_target_size)
                    * max(1.0, float(self.p.supercluster_max_size_mult))
                )
            ),
        )

        if best_sim < float(self.p.supercluster_sim) or len(self.super_members[best]) >= max_size:
            sid = snum
            self.super_members.append([cid])
            self.super_centroids[sid] = cvec
            self.super_of_cluster[cid] = sid
            return sid

        members = self.super_members[best]
        n = len(members)
        members.append(cid)
        self.super_of_cluster[cid] = best
        prev = self.super_centroids[best]
        self.super_centroids[best] = unit(((prev * n) + cvec) / max(1, n + 1))
        return best

    def _ensure_supercluster_index(self) -> bool:
        if not self._use_superclusters():
            return False
        cnum = len(self.clusters)
        if cnum <= 0:
            return False
        if not self.super_members:
            self._rebuild_superclusters()
            return True

        if self.super_indexed_clusters < cnum:
            for cid in range(self.super_indexed_clusters, cnum):
                self._attach_cluster_to_super(cid)
            self.super_indexed_clusters = cnum

        every = max(0, int(self.p.supercluster_rebuild_every))
        if every > 0 and (cnum - self.super_last_rebuild_clusters) >= every:
            self._rebuild_superclusters()
        return True

    def _super_candidate_clusters(self, vec: np.ndarray, base_topn: int) -> Tuple[np.ndarray, int]:
        cnum = len(self.clusters)
        if cnum == 0:
            return np.asarray([], dtype=np.int32), 0
        if not self._ensure_supercluster_index():
            return np.arange(cnum, dtype=np.int32), 0

        snum = len(self.super_members)
        if snum <= 0:
            return np.arange(cnum, dtype=np.int32), 0

        sims = self.super_centroids[:snum] @ vec
        ops = snum
        k = self._effective_super_topn(base_topn)
        if k <= 0:
            return np.arange(cnum, dtype=np.int32), ops
        if k == snum:
            sid_order = np.argsort(sims)[::-1]
        else:
            keep = np.argpartition(sims, -k)[-k:]
            sid_order = keep[np.argsort(sims[keep])[::-1]]

        cand: List[int] = []
        for sid in sid_order.tolist():
            cand.extend(self.super_members[int(sid)])
        if not cand:
            return np.arange(cnum, dtype=np.int32), ops
        return np.asarray(cand, dtype=np.int32), ops

    def _find_best_cluster(
        self,
        vec: np.ndarray,
        super_topn: Optional[int] = None,
    ) -> Tuple[Optional[int], float, int]:
        cnum = len(self.clusters)
        if cnum == 0:
            return None, -1.0, 0
        topn = self.p.supercluster_topn_add if super_topn is None else int(super_topn)
        cand, ops0 = self._super_candidate_clusters(vec, topn)
        if cand.size == 0:
            return None, -1.0, ops0
        sims = self.cluster_centroids[cand] @ vec
        i = int(np.argmax(sims))
        return int(cand[i]), float(sims[i]), ops0 + int(cand.size)

    def _find_sim_in_indices(
        self,
        vec: np.ndarray,
        indices: np.ndarray,
        max_results: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], int]:
        if indices.size == 0:
            return [], 0
        sims = self.vec_pool[indices] @ vec
        ops = int(indices.size)
        if max_results is not None and max_results > 0 and sims.size > max_results:
            keep = np.argpartition(sims, -max_results)[-max_results:]
            order = keep[np.argsort(sims[keep])[::-1]]
        else:
            order = np.argsort(sims)[::-1]
        out = [(int(indices[i]), float(sims[i])) for i in order]
        return out, ops

    def _find_sim_in_cluster(
        self,
        vec: np.ndarray,
        cid: int,
        only_hot: bool = False,
        only_cold: bool = False,
        max_results: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], int]:
        if cid < 0 or cid >= len(self.clusters):
            return [], 0
        members = self.clusters[cid].members
        if not members:
            return [], 0
        idx = np.asarray(members, dtype=np.int32)
        if only_hot and only_cold:
            return [], 0
        if only_hot:
            idx = idx[self.heat_pool[idx] > 0.0]
        elif only_cold:
            idx = idx[self.heat_pool[idx] <= 0.0]
        return self._find_sim_in_indices(vec, idx, max_results=max_results)

    def _find_sim_all_hot(
        self,
        vec: np.ndarray,
        max_results: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], int]:
        hot_idx = np.flatnonzero(self.heat_pool[: self.mem_count] > 0.0)
        return self._find_sim_in_indices(vec, hot_idx, max_results=max_results)

    def _find_sim_all_cold(
        self,
        vec: np.ndarray,
        max_results: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], int]:
        cold_idx = np.flatnonzero(self.heat_pool[: self.mem_count] <= 0.0)
        return self._find_sim_in_indices(vec, cold_idx, max_results=max_results)

    def _normalize_heat_if_needed(self) -> None:
        if not self.p.softmax:
            return
        if self.total_heat <= self.p.total_heat + self.p.tolerance:
            return

        hot_idx = np.flatnonzero(self.heat_pool[: self.mem_count] > 0.0)
        if hot_idx.size == 0:
            return

        old = self.heat_pool[hot_idx].astype(np.float64, copy=True)
        debt = self.total_heat - float(self.p.total_heat)
        hs = np.sort(old)

        deduction = 0.0
        for i, h in enumerate(hs, start=1):
            remaining = hs.size - (i - 1)
            share = debt / remaining
            if h < share:
                debt -= h
            else:
                deduction = share
                break

        new = np.floor(np.maximum(old - deduction, 0.0))
        self.heat_pool[hot_idx] = new
        self.total_heat += float(np.sum(new - old))

        became_cold = hot_idx[(old > 0.0) & (new <= 0.0)]
        for mem_idx in became_cold:
            cid = int(self.cluster_of[mem_idx])
            if cid < 0 or cid >= len(self.clusters):
                continue
            clu = self.clusters[cid]
            clu.cold_count += 1
            clu.hot_count = max(0, clu.hot_count - 1)
            self._refresh_hot_flag(cid)

        self.normalize_events += 1

    def _attach_cluster(self, vec: np.ndarray, mem_idx: int) -> int:
        best_id, best_sim, ops = self._find_best_cluster(vec, super_topn=self.p.supercluster_topn_add)
        self.sim_ops_add_total += ops

        if best_id is not None and best_sim >= self.p.cluster_sim:
            cid = best_id
            clu = self.clusters[cid]
            clu.members.append(mem_idx)
            clu.cold_count += 1
            self._refresh_hot_flag(cid)
        else:
            cid = len(self.clusters)
            self.cluster_centroids[cid] = vec
            clu = ClusterState(members=[mem_idx], hot_count=0, cold_count=1, is_hot_cluster=False)
            self.clusters.append(clu)
            self._refresh_hot_flag(cid)

        self.cluster_of[mem_idx] = cid
        return cid

    def _add_new_with_cluster_cap(self, new_idx: int) -> None:
        cid = int(self.cluster_of[new_idx])
        if cid < 0 or cid >= len(self.clusters):
            self._set_heat(new_idx, float(self.p.new_memory_heat))
            self._normalize_heat_if_needed()
            return

        clu = self.clusters[cid]
        members = clu.members
        num_old = len(members) - 1
        if num_old <= 0:
            self._set_heat(new_idx, float(self.p.new_memory_heat))
            self._normalize_heat_if_needed()
            return

        cap = float(self.p.cluster_heat_cap)
        base_share = 0.46
        decay = 0.0092
        new_share = base_share * math.exp(-decay * num_old)
        new_share = max(0.085, min(0.46, new_share))

        target_new = min(math.floor(cap * new_share), int(self.p.new_memory_heat))
        cap_div = cap / 6.2

        old_total = 0.0
        for idx in members:
            if idx == new_idx:
                continue
            h = float(self.heat_pool[idx])
            if h > 0.0:
                old_total += math.floor(cap_div * math.tanh(h / cap_div))

        total_after = old_total + float(target_new)
        scale = (cap / total_after) if total_after > cap and total_after > 0 else 1.0

        for idx in members:
            if idx == new_idx:
                continue
            h = float(self.heat_pool[idx])
            if h > 0.0:
                compressed = math.floor(cap_div * math.tanh(h / cap_div))
                final_h = math.floor(compressed * scale)
                self._set_heat(idx, float(final_h))

        final_new = math.floor(target_new * scale)
        self._set_heat(new_idx, float(final_new))
        self._normalize_heat_if_needed()

    def _neighbors_add_heat(self, vec: np.ndarray, total_turn: int, target_idx: int) -> None:
        cid = int(self.cluster_of[target_idx])
        is_cold_cluster = cid >= 0 and cid < len(self.clusters) and (not self.clusters[cid].is_hot_cluster)
        if is_cold_cluster and self.heat_pool[target_idx] <= 0:
            return

        if cid >= 0:
            sim_results, ops = self._find_sim_in_cluster(vec, cid, only_hot=True, max_results=30)
        else:
            sim_results, ops = self._find_sim_all_hot(vec, max_results=30)
        self.sim_ops_add_total += ops

        if not sim_results:
            return

        weighted: List[Tuple[int, float]] = []
        for idx, sim in sim_results:
            age = max(0, total_turn - (idx + 1))
            if age < self.p.loss_turn:
                w = self.p.time_boost * (1.0 - age / self.p.loss_turn)
            else:
                w = 0.0
            weighted.append((idx, sim + w))
        weighted.sort(key=lambda it: it[1], reverse=True)

        neighbors: List[int] = []
        for idx, _ in weighted:
            if idx == target_idx:
                continue
            neighbors.append(idx)
            if len(neighbors) >= self.p.max_neighbors:
                break

        if not neighbors:
            return

        nbheat = int(self.p.neighbors_heat)
        if is_cold_cluster:
            nbheat = math.floor(nbheat * self.p.cold_neighbor_multiplier)

        per = nbheat // len(neighbors)
        extra = nbheat - per * len(neighbors)

        for idx in neighbors:
            self._set_heat(idx, self.heat_pool[idx] + per)
        self._set_heat(neighbors[0], self.heat_pool[neighbors[0]] + extra)
        self._normalize_heat_if_needed()

    def add_memory(self, vec: np.ndarray, turn: int) -> None:
        best_id, best_sim, ops = self._find_best_cluster(vec, super_topn=self.p.supercluster_topn_add)
        self.sim_ops_add_total += ops

        if best_id is not None and best_sim >= self.p.cluster_sim:
            sim_results, ops2 = self._find_sim_in_cluster(
                vec, best_id, only_hot=True, max_results=self.p.fast_scan_topk
            )
            self.sim_ops_add_total += ops2
            if not sim_results:
                fallback, ops3 = self._find_sim_all_hot(vec, max_results=self.p.fast_scan_topk)
                self.sim_ops_add_total += ops3
                sim_results = fallback
        else:
            sim_results, ops2 = self._find_sim_all_hot(vec, max_results=self.p.fast_scan_topk)
            self.sim_ops_add_total += ops2

        if sim_results and sim_results[0][1] >= self.online_merge_limit:
            target = sim_results[0][0]
            self.mem_turns[target].append(turn)
            topic_id = self.turn_topics[turn - 1] if 0 < turn <= len(self.turn_topics) else None
            if topic_id is not None:
                self._bind_memory_topic(target, int(topic_id), amount=1)
            self._set_heat(target, self.heat_pool[target] + self.p.new_memory_heat)
            self._normalize_heat_if_needed()
            self.merge_count += 1
            return

        if self.mem_count >= self.max_memories:
            return

        new_idx = self.mem_count
        self.vec_pool[new_idx] = vec
        self.heat_pool[new_idx] = 0.0
        self.cluster_of[new_idx] = -1
        self.mem_turns.append([turn])
        self.mem_topic_counts.append({})
        self.mem_count += 1
        topic_id = self.turn_topics[turn - 1] if 0 < turn <= len(self.turn_topics) else None
        if topic_id is not None:
            self._bind_memory_topic(new_idx, int(topic_id), amount=1)

        self._attach_cluster(vec, new_idx)
        self._add_new_with_cluster_cap(new_idx)
        self._neighbors_add_heat(vec, turn, new_idx)
        self.new_count += 1

    def _register_turn_topic(self, turn: int, topic_id: int) -> None:
        self.turn_topics.append(topic_id)
        bucket = self.turns_by_topic.get(topic_id)
        if bucket is None:
            bucket = []
            self.turns_by_topic[topic_id] = bucket
        bucket.append(turn)

    def _bind_memory_topic(self, mem_idx: int, topic_id: int, amount: int = 1) -> None:
        if mem_idx < 0:
            return
        while len(self.mem_topic_counts) <= mem_idx:
            self.mem_topic_counts.append({})
        d = self.mem_topic_counts[mem_idx]
        d[topic_id] = d.get(topic_id, 0) + max(1, int(amount))

        bucket = self.topic_to_mem.get(topic_id)
        if bucket is None:
            bucket = set()
            self.topic_to_mem[topic_id] = bucket
        bucket.add(mem_idx)

    def _update_topic_stability(self, current_topic: int, turn_vec: np.ndarray) -> Tuple[bool, float]:
        if self.last_turn_topic == current_topic:
            self.same_topic_streak += 1
            if self.prev_turn_vec is not None:
                sim = float(np.dot(self.prev_turn_vec, turn_vec))
                self.streak_sim_sum += sim
                self.streak_sim_count += 1
        else:
            self.same_topic_streak = 1
            self.streak_sim_sum = 0.0
            self.streak_sim_count = 0

        self.last_turn_topic = current_topic
        self.prev_turn_vec = turn_vec
        avg_pair = (self.streak_sim_sum / self.streak_sim_count) if self.streak_sim_count > 0 else 1.0
        stable = (
            self.same_topic_streak >= self.p.stable_warmup_turns
            and avg_pair >= self.p.stable_min_pair_sim
        )
        return stable, avg_pair

    def _unload_topic_cache(self) -> None:
        if self.topic_cache_mem:
            self.topic_cache_unload_count += 1
        self.topic_cache_mem = set()
        self.topic_cache_topic = None

    def _topic_random_lift(self, turn: int, current_topic: int, stable_ready: bool) -> None:
        if not stable_ready:
            return
        if self.p.topic_random_lift_interval > 1 and turn % self.p.topic_random_lift_interval != 0:
            return
        if self.rng.random() > self.p.topic_random_lift_prob:
            return

        mem_set = self.topic_to_mem.get(current_topic)
        if not mem_set:
            return
        self.topic_lift_attempted += 1

        candidates = list(mem_set)
        if not candidates:
            return

        self.rng.shuffle(candidates)
        picked: List[int] = []
        for idx in candidates:
            if idx >= self.mem_count:
                continue
            if self.p.topic_random_lift_only_cold and self.heat_pool[idx] > 0.0:
                continue
            picked.append(idx)
            if len(picked) >= self.p.topic_random_lift_count:
                break

        if not picked:
            return

        # Topic preload cache: loaded memories bypass hot competition.
        self.topic_cache_topic = current_topic
        self.topic_cache_mem = set(picked)
        self.topic_lift_executed += len(picked)

    def _sample_query_topic(self, current_topic: int, current_turn: int) -> int:
        if not self.turn_topics:
            return current_topic
        if self.rng.random() < self.p.query_current_intent_prob:
            return current_topic

        max_old_turn = current_turn - self.p.query_long_term_min_age
        if max_old_turn > 1:
            old_turn = int(self.rng.integers(1, max_old_turn))
            return self.turn_topics[old_turn - 1]

        past_turn = int(self.rng.integers(1, current_turn))
        return self.turn_topics[past_turn - 1]

    def _in_shift_window(self, turn: int) -> bool:
        if self.p.shift_probe_turns <= 0:
            return False
        return (turn - self.last_switch_turn) <= self.p.shift_probe_turns

    @staticmethod
    def _lerp(a: float, b: float, t: float) -> float:
        return float(a + (b - a) * t)

    def _learning_progress(self, turn: int) -> float:
        if not self.p.learning_curve_enabled:
            return 1.0
        warm = max(0, int(self.p.learning_warmup_turns))
        full = max(warm + 1, int(self.p.learning_full_turns))
        if turn <= warm:
            return 0.0
        if turn >= full:
            return 1.0
        return float((turn - warm) / float(full - warm))

    def _refinement_progress(self, turn: int) -> float:
        if not self.p.refinement_enabled:
            return 0.0
        start = max(0, int(self.p.refinement_start_turn))
        if turn <= start:
            return 0.0
        end = max(start + 1, int(self.p.learning_full_turns))
        if turn >= end:
            return 1.0
        return float((turn - start) / float(end - start))

    def _effective_probe_clusters(self, turn: int) -> int:
        start_k = max(1, int(self.p.refinement_probe_clusters_start))
        end_k = max(1, int(self.p.refinement_probe_clusters_end))
        if not self.p.refinement_enabled:
            return end_k
        prog = self._refinement_progress(turn)
        val = int(round(self._lerp(start_k, end_k, prog)))
        return max(1, val)

    def _dominant_memory_topic(self, mem_idx: int) -> Optional[int]:
        if mem_idx < 0 or mem_idx >= len(self.mem_topic_counts):
            return None
        counts = self.mem_topic_counts[mem_idx]
        if not counts:
            return None
        best_topic = None
        best_n = -1
        for tid, n in counts.items():
            if n > best_n:
                best_topic = int(tid)
                best_n = int(n)
        return best_topic

    def _top_probe_clusters(
        self,
        vec: np.ndarray,
        probe_clusters: int,
        super_topn: int,
        turn: int,
    ) -> Tuple[List[int], int]:
        cnum = len(self.clusters)
        if cnum <= 0:
            return [], 0
        cand, ops0 = self._super_candidate_clusters(vec, super_topn)
        if cand.size <= 0:
            return [], ops0

        sims = self.cluster_centroids[cand] @ vec
        ops = ops0 + int(cand.size)

        route_scale = 0.0
        if self.p.refinement_enabled and turn >= self.p.refinement_start_turn:
            route_scale = float(self.p.refinement_route_bias_scale) * self._refinement_progress(turn)
        if route_scale > 0.0:
            route_bias = self.cluster_route_score[cand].astype(np.float64)
            adjusted = sims.astype(np.float64) + route_scale * route_bias
        else:
            adjusted = sims.astype(np.float64)

        k = max(1, min(int(probe_clusters), int(cand.size)))
        if k >= cand.size:
            order = np.argsort(adjusted)[::-1]
        else:
            keep = np.argpartition(adjusted, -k)[-k:]
            order = keep[np.argsort(adjusted[keep])[::-1]]
        out = [int(cand[i]) for i in order.tolist()]

        # Early-stage exploration: intentionally inject random clusters so precision starts lower.
        if self.p.refinement_enabled:
            rp = self._refinement_progress(turn)
            explore_n = int(round((1.0 - rp) * k * 0.75))
            if explore_n > 0 and k < cand.size:
                picked = set(out)
                remain = [int(c) for c in cand.tolist() if int(c) not in picked]
                if remain:
                    self.rng.shuffle(remain)
                    out.extend(remain[: min(explore_n, len(remain))])
        return out, ops

    def _effective_retrieval_knobs(self, turn: int) -> Dict[str, float]:
        prog = self._learning_progress(turn)
        min_gate = self._lerp(self.p.learning_min_sim_gate_start, self.p.min_sim_gate, prog)
        power = self._lerp(self.p.learning_power_suppress_start, self.p.power_suppress, prog)
        cross = self._lerp(self.p.learning_topic_cross_quota_start, self.p.topic_cross_quota_ratio, prog)
        kw_weight = self._lerp(self.p.learning_keyword_weight_start, self.p.keyword_weight, prog)
        max_memory = int(round(self._lerp(self.p.learning_max_memory_start, self.p.max_memory, prog)))
        max_turns = int(round(self._lerp(self.p.learning_max_turns_start, self.p.max_turns, prog)))
        super_topn_q = int(
            round(self._lerp(self.p.learning_super_topn_query_start, self.p.supercluster_topn_query, prog))
        )
        if self.p.refinement_enabled:
            min_gate = self.learned_min_gate
        probe_clusters = self._effective_probe_clusters(turn)
        return {
            "progress": prog,
            "min_sim_gate": min_gate,
            "power_suppress": max(1.0, power),
            "topic_cross_quota_ratio": min(0.5, max(0.0, cross)),
            "keyword_weight": max(0.0, kw_weight),
            "max_memory": max(1, max_memory),
            "max_turns": max(1, max_turns),
            "supercluster_topn_query": max(1, super_topn_q),
            "probe_clusters": max(1, probe_clusters),
        }

    def _apply_refinement(
        self,
        turn: int,
        target_topic: int,
        hits_all: int,
        candidate_samples: Sequence[Tuple[int, int, float, float]],
        selected_memories: Sequence[int],
        evidence_memories: Sequence[int],
    ) -> None:
        if not self.p.refinement_enabled:
            return
        if turn < max(0, int(self.p.refinement_start_turn)):
            return
        if not candidate_samples:
            return

        refine_prog = self._refinement_progress(turn)
        route_lr = max(0.0, float(self.p.refinement_route_lr)) * (0.18 + 0.82 * refine_prog)
        gate_lr = max(0.0, float(self.p.refinement_gate_lr)) * (refine_prog ** 1.6)
        merge_lr = max(0.0, float(self.p.refinement_merge_lr)) * (0.12 + 0.88 * refine_prog)

        evidence_set = set(int(i) for i in evidence_memories)
        selected_set = set(int(i) for i in selected_memories)
        pos_sims: List[float] = []
        neg_sims: List[float] = []
        clu_pos: Dict[int, float] = {}
        clu_neg: Dict[int, float] = {}
        sampled = 0

        for mem_idx, cid, sim, eff in candidate_samples:
            if mem_idx < 0 or mem_idx >= self.mem_count:
                continue
            if cid < 0 or cid >= len(self.clusters):
                continue
            sampled += 1
            score = max(0.0, float(eff))
            simf = float(sim)
            is_pos = mem_idx in evidence_set
            if not is_pos:
                topic_count = self.mem_topic_counts[mem_idx].get(target_topic, 0)
                is_pos = topic_count > 0 and mem_idx in selected_set and simf >= self.learned_min_gate

            is_neg = False
            if not is_pos:
                dom_topic = self._dominant_memory_topic(mem_idx)
                if dom_topic is not None:
                    same_band = self.topic.topic_similarity(dom_topic, target_topic) >= self.p.topic_sim_threshold
                    if same_band and (mem_idx in selected_set or simf >= self.learned_min_gate * 0.92):
                        is_neg = True

            if is_pos:
                pos_sims.append(simf)
                clu_pos[cid] = clu_pos.get(cid, 0.0) + max(1e-6, score)
                old_u = float(self.mem_useful_score[mem_idx])
                old_r = float(self.mem_redundant_score[mem_idx])
                self.mem_useful_score[mem_idx] = (1.0 - route_lr) * old_u + route_lr
                self.mem_redundant_score[mem_idx] = (1.0 - 0.5 * route_lr) * old_r
            elif is_neg:
                neg_sims.append(simf)
                clu_neg[cid] = clu_neg.get(cid, 0.0) + max(1e-6, score)
                old_r = float(self.mem_redundant_score[mem_idx])
                self.mem_redundant_score[mem_idx] = (1.0 - route_lr) * old_r + route_lr

        if sampled <= 0:
            return

        route_decay = 1.0 - min(0.12, route_lr * 0.22)
        for cid in set(list(clu_pos.keys()) + list(clu_neg.keys())):
            p_w = clu_pos.get(cid, 0.0)
            n_w = clu_neg.get(cid, 0.0)
            total = p_w + n_w
            if total <= 0.0:
                continue
            signal = (p_w - n_w) / total
            prev = float(self.cluster_route_score[cid])
            nxt = prev * route_decay + route_lr * signal
            self.cluster_route_score[cid] = float(np.clip(nxt, -2.0, 2.0))
            self.cluster_route_seen[cid] += 1.0

        gate_target = self.learned_min_gate
        if pos_sims and neg_sims:
            pos_floor = float(np.quantile(np.asarray(pos_sims, dtype=np.float32), 0.20))
            neg_ceil = float(np.quantile(np.asarray(neg_sims, dtype=np.float32), 0.80))
            gate_target = 0.5 * (pos_floor + neg_ceil)
        elif neg_sims:
            gate_target = self.learned_min_gate + (0.035 if hits_all <= 0 else 0.018)
        elif pos_sims:
            gate_target = self.learned_min_gate - (0.020 if hits_all <= 0 else 0.008)

        gate_lo = max(0.05, min(self.p.learning_min_sim_gate_start, self.p.min_sim_gate) - 0.22)
        gate_hi = min(0.90, self.p.min_sim_gate + 0.08)
        gate_target = float(np.clip(gate_target, gate_lo, gate_hi))
        self.learned_min_gate = float(self.learned_min_gate + gate_lr * (gate_target - self.learned_min_gate))

        pos_n = len(pos_sims)
        neg_n = len(neg_sims)
        neg_ratio = float(neg_n / max(1, pos_n + neg_n))
        merge_target = self.p.merge_limit
        if neg_ratio >= 0.66:
            merge_target = self.p.merge_limit - 0.055
        elif neg_ratio >= 0.52:
            merge_target = self.p.merge_limit - 0.030
        elif hits_all <= 0 and pos_n <= 0:
            merge_target = self.p.merge_limit + 0.020
        elif pos_n > neg_n * 1.4:
            merge_target = self.p.merge_limit + 0.010

        merge_lo = max(0.50, self.p.merge_limit - 0.10)
        merge_hi = min(0.995, self.p.merge_limit + 0.03)
        merge_target = float(np.clip(merge_target, merge_lo, merge_hi))
        self.online_merge_limit = float(
            self.online_merge_limit + merge_lr * (merge_target - self.online_merge_limit)
        )
        self.refinement_events += 1

    def _enqueue_cold_rescue(self, query_vec: np.ndarray, target_topic: int, current_turn: int) -> None:
        if self.mem_count <= 0:
            return
        if len(self.cold_rescue_queue) >= self.p.cold_rescue_max_queue:
            return

        best_id, best_sim, ops = self._find_best_cluster(query_vec, super_topn=self.p.supercluster_topn_query)
        self.sim_ops_query_total += ops
        if best_id is not None and best_sim >= self.p.cluster_sim:
            candidates, ops2 = self._find_sim_in_cluster(
                query_vec,
                best_id,
                only_cold=True,
                max_results=max(self.p.cold_rescue_topn * 6, 18),
            )
        else:
            candidates, ops2 = self._find_sim_all_cold(
                query_vec,
                max_results=max(self.p.cold_rescue_topn * 6, 18),
            )
        self.sim_ops_query_total += ops2
        if not candidates:
            return

        chosen = 0
        gate = float(self._effective_retrieval_knobs(current_turn)["min_sim_gate"])
        for mem_idx, sim in candidates:
            if sim < gate:
                continue

            has_target_turn = False
            for t in self.mem_turns[mem_idx]:
                if self.turn_topics[t - 1] == target_topic:
                    has_target_turn = True
                    break
            if not has_target_turn:
                continue

            if mem_idx in self.cold_rescue_pending:
                continue
            delay = int(self.rng.integers(self.p.cold_rescue_delay_min, self.p.cold_rescue_delay_max + 1))
            due_turn = current_turn + delay
            heapq.heappush(self.cold_rescue_queue, (due_turn, mem_idx))
            self.cold_rescue_pending.add(mem_idx)
            self.cold_rescue_enqueued += 1
            chosen += 1
            if chosen >= self.p.cold_rescue_topn:
                break

    def _process_cold_rescue(self, turn: int) -> None:
        if not self.cold_rescue_queue:
            return
        if self.p.maintenance_task > 0 and turn % self.p.maintenance_task != 0:
            return

        done = 0
        while self.cold_rescue_queue and done < self.p.cold_rescue_batch:
            due_turn, mem_idx = self.cold_rescue_queue[0]
            if due_turn > turn:
                break
            heapq.heappop(self.cold_rescue_queue)
            self.cold_rescue_pending.discard(mem_idx)

            if mem_idx >= self.mem_count:
                continue
            if self.heat_pool[mem_idx] > 0.0:
                continue

            wake_heat = max(
                self.p.new_memory_heat * self.p.cold_wake_multiplier,
                self.p.new_memory_heat,
            )
            self._set_heat(mem_idx, wake_heat)

            old_nb = self.p.neighbors_heat
            self.p.neighbors_heat = max(old_nb, self.p.cold_extra_neighbor_heat)
            self._neighbors_add_heat(self.vec_pool[mem_idx], turn, mem_idx)
            self.p.neighbors_heat = old_nb

            done += 1
            self.cold_rescue_executed += 1

        if done > 0:
            self._normalize_heat_if_needed()

    def _query_vectors(self, primary: np.ndarray, keyword_weight: Optional[float] = None) -> List[Tuple[np.ndarray, float]]:
        out: List[Tuple[np.ndarray, float]] = [(primary, 1.0)]
        kw_weight = self.p.keyword_weight if keyword_weight is None else float(keyword_weight)
        for _ in range(max(0, self.p.keyword_queries)):
            noise = unit(self.rng.normal(size=self.p.dim).astype(np.float32))
            kw = unit((1.0 - self.p.keyword_noise_mix) * primary + self.p.keyword_noise_mix * noise)
            out.append((kw.astype(np.float32), kw_weight))
        return out

    def retrieve(
        self,
        query_vec: np.ndarray,
        current_topic: Optional[int] = None,
        current_turn: Optional[int] = None,
        target_topic: Optional[int] = None,
    ) -> Tuple[List[int], int, Dict[str, object]]:
        if self.mem_count <= 0:
            return [], 0, {"candidate_samples": [], "selected_memories": [], "evidence_memories": []}

        turn_best: Dict[int, float] = {}
        turn_src: Dict[int, str] = {}
        turn_mem: Dict[int, int] = {}
        sim_ops = 0
        query_turn = current_turn if current_turn is not None else self.p.turns
        knobs = self._effective_retrieval_knobs(query_turn)
        max_memory = int(knobs["max_memory"])
        max_turns = int(knobs["max_turns"])
        min_gate = float(knobs["min_sim_gate"])
        power = float(knobs["power_suppress"])
        cross_quota_ratio = float(knobs["topic_cross_quota_ratio"])
        query_topn = int(knobs["supercluster_topn_query"])
        keyword_weight = float(knobs["keyword_weight"])
        probe_clusters = int(knobs["probe_clusters"])
        per_cluster_limit = max(2, int(self.p.refinement_probe_per_cluster_limit))
        base_scan_limit = max(per_cluster_limit, max_memory)
        persistent_cap = max(1, int(self.p.persistent_explore_candidate_cap))

        persistent_extra_budget = 0
        if self.p.persistent_explore_enabled and len(self.clusters) > 1:
            eps = min(1.0, max(0.0, float(self.p.persistent_explore_epsilon)))
            periodic = max(0, int(self.p.persistent_explore_period_turns))
            trigger = False
            if eps > 0.0 and self.rng.random() < eps:
                trigger = True
            if periodic > 0 and (query_turn % periodic == 0):
                trigger = True
            if trigger:
                persistent_extra_budget = max(1, int(self.p.persistent_explore_extra_clusters))
                self.persistent_explore_events += 1

        mem_best_sim: Dict[int, float] = {}
        mem_best_cluster: Dict[int, int] = {}
        mem_best_effective: Dict[int, float] = {}
        persistent_probe_clusters_query = set()
        for q_idx, (qv, weight) in enumerate(self._query_vectors(query_vec, keyword_weight=keyword_weight)):
            cluster_ids, ops = self._top_probe_clusters(
                qv,
                probe_clusters=max(1, probe_clusters),
                super_topn=query_topn,
                turn=query_turn,
            )
            sim_ops += ops
            persistent_probe_clusters_vec = set()
            if q_idx == 0 and persistent_extra_budget > 0:
                picked = set(cluster_ids)
                if len(picked) < len(self.clusters):
                    pool = [cid for cid in range(len(self.clusters)) if cid not in picked]
                    self.rng.shuffle(pool)
                    take = min(persistent_extra_budget, len(pool))
                    extras = pool[:take]
                    if extras:
                        cluster_ids.extend(extras)
                        persistent_probe_clusters_vec.update(extras)
                        persistent_probe_clusters_query.update(extras)
                        self.persistent_explore_cluster_probes += len(extras)

            scanned_any = False
            for cid in cluster_ids:
                if cid < 0 or cid >= len(self.clusters):
                    continue
                scan_limit = base_scan_limit
                if cid in persistent_probe_clusters_vec:
                    scan_limit = min(scan_limit, persistent_cap)
                sim_results, ops2 = self._find_sim_in_cluster(
                    qv,
                    cid,
                    only_hot=True,
                    max_results=scan_limit,
                )
                sim_ops += ops2
                if not sim_results:
                    continue
                scanned_any = True

                for mem_idx, sim in sim_results:
                    sim_pos = max(0.0, sim)
                    prev_sim = mem_best_sim.get(mem_idx)
                    if prev_sim is None or sim > prev_sim:
                        mem_best_sim[mem_idx] = sim
                        mem_best_cluster[mem_idx] = cid
                        mem_best_effective[mem_idx] = (sim_pos ** power) * weight

                    if sim < min_gate:
                        break
                    effective = (sim ** power) * weight
                    for t in self.mem_turns[mem_idx]:
                        prev = turn_best.get(t)
                        if prev is None or effective > prev:
                            turn_best[t] = effective
                            turn_src[t] = "explore" if cid in persistent_probe_clusters_vec else "hot"
                            turn_mem[t] = mem_idx

            if not scanned_any:
                fallback, ops2 = self._find_sim_all_hot(qv, max_results=max_memory * 2)
                sim_ops += ops2
                for mem_idx, sim in fallback:
                    sim_pos = max(0.0, sim)
                    prev_sim = mem_best_sim.get(mem_idx)
                    if prev_sim is None or sim > prev_sim:
                        mem_best_sim[mem_idx] = sim
                        mem_best_cluster[mem_idx] = int(self.cluster_of[mem_idx])
                        mem_best_effective[mem_idx] = (sim_pos ** power) * weight
                    if sim < min_gate:
                        break
                    effective = (sim ** power) * weight
                    for t in self.mem_turns[mem_idx]:
                        prev = turn_best.get(t)
                        if prev is None or effective > prev:
                            turn_best[t] = effective
                            turn_src[t] = "hot"
                            turn_mem[t] = mem_idx

            # Topic cache candidates do not participate in heat competition.
            if current_topic is not None and self.topic_cache_topic == current_topic and self.topic_cache_mem:
                cache_idx = np.asarray(
                    [idx for idx in self.topic_cache_mem if 0 <= idx < self.mem_count],
                    dtype=np.int32,
                )
                if cache_idx.size > 0:
                    cache_sims = self.vec_pool[cache_idx] @ qv
                    sim_ops += int(cache_idx.size)
                    order = np.argsort(cache_sims)[::-1]
                    for oi in order:
                        sim = float(cache_sims[oi])
                        sim_pos = max(0.0, sim)
                        if sim < min_gate:
                            break
                        mem_idx = int(cache_idx[oi])
                        prev_sim = mem_best_sim.get(mem_idx)
                        if prev_sim is None or sim > prev_sim:
                            mem_best_sim[mem_idx] = sim
                            mem_best_cluster[mem_idx] = int(self.cluster_of[mem_idx])
                            mem_best_effective[mem_idx] = (sim_pos ** power) * weight
                        effective = (sim ** power) * weight * self.p.topic_cache_weight
                        for t in self.mem_turns[mem_idx]:
                            prev = turn_best.get(t)
                            if prev is None or effective > prev:
                                turn_best[t] = effective
                                turn_src[t] = "cache"
                                turn_mem[t] = mem_idx

        sample_limit = max(1, int(self.p.refinement_sample_mem_topk))
        if mem_best_sim:
            mem_items = sorted(mem_best_sim.items(), key=lambda it: it[1], reverse=True)
            candidate_samples = [
                (
                    int(mem_idx),
                    int(mem_best_cluster.get(mem_idx, -1)),
                    float(sim),
                    float(mem_best_effective.get(mem_idx, sim)),
                )
                for mem_idx, sim in mem_items[:sample_limit]
            ]
        else:
            candidate_samples = []

        if not turn_best:
            return [], sim_ops, {
                "candidate_samples": candidate_samples,
                "selected_memories": [],
                "evidence_memories": [],
            }

        ranked = sorted(turn_best.items(), key=lambda it: it[1], reverse=True)

        if (not self.p.use_topic_buckets) or current_topic is None:
            selected = ranked[:max_turns]
            if self.p.refinement_enabled:
                rp = self._refinement_progress(query_turn)
                explore_slots = int(round(max_turns * 0.55 * (1.0 - rp)))
                if explore_slots > 0 and len(ranked) > max_turns:
                    core_n = max(1, max_turns - explore_slots)
                    core = ranked[:core_n]
                    pool = ranked[core_n : min(len(ranked), max_turns * 6)]
                    if pool:
                        choose_n = min(explore_slots, len(pool))
                        pick = self.rng.choice(len(pool), size=choose_n, replace=False)
                        explore = [pool[int(i)] for i in np.asarray(pick).tolist()]
                        selected = core + explore
                        selected = selected[:max_turns]
                        selected.sort(key=lambda it: it[1], reverse=True)
            selected_turns = [t for t, _ in selected]
            self.topic_cache_selected_turns_total += sum(1 for t in selected_turns if turn_src.get(t) == "cache")
            selected_memories = [turn_mem[t] for t in selected_turns if t in turn_mem]
            if persistent_probe_clusters_query:
                self.persistent_explore_turn_hits += sum(
                    1
                    for mem_idx in selected_memories
                    if int(self.cluster_of[mem_idx]) in persistent_probe_clusters_query
                )
            else:
                self.persistent_explore_turn_hits += sum(1 for t in selected_turns if turn_src.get(t) == "explore")
            evidence_memories: List[int] = []
            if target_topic is not None:
                evidence_memories = [
                    turn_mem[t]
                    for t in selected_turns
                    if t in turn_mem and self.turn_topics[t - 1] == target_topic
                ]
            return selected_turns, sim_ops, {
                "candidate_samples": candidate_samples,
                "selected_memories": selected_memories,
                "evidence_memories": evidence_memories,
            }

        same: List[Tuple[int, float]] = []
        near: List[Tuple[int, float]] = []
        cross: List[Tuple[int, float]] = []

        for turn, score in ranked:
            topic_id = self.turn_topics[turn - 1]
            if topic_id == current_topic:
                same.append((turn, score))
            elif self.topic.topic_similarity(topic_id, current_topic) >= self.p.topic_sim_threshold:
                near.append((turn, score))
            else:
                cross.append((turn, score))

        reserved_cross = min(len(cross), int(max_turns * cross_quota_ratio))
        in_topic_budget = max_turns - reserved_cross

        selected: List[Tuple[int, float]] = []
        used = set()

        def append_until(src: Sequence[Tuple[int, float]], limit: int) -> None:
            for turn, score in src:
                if len(selected) >= limit:
                    break
                if turn in used:
                    continue
                used.add(turn)
                selected.append((turn, score))

        append_until(same, in_topic_budget)
        append_until(near, in_topic_budget)
        append_until(cross, max_turns)

        if len(selected) < max_turns:
            append_until(near, max_turns)
            append_until(same, max_turns)
            append_until(cross, max_turns)

        selected.sort(key=lambda it: it[1], reverse=True)
        selected = selected[:max_turns]
        selected_turns = [t for t, _ in selected]
        self.topic_cache_selected_turns_total += sum(1 for t in selected_turns if turn_src.get(t) == "cache")
        selected_memories = [turn_mem[t] for t in selected_turns if t in turn_mem]
        if persistent_probe_clusters_query:
            self.persistent_explore_turn_hits += sum(
                1
                for mem_idx in selected_memories
                if int(self.cluster_of[mem_idx]) in persistent_probe_clusters_query
            )
        else:
            self.persistent_explore_turn_hits += sum(1 for t in selected_turns if turn_src.get(t) == "explore")
        evidence_memories = []
        if target_topic is not None:
            evidence_memories = [
                turn_mem[t]
                for t in selected_turns
                if t in turn_mem and self.turn_topics[t - 1] == target_topic
            ]
        return selected_turns, sim_ops, {
            "candidate_samples": candidate_samples,
            "selected_memories": selected_memories,
            "evidence_memories": evidence_memories,
        }

    def _eval_query(self, selected_turns: List[int], current_turn: int, target_topic: int) -> Tuple[int, int]:
        k = len(selected_turns)
        self.returned_turns_sum += k

        all_turns = self.turns_by_topic.get(target_topic, [])
        if not all_turns:
            return 0, 0

        hits_all = 0
        mrr = 0.0
        for rank, t in enumerate(selected_turns, start=1):
            if self.turn_topics[t - 1] == target_topic:
                hits_all += 1
                if mrr == 0.0:
                    mrr = 1.0 / rank

        precision = (hits_all / k) if k > 0 else 0.0
        recall_all = hits_all / len(all_turns)
        hit_any = 1.0 if hits_all > 0 else 0.0

        left = max(1, current_turn - self.p.relevant_window)
        left_idx = bisect.bisect_left(all_turns, left)
        recent_total = len(all_turns) - left_idx
        hits_recent = 0
        if recent_total > 0:
            for t in selected_turns:
                if t >= left and self.turn_topics[t - 1] == target_topic:
                    hits_recent += 1
            recall_recent = hits_recent / recent_total
            recent_hit_any = 1.0 if hits_recent > 0 else 0.0
        else:
            recall_recent = 0.0
            recent_hit_any = 0.0

        self.target_precision_sum += precision
        self.target_recall_all_sum += recall_all
        self.target_recall_recent_sum += recall_recent
        self.target_hit_sum += hit_any
        self.target_recent_hit_sum += recent_hit_any
        self.target_mrr_sum += mrr
        self.eval_count += 1
        return hits_all, hits_recent

    def _heat_gini(self) -> float:
        h = self.heat_pool[: self.mem_count]
        h = h[h > 0.0]
        if h.size == 0:
            return 0.0
        hs = np.sort(h)
        n = hs.size
        cum = np.cumsum(hs)
        g = (n + 1.0 - 2.0 * np.sum(cum) / cum[-1]) / n
        return float(g)

    def _summary(self) -> Dict[str, float]:
        hot_count = int(np.sum(self.heat_pool[: self.mem_count] > 0.0))
        hot_ratio = (hot_count / self.mem_count) if self.mem_count > 0 else 0.0
        merge_rate = self.merge_count / max(1, self.p.turns)
        eval_n = max(1, self.eval_count)
        target_p = self.target_precision_sum / eval_n
        target_r_recent = self.target_recall_recent_sum / eval_n
        target_r_all = self.target_recall_all_sum / eval_n
        target_hit = self.target_hit_sum / eval_n
        target_recent_hit = self.target_recent_hit_sum / eval_n
        target_mrr = self.target_mrr_sum / eval_n
        avg_returned = self.returned_turns_sum / eval_n
        avg_add_ops = self.sim_ops_add_total / max(1, self.p.turns)
        avg_query_ops = self.sim_ops_query_total / max(1, self.query_turns)
        avg_turn_ops = float(np.mean(self.turn_sim_ops)) if self.turn_sim_ops else 0.0
        p95_turn_ops = float(np.quantile(self.turn_sim_ops, 0.95)) if self.turn_sim_ops else 0.0
        hot_clusters = sum(1 for c in self.clusters if c.is_hot_cluster)
        super_count = len(self.super_members)
        empty_rate = (self.empty_query_count / max(1, self.query_turns))
        empty_target_rate = (self.empty_target_query_count / max(1, self.query_turns))
        learning_progress_end = self._learning_progress(self.p.turns)
        route_vals = self.cluster_route_score[: len(self.clusters)] if self.clusters else np.asarray([], dtype=np.float32)
        route_abs_mean = float(np.mean(np.abs(route_vals))) if route_vals.size > 0 else 0.0
        route_pos = float(np.sum(route_vals > 0.0)) if route_vals.size > 0 else 0.0
        route_neg = float(np.sum(route_vals < 0.0)) if route_vals.size > 0 else 0.0
        route_seen = self.cluster_route_seen[: len(self.clusters)] if self.clusters else np.asarray([], dtype=np.float32)
        route_seen_avg = float(np.mean(route_seen)) if route_seen.size > 0 else 0.0
        persistent_hit_ratio = self.persistent_explore_turn_hits / max(1, self.returned_turns_sum)

        return {
            "turns": float(self.p.turns),
            "query_turns": float(self.query_turns),
            "memory_count": float(self.mem_count),
            "hot_memory_count": float(hot_count),
            "hot_memory_ratio": hot_ratio,
            "cluster_count": float(len(self.clusters)),
            "hot_cluster_count": float(hot_clusters),
            "supercluster_count": float(super_count),
            "supercluster_rebuild_count": float(self.super_rebuild_count),
            "avg_clusters_per_super": (float(len(self.clusters)) / max(1.0, float(super_count))),
            "merge_count": float(self.merge_count),
            "new_count": float(self.new_count),
            "merge_rate": merge_rate,
            "normalize_events": float(self.normalize_events),
            "total_heat": float(self.total_heat),
            "heat_gini": self._heat_gini(),
            "target_precision_at_k": target_p,
            "target_recall_recent": target_r_recent,
            "target_recall_all": target_r_all,
            "target_hit_rate": target_hit,
            "target_recent_hit_rate": target_recent_hit,
            "target_mrr": target_mrr,
            "empty_query_rate": empty_rate,
            "empty_target_query_rate": empty_target_rate,
            "learning_progress_end": learning_progress_end,
            "refinement_progress_end": self._refinement_progress(self.p.turns),
            "refinement_events": float(self.refinement_events),
            "learned_min_sim_gate": float(self.learned_min_gate),
            "online_merge_limit": float(self.online_merge_limit),
            "route_score_abs_mean": route_abs_mean,
            "route_positive_clusters": route_pos,
            "route_negative_clusters": route_neg,
            "route_seen_avg": route_seen_avg,
            "effective_probe_clusters_end": float(self._effective_probe_clusters(self.p.turns)),
            "persistent_explore_events": float(self.persistent_explore_events),
            "persistent_explore_cluster_probes": float(self.persistent_explore_cluster_probes),
            "persistent_explore_turn_hits": float(self.persistent_explore_turn_hits),
            "persistent_explore_hit_ratio": persistent_hit_ratio,
            "empty_query_count": float(self.empty_query_count),
            "empty_target_query_count": float(self.empty_target_query_count),
            "cold_rescue_enqueued": float(self.cold_rescue_enqueued),
            "cold_rescue_executed": float(self.cold_rescue_executed),
            "cold_rescue_queue_size": float(len(self.cold_rescue_queue)),
            "topic_lift_attempted": float(self.topic_lift_attempted),
            "topic_lift_executed": float(self.topic_lift_executed),
            "topic_lift_exec_rate": (self.topic_lift_executed / max(1, self.topic_lift_attempted)),
            "topic_cache_size": float(len(self.topic_cache_mem)),
            "topic_cache_unload_count": float(self.topic_cache_unload_count),
            "topic_cache_contrib_ratio": (self.topic_cache_selected_turns_total / max(1, self.returned_turns_sum)),
            "avg_returned_turns": avg_returned,
            "avg_sim_ops_add_per_turn": avg_add_ops,
            "avg_sim_ops_query_per_query": avg_query_ops,
            "avg_sim_ops_total_per_turn": avg_turn_ops,
            "p95_sim_ops_total_per_turn": p95_turn_ops,
        }

    def _record_snapshot(self, turn: int) -> None:
        s = self._summary()
        s["turn"] = float(turn)
        s["learning_progress_end"] = self._learning_progress(turn)
        s["refinement_progress_end"] = self._refinement_progress(turn)
        s["effective_probe_clusters_end"] = float(self._effective_probe_clusters(turn))
        self.snapshots.append(s)

    def _finalize_summary(self, summary: Dict[str, float]) -> Dict[str, float]:
        if not self.snapshots:
            summary["heat_gini_start"] = summary["heat_gini"]
            summary["heat_gini_end"] = summary["heat_gini"]
            summary["heat_gini_delta"] = 0.0
            summary["heat_gini_max"] = summary["heat_gini"]
            summary["heat_gini_min"] = summary["heat_gini"]
            return summary

        gini_vals = [float(s.get("heat_gini", 0.0)) for s in self.snapshots]
        g_start = gini_vals[0]
        g_end = float(summary.get("heat_gini", gini_vals[-1]))
        summary["heat_gini_start"] = g_start
        summary["heat_gini_end"] = g_end
        summary["heat_gini_delta"] = g_end - g_start
        summary["heat_gini_max"] = max(gini_vals)
        summary["heat_gini_min"] = min(gini_vals)
        return summary

    def run(self) -> Dict[str, object]:
        current_topic = self.topic.initial_topic()

        for turn in range(1, self.p.turns + 1):
            if turn > 1:
                next_topic = self.topic.next_topic(current_topic)
                if next_topic != current_topic:
                    self.prev_topic_for_shift = current_topic
                    self.last_switch_turn = turn
                    self._unload_topic_cache()
                current_topic = next_topic
            turn_vec = self.topic.turn_vector(current_topic)

            query_ops = 0
            query_prob = self.p.query_prob
            in_shift_window = self._in_shift_window(turn)
            if in_shift_window:
                query_prob = min(1.0, query_prob + self.p.shift_query_prob_boost)

            if turn > self.p.warmup_turns and self.rng.random() < query_prob:
                if (
                    in_shift_window
                    and self.prev_topic_for_shift is not None
                    and self.rng.random() < self.p.shift_target_prev_prob
                ):
                    target_topic = self.prev_topic_for_shift
                else:
                    target_topic = self._sample_query_topic(current_topic, turn)

                noise_mix = self.p.query_noise_mix
                prog = self._learning_progress(turn)
                if self.p.learning_curve_enabled:
                    noise_mix = min(0.95, noise_mix + (1.0 - prog) * self.p.learning_query_noise_extra)
                if in_shift_window:
                    noise_mix = min(0.95, noise_mix + self.p.shift_query_noise_boost)
                query_vec = self.topic.query_vector(target_topic, noise_mix)

                selected, query_ops, debug = self.retrieve(
                    query_vec,
                    current_topic=current_topic,
                    current_turn=turn,
                    target_topic=target_topic,
                )
                self.sim_ops_query_total += query_ops
                self.query_turns += 1
                if not selected:
                    self.empty_query_count += 1

                hits_all, _ = self._eval_query(selected, turn, target_topic)
                if hits_all <= 0:
                    self.empty_target_query_count += 1

                self._apply_refinement(
                    turn=turn,
                    target_topic=target_topic,
                    hits_all=hits_all,
                    candidate_samples=debug.get("candidate_samples", []),  # type: ignore[arg-type]
                    selected_memories=debug.get("selected_memories", []),  # type: ignore[arg-type]
                    evidence_memories=debug.get("evidence_memories", []),  # type: ignore[arg-type]
                )

                need_rescue = (len(selected) == 0) or ((not self.p.cold_rescue_on_empty_only) and hits_all <= 0)
                if need_rescue:
                    self._enqueue_cold_rescue(query_vec, target_topic, turn)

            add_ops_before = self.sim_ops_add_total
            self._register_turn_topic(turn, current_topic)
            stable_ready, _ = self._update_topic_stability(current_topic, turn_vec)
            self.add_memory(turn_vec, turn)
            self._topic_random_lift(turn, current_topic, stable_ready)
            self._process_cold_rescue(turn)
            add_ops = self.sim_ops_add_total - add_ops_before
            self.turn_sim_ops.append(int(add_ops + query_ops))

            if turn % self.p.report_every == 0 or turn == self.p.turns:
                self._record_snapshot(turn)

        summary = self._summary()
        summary = self._finalize_summary(summary)
        return {
            "params": vars(self.p),
            "summary": summary,
            "snapshots": self.snapshots,
        }


def save_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    path: Path,
    rows: List[Dict[str, float]],
    summary: Dict[str, float],
    dpi: int = 160,
) -> Tuple[bool, str]:
    if not rows:
        return False, "no snapshots"

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - runtime dependency check
        return False, f"matplotlib/seaborn unavailable: {exc}"

    def col(name: str) -> List[float]:
        return [float(r.get(name, 0.0)) for r in rows]

    turns = col("turn")
    heat_gini = col("heat_gini")
    precision = col("target_precision_at_k")
    recall_recent = col("target_recall_recent")
    hit_rate = col("target_hit_rate")
    memory_count = col("memory_count")
    hot_count = col("hot_memory_count")
    hot_ratio = col("hot_memory_ratio")
    empty_target_rate = col("empty_target_query_rate")
    sim_ops_avg = col("avg_sim_ops_total_per_turn")
    sim_ops_p95 = col("p95_sim_ops_total_per_turn")
    norm_events = col("normalize_events")
    rescue_exec = col("cold_rescue_executed")
    topic_lift_exec = col("topic_lift_executed")

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(turns, heat_gini, marker="o", linewidth=2.0, label="heat_gini")
    ax.axhline(float(summary.get("heat_gini_start", heat_gini[0])), color="#888888", linestyle="--", linewidth=1.4)
    ax.set_title("Heat Gini Trend")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Gini")
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(turns, precision, marker="o", linewidth=2.0, label="target_precision@k")
    ax.plot(turns, recall_recent, marker="o", linewidth=2.0, label="target_recall_recent")
    ax.plot(turns, hit_rate, marker="o", linewidth=2.0, label="target_hit_rate")
    ax.plot(turns, empty_target_rate, marker="o", linestyle="--", linewidth=1.8, label="empty_target_rate")
    ax.set_title("Recall Quality")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(turns, memory_count, marker="o", linewidth=2.0, color="#1f77b4", label="memory_count")
    ax.plot(turns, hot_count, marker="o", linewidth=2.0, color="#2ca02c", label="hot_memory_count")
    ax.set_title("Memory Growth")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Memory Count", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(turns, hot_ratio, marker="o", linewidth=2.0, color="#ff7f0e", label="hot_memory_ratio")
    ax2.set_ylabel("Hot Ratio", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")
    ax2.set_ylim(0.0, 1.0)

    ax = axes[1, 1]
    ax.plot(turns, sim_ops_avg, marker="o", linewidth=2.0, label="avg_sim_ops/turn")
    ax.plot(turns, sim_ops_p95, marker="o", linewidth=2.0, label="p95_sim_ops/turn")
    ax.set_title("Compute Load Proxy")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Similarity Ops")
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(turns, norm_events, linestyle="--", color="#8c564b", linewidth=1.8, label="normalize_events")
    ax2.plot(turns, rescue_exec, linestyle="--", color="#d62728", linewidth=1.8, label="cold_rescue_executed")
    ax2.plot(turns, topic_lift_exec, linestyle="--", color="#9467bd", linewidth=1.8, label="topic_lift_executed")
    ax2.set_ylabel("Normalize Events")

    fig.suptitle(
        "Agent Memory Simulation Dashboard\n"
        f"turns={int(summary.get('turns', 0))}, merge_rate={summary.get('merge_rate', 0.0):.3f}, "
        f"gini_delta={summary.get('heat_gini_delta', 0.0):+.3f}",
        fontsize=15,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=max(72, int(dpi)), bbox_inches="tight")
    plt.close(fig)
    return True, ""


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-memory long-run simplified simulator")
    parser.add_argument("--turns", type=int, default=20000)
    parser.add_argument("--dim", type=int, default=256, help="Use 1024 for full project dimension")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query-prob", type=float, default=0.45)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--out-csv", type=str, default="memory/sim_metrics.csv")
    parser.add_argument("--out-json", type=str, default="memory/sim_summary.json")
    parser.add_argument("--keyword-queries", type=int, default=2)
    parser.add_argument("--turn-noise-mix", type=float, default=0.18)
    parser.add_argument("--switch-prob", type=float, default=0.12)
    parser.add_argument("--near-switch-prob", type=float, default=0.55)
    parser.add_argument("--query-current-intent-prob", type=float, default=0.60)
    parser.add_argument("--query-long-term-min-age", type=int, default=120)
    parser.add_argument("--query-noise-mix", type=float, default=0.14)
    parser.add_argument("--disable-learning-curve", action="store_true")
    parser.add_argument("--learning-warmup-turns", type=int, default=500)
    parser.add_argument("--learning-full-turns", type=int, default=12000)
    parser.add_argument("--learning-query-noise-extra", type=float, default=0.18)
    parser.add_argument("--learning-min-sim-gate-start", type=float, default=0.42)
    parser.add_argument("--learning-power-suppress-start", type=float, default=1.15)
    parser.add_argument("--learning-topic-cross-quota-start", type=float, default=0.48)
    parser.add_argument("--learning-max-memory-start", type=int, default=3)
    parser.add_argument("--learning-max-turns-start", type=int, default=14)
    parser.add_argument("--learning-keyword-weight-start", type=float, default=0.78)
    parser.add_argument("--learning-super-topn-query-start", type=int, default=2)
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--refinement-start-turn", type=int, default=200)
    parser.add_argument("--refinement-sample-mem-topk", type=int, default=48)
    parser.add_argument("--refinement-route-lr", type=float, default=0.10)
    parser.add_argument("--refinement-gate-lr", type=float, default=0.08)
    parser.add_argument("--refinement-merge-lr", type=float, default=0.05)
    parser.add_argument("--refinement-route-bias-scale", type=float, default=0.08)
    parser.add_argument("--refinement-probe-clusters-start", type=int, default=8)
    parser.add_argument("--refinement-probe-clusters-end", type=int, default=2)
    parser.add_argument("--refinement-probe-per-cluster-limit", type=int, default=12)
    parser.add_argument("--disable-persistent-explore", action="store_true")
    parser.add_argument("--persistent-explore-epsilon", type=float, default=0.01)
    parser.add_argument("--persistent-explore-period-turns", type=int, default=0)
    parser.add_argument("--persistent-explore-extra-clusters", type=int, default=1)
    parser.add_argument("--persistent-explore-candidate-cap", type=int, default=32)
    parser.add_argument("--maintenance-task", type=int, default=75)
    parser.add_argument("--cold-rescue-delay-min", type=int, default=24)
    parser.add_argument("--cold-rescue-delay-max", type=int, default=120)
    parser.add_argument("--cold-rescue-topn", type=int, default=3)
    parser.add_argument("--cold-rescue-batch", type=int, default=24)
    parser.add_argument("--cold-rescue-on-empty-only", action="store_true")
    parser.add_argument("--cold-wake-multiplier", type=float, default=1.8)
    parser.add_argument("--cold-extra-neighbor-heat", type=int, default=18000)
    parser.add_argument("--shift-probe-turns", type=int, default=12)
    parser.add_argument("--shift-query-prob-boost", type=float, default=0.12)
    parser.add_argument("--shift-target-prev-prob", type=float, default=0.55)
    parser.add_argument("--shift-query-noise-boost", type=float, default=0.06)
    parser.add_argument("--topic-flow-drift", type=float, default=0.05)
    parser.add_argument("--topic-flow-anchor-mix", type=float, default=0.22)
    parser.add_argument("--topic-flow-switch-jolt", type=float, default=0.08)
    parser.add_argument("--stable-warmup-turns", type=int, default=6)
    parser.add_argument("--stable-min-pair-sim", type=float, default=0.72)
    parser.add_argument("--topic-random-lift-interval", type=int, default=3)
    parser.add_argument("--topic-random-lift-count", type=int, default=2)
    parser.add_argument("--topic-random-lift-prob", type=float, default=0.85)
    parser.add_argument("--topic-random-lift-include-hot", action="store_true")
    parser.add_argument("--topic-cache-weight", type=float, default=1.02)
    parser.add_argument("--disable-hierarchical-cluster", action="store_true")
    parser.add_argument("--supercluster-min-clusters", type=int, default=64)
    parser.add_argument("--supercluster-target-size", type=int, default=64)
    parser.add_argument("--supercluster-sim", type=float, default=0.52)
    parser.add_argument("--supercluster-max-size-mult", type=float, default=1.8)
    parser.add_argument("--supercluster-topn-add", type=int, default=3)
    parser.add_argument("--supercluster-topn-query", type=int, default=4)
    parser.add_argument("--supercluster-topn-scale", type=float, default=0.20)
    parser.add_argument("--supercluster-rebuild-every", type=int, default=600)
    parser.add_argument("--use-topic-buckets", action="store_true")
    parser.add_argument("--disable-softmax", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Disable png dashboard generation")
    parser.add_argument("--plot-out", type=str, default="", help="Output png path (default: out-csv with .png)")
    parser.add_argument("--plot-dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    p = SimParams(
        turns=max(100, args.turns),
        dim=max(32, args.dim),
        seed=args.seed,
        query_prob=min(1.0, max(0.0, args.query_prob)),
        report_every=max(100, args.report_every),
        keyword_queries=max(0, args.keyword_queries),
        turn_noise_mix=min(0.9, max(0.01, args.turn_noise_mix)),
        switch_prob=min(1.0, max(0.0, args.switch_prob)),
        near_switch_prob=min(1.0, max(0.0, args.near_switch_prob)),
        query_current_intent_prob=min(1.0, max(0.0, args.query_current_intent_prob)),
        query_long_term_min_age=max(2, args.query_long_term_min_age),
        query_noise_mix=min(0.9, max(0.01, args.query_noise_mix)),
        learning_curve_enabled=(not args.disable_learning_curve),
        learning_warmup_turns=max(0, args.learning_warmup_turns),
        learning_full_turns=max(1, args.learning_full_turns),
        learning_query_noise_extra=min(0.9, max(0.0, args.learning_query_noise_extra)),
        learning_min_sim_gate_start=min(1.0, max(0.0, args.learning_min_sim_gate_start)),
        learning_power_suppress_start=max(1.0, args.learning_power_suppress_start),
        learning_topic_cross_quota_start=min(0.5, max(0.0, args.learning_topic_cross_quota_start)),
        learning_max_memory_start=max(1, args.learning_max_memory_start),
        learning_max_turns_start=max(1, args.learning_max_turns_start),
        learning_keyword_weight_start=max(0.0, args.learning_keyword_weight_start),
        learning_super_topn_query_start=max(1, args.learning_super_topn_query_start),
        refinement_enabled=(not args.disable_refinement),
        refinement_start_turn=max(0, args.refinement_start_turn),
        refinement_sample_mem_topk=max(8, args.refinement_sample_mem_topk),
        refinement_route_lr=min(1.0, max(0.0, args.refinement_route_lr)),
        refinement_gate_lr=min(1.0, max(0.0, args.refinement_gate_lr)),
        refinement_merge_lr=min(1.0, max(0.0, args.refinement_merge_lr)),
        refinement_route_bias_scale=max(0.0, args.refinement_route_bias_scale),
        refinement_probe_clusters_start=max(1, args.refinement_probe_clusters_start),
        refinement_probe_clusters_end=max(1, args.refinement_probe_clusters_end),
        refinement_probe_per_cluster_limit=max(2, args.refinement_probe_per_cluster_limit),
        persistent_explore_enabled=(not args.disable_persistent_explore),
        persistent_explore_epsilon=min(1.0, max(0.0, args.persistent_explore_epsilon)),
        persistent_explore_period_turns=max(0, args.persistent_explore_period_turns),
        persistent_explore_extra_clusters=max(1, args.persistent_explore_extra_clusters),
        persistent_explore_candidate_cap=max(1, args.persistent_explore_candidate_cap),
        maintenance_task=max(1, args.maintenance_task),
        cold_rescue_delay_min=max(1, args.cold_rescue_delay_min),
        cold_rescue_delay_max=max(1, args.cold_rescue_delay_max),
        cold_rescue_topn=max(1, args.cold_rescue_topn),
        cold_rescue_batch=max(1, args.cold_rescue_batch),
        cold_rescue_on_empty_only=args.cold_rescue_on_empty_only,
        cold_wake_multiplier=max(1.0, args.cold_wake_multiplier),
        cold_extra_neighbor_heat=max(0, args.cold_extra_neighbor_heat),
        shift_probe_turns=max(0, args.shift_probe_turns),
        shift_query_prob_boost=min(1.0, max(0.0, args.shift_query_prob_boost)),
        shift_target_prev_prob=min(1.0, max(0.0, args.shift_target_prev_prob)),
        shift_query_noise_boost=min(0.9, max(0.0, args.shift_query_noise_boost)),
        topic_flow_drift=min(0.9, max(0.0, args.topic_flow_drift)),
        topic_flow_anchor_mix=min(0.9, max(0.0, args.topic_flow_anchor_mix)),
        topic_flow_switch_jolt=min(0.9, max(0.0, args.topic_flow_switch_jolt)),
        stable_warmup_turns=max(1, args.stable_warmup_turns),
        stable_min_pair_sim=min(1.0, max(0.0, args.stable_min_pair_sim)),
        topic_random_lift_interval=max(1, args.topic_random_lift_interval),
        topic_random_lift_count=max(1, args.topic_random_lift_count),
        topic_random_lift_prob=min(1.0, max(0.0, args.topic_random_lift_prob)),
        topic_random_lift_only_cold=(not args.topic_random_lift_include_hot),
        topic_cache_weight=max(0.5, args.topic_cache_weight),
        hierarchical_cluster_enabled=(not args.disable_hierarchical_cluster),
        supercluster_min_clusters=max(8, args.supercluster_min_clusters),
        supercluster_target_size=max(8, args.supercluster_target_size),
        supercluster_sim=min(1.0, max(-1.0, args.supercluster_sim)),
        supercluster_max_size_mult=max(1.0, args.supercluster_max_size_mult),
        supercluster_topn_add=max(1, args.supercluster_topn_add),
        supercluster_topn_query=max(1, args.supercluster_topn_query),
        supercluster_topn_scale=max(0.0, args.supercluster_topn_scale),
        supercluster_rebuild_every=max(0, args.supercluster_rebuild_every),
        use_topic_buckets=args.use_topic_buckets,
        softmax=not args.disable_softmax,
    )

    if p.cold_rescue_delay_max < p.cold_rescue_delay_min:
        p.cold_rescue_delay_max = p.cold_rescue_delay_min
    if p.learning_full_turns <= p.learning_warmup_turns:
        p.learning_full_turns = p.learning_warmup_turns + 1

    sim = AgentMemorySim(p)
    result = sim.run()

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    save_csv(out_csv, result["snapshots"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    summary = result["summary"]
    print("=== Agent Memory Simulation Summary ===")
    print(f"turns: {int(summary['turns'])}")
    print(f"query_turns: {int(summary['query_turns'])}")
    print(f"memory_count: {int(summary['memory_count'])}")
    print(f"hot_memory_ratio: {summary['hot_memory_ratio']:.4f}")
    print(f"cluster_count: {int(summary['cluster_count'])}")
    print(f"supercluster_count: {int(summary['supercluster_count'])}")
    print(f"supercluster_rebuild_count: {int(summary['supercluster_rebuild_count'])}")
    print(f"merge_rate: {summary['merge_rate']:.4f}")
    print(f"target_precision_at_k: {summary['target_precision_at_k']:.4f}")
    print(f"target_recall_recent: {summary['target_recall_recent']:.4f}")
    print(f"target_recall_all: {summary['target_recall_all']:.4f}")
    print(f"target_hit_rate: {summary['target_hit_rate']:.4f}")
    print(f"target_recent_hit_rate: {summary['target_recent_hit_rate']:.4f}")
    print(f"target_mrr: {summary['target_mrr']:.4f}")
    print(f"empty_query_rate: {summary['empty_query_rate']:.4f}")
    print(f"empty_target_query_rate: {summary['empty_target_query_rate']:.4f}")
    print(f"learning_progress_end: {summary['learning_progress_end']:.4f}")
    print(f"refinement_progress_end: {summary['refinement_progress_end']:.4f}")
    print(f"refinement_events: {int(summary['refinement_events'])}")
    print(f"learned_min_sim_gate: {summary['learned_min_sim_gate']:.4f}")
    print(f"online_merge_limit: {summary['online_merge_limit']:.4f}")
    print(f"route_score_abs_mean: {summary['route_score_abs_mean']:.4f}")
    print(f"effective_probe_clusters_end: {summary['effective_probe_clusters_end']:.1f}")
    print(f"persistent_explore_events: {int(summary['persistent_explore_events'])}")
    print(f"persistent_explore_cluster_probes: {int(summary['persistent_explore_cluster_probes'])}")
    print(f"persistent_explore_turn_hits: {int(summary['persistent_explore_turn_hits'])}")
    print(f"persistent_explore_hit_ratio: {summary['persistent_explore_hit_ratio']:.4f}")
    print(f"cold_rescue_enqueued: {int(summary['cold_rescue_enqueued'])}")
    print(f"cold_rescue_executed: {int(summary['cold_rescue_executed'])}")
    print(f"topic_lift_attempted: {int(summary['topic_lift_attempted'])}")
    print(f"topic_lift_executed: {int(summary['topic_lift_executed'])}")
    print(f"topic_lift_exec_rate: {summary['topic_lift_exec_rate']:.4f}")
    print(f"topic_cache_size: {int(summary['topic_cache_size'])}")
    print(f"topic_cache_unload_count: {int(summary['topic_cache_unload_count'])}")
    print(f"topic_cache_contrib_ratio: {summary['topic_cache_contrib_ratio']:.4f}")
    print(f"avg_sim_ops_total_per_turn: {summary['avg_sim_ops_total_per_turn']:.2f}")
    print(f"p95_sim_ops_total_per_turn: {summary['p95_sim_ops_total_per_turn']:.2f}")
    print(f"normalize_events: {int(summary['normalize_events'])}")
    print(f"heat_gini_start: {summary['heat_gini_start']:.4f}")
    print(f"heat_gini_end: {summary['heat_gini_end']:.4f}")
    print(f"heat_gini_delta: {summary['heat_gini_delta']:.4f}")
    print(f"heat_gini_max: {summary['heat_gini_max']:.4f}")
    print(f"snapshot_csv: {out_csv}")
    print(f"summary_json: {out_json}")

    if not args.no_plot:
        plot_out = Path(args.plot_out) if args.plot_out else out_csv.with_suffix(".png")
        ok, reason = save_plot(plot_out, result["snapshots"], summary, dpi=args.plot_dpi)
        if ok:
            print(f"snapshot_plot: {plot_out}")
        else:
            print(f"snapshot_plot: skipped ({reason})")


if __name__ == "__main__":
    main()
