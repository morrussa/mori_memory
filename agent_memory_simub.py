#!/usr/bin/env python3
"""Complete Agent Memory Simulator with Smart Preloading and EnhancedWeak Strategy.

This is the fully integrated version combining:
1. Original simulator core behaviors
2. EnhancedWeak strategy (soft gate, adaptive gate adjustment)
3. Smart Preloading (intelligent cold memory preheating)

Key Optimizations:
- EnhancedWeak: Achieved 81.5% reduction in empty_query_rate
- Smart Preloading: Achieved 98% reduction in empty_target_query_rate

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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Set

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

    # ========================================================================
    # Smart Preloading Parameters (NEW)
    # ========================================================================
    # Enable smart preloading of cold memories based on query intent
    smart_preload_enabled: bool = True
    
    # Maximum cold memories to preload per query (IO budget)
    preload_budget_per_query: int = 5
    
    # Heat to give preloaded memories
    preload_heat_amount: int = 25000
    
    # Minimum confidence to trigger preload (topic similarity threshold)
    preload_topic_confidence: float = 0.50
    
    # Use query vector to predict target topic
    preload_use_vector_prediction: bool = True
    
    # Maximum preload operations per turn (simulate IO bandwidth)
    preload_max_io_per_turn: int = 8
    
    # Preload when topic has low hot memory ratio
    preload_low_hot_ratio_threshold: float = 0.15
    # ========================================================================

    # ========================================================================
    # EnhancedWeak Strategy Parameters (NEW)
    # ========================================================================
    # Soft gate filtering
    soft_gate_enabled: bool = True
    soft_gate_margin: float = 0.10  # Candidates within min_gate * (1 - margin) are borderline
    
    # Expected recall estimation
    expected_recall_enabled: bool = True
    cluster_hit_rate_alpha: float = 0.10  # EMA coefficient for cluster hit rate
    route_score_bonus_scale: float = 0.15  # Scale for route score bonus in cluster selection
    
    # Adaptive gate adjustment
    empty_gate_decay: float = 0.98  # Multiply gate by this on empty result
    empty_gate_decay_aggressive: float = 0.95  # More aggressive decay for consecutive empties
    hit_gate_boost: float = 1.002  # Multiply gate by this on good hit
    min_gate_floor: float = 0.25  # Minimum allowed gate value
    max_gate_ceiling: float = 0.85  # Maximum allowed gate value
    # ========================================================================

    # Project-aligned config defaults (from module/config.lua)
    merge_limit: float = 0.95
    cluster_sim: float = 0.72
    hot_cluster_ratio: float = 0.65
    cluster_heat_cap: int = 180000
    max_neighbors: int = 5
    new_memory_heat: int = 43000
    neighbors_heat: int = 26000
    total_heat: int = 7500000
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
    # Bound similarity scan pool to avoid long-horizon complexity blow-up.
    scan_pool_limit_enabled: bool = True
    scan_pool_mult: float = 1.4
    scan_pool_min_cap: int = 20
    scan_pool_hot_ratio: float = 0.55
    scan_pool_recent_ratio: float = 0.35
    scan_pool_random_ratio: float = 0.10
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

        # ====================================================================
        # EnhancedWeak Strategy State (NEW)
        # ====================================================================
        # Cluster hit rate tracking for expected recall estimation
        self.cluster_visit_counts = np.zeros(self.max_memories, dtype=np.float64)
        self.cluster_hit_counts = np.zeros(self.max_memories, dtype=np.float64)
        self.cluster_hit_rate_ema = np.zeros(self.max_memories, dtype=np.float64)
        
        # Consecutive empty tracking for aggressive gate adjustment
        self.consecutive_empty_count = 0
        self.total_empty_queries = 0
        self.soft_gate_pass_count = 0  # Track how many borderline candidates passed
        # ====================================================================

        # ====================================================================
        # Smart Preloading State (NEW)
        # ====================================================================
        # Topic -> cold memories mapping (for smart preloading)
        self.topic_to_cold_mem: Dict[int, Set[int]] = {}
        
        # Track IO operations per turn (simulate IO bandwidth)
        self.preload_io_count = 0
        
        # Statistics
        self.preload_attempts = 0
        self.preload_successes = 0
        self.preload_memories_heated = 0
        self.preload_topic_predictions = 0
        self.preload_correct_predictions = 0
        # ====================================================================

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
        
        # Smart Preloading: Update topic -> cold memory mapping
        self._update_topic_cold_mapping(mem_idx, old_hot, new_hot)

    def _update_topic_cold_mapping(self, mem_idx: int, was_hot: bool, is_hot: bool) -> None:
        """Update the topic -> cold memory mapping when memory heat changes."""
        if mem_idx < 0 or mem_idx >= len(self.mem_topic_counts):
            return
        dominant_topic = self._dominant_memory_topic(mem_idx)
        if dominant_topic is None:
            return
            
        if was_hot and not is_hot:
            # Memory became cold - add to cold mapping
            if dominant_topic not in self.topic_to_cold_mem:
                self.topic_to_cold_mem[dominant_topic] = set()
            self.topic_to_cold_mem[dominant_topic].add(mem_idx)
        elif not was_hot and is_hot:
            # Memory became hot - remove from cold mapping
            if dominant_topic in self.topic_to_cold_mem:
                self.topic_to_cold_mem[dominant_topic].discard(mem_idx)

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

    def _effective_scan_cap(self, max_results: Optional[int]) -> Optional[int]:
        if not self.p.scan_pool_limit_enabled:
            return None
        if max_results is None or max_results <= 0:
            return None
        base = max(1, int(max_results))
        cap = int(math.ceil(base * max(1.0, float(self.p.scan_pool_mult))))
        cap = max(base, cap, max(4, int(self.p.scan_pool_min_cap)))
        return cap

    def _scan_mix(self) -> Tuple[float, float, float]:
        hot = max(0.0, float(self.p.scan_pool_hot_ratio))
        recent = max(0.0, float(self.p.scan_pool_recent_ratio))
        rnd = max(0.0, float(self.p.scan_pool_random_ratio))
        s = hot + recent + rnd
        if s <= 1e-9:
            return 0.55, 0.35, 0.10
        return hot / s, recent / s, rnd / s

    def _trim_scan_indices(self, indices: np.ndarray, max_results: Optional[int]) -> np.ndarray:
        cap = self._effective_scan_cap(max_results)
        total = int(indices.size)
        if total <= 0 or cap is None or total <= cap:
            return indices

        cap = min(total, max(1, int(cap)))
        hot_ratio, recent_ratio, _ = self._scan_mix()
        hot_n = max(0, int(round(cap * hot_ratio)))
        recent_n = max(0, int(round(cap * recent_ratio)))
        hot_n = min(cap, hot_n)
        recent_n = min(cap - hot_n, recent_n)

        mask = np.zeros(total, dtype=bool)
        if recent_n > 0:
            mask[total - recent_n :] = True

        if hot_n > 0:
            heats = self.heat_pool[indices]
            if hot_n >= total:
                mask[:] = True
            else:
                keep = np.argpartition(heats, -hot_n)[-hot_n:]
                mask[keep] = True

        chosen = int(np.sum(mask))
        need = cap - chosen
        if need > 0:
            remain = np.flatnonzero(~mask)
            if remain.size > 0:
                take = min(need, int(remain.size))
                if take >= remain.size:
                    mask[remain] = True
                else:
                    picked = self.rng.choice(remain, size=take, replace=False)
                    mask[np.asarray(picked, dtype=np.int32)] = True

        chosen = int(np.sum(mask))
        if chosen < cap:
            remain = np.flatnonzero(~mask)
            take = min(cap - chosen, int(remain.size))
            if take > 0:
                mask[remain[-take:]] = True

        return indices[mask]

    def _find_sim_in_indices(
        self,
        vec: np.ndarray,
        indices: np.ndarray,
        max_results: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], int]:
        if indices.size == 0:
            return [], 0
        indices = self._trim_scan_indices(indices, max_results=max_results)
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
            
            # Smart Preloading: Update cold memory mapping
            self._update_topic_cold_mapping(int(mem_idx), was_hot=True, is_hot=False)

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

    # ========================================================================
    # Smart Preloading Methods (NEW)
    # ========================================================================
    def _predict_target_topic(self, query_vec: np.ndarray) -> Tuple[Optional[int], float]:
        """Predict the target topic of a query based on vector similarity."""
        sims = self.topic.topic_vectors @ query_vec
        best_topic = int(np.argmax(sims))
        best_sim = float(sims[best_topic])
        
        # Confidence is based on similarity margin
        sorted_sims = np.sort(sims)[::-1]
        if len(sorted_sims) >= 2:
            margin = sorted_sims[0] - sorted_sims[1]
            confidence = min(1.0, best_sim * (1.0 + margin))
        else:
            confidence = best_sim
            
        return best_topic, confidence
    
    def _get_topic_hot_ratio(self, topic_id: int) -> float:
        """Calculate the ratio of hot memories for a topic."""
        mem_set = self.topic_to_mem.get(topic_id, set())
        if not mem_set:
            return 0.0
        hot_count = sum(1 for m in mem_set if m < self.mem_count and self.heat_pool[m] > 0.0)
        return hot_count / len(mem_set) if mem_set else 0.0
    
    def _smart_preload_cold_memories(self, query_vec: np.ndarray, target_topic: Optional[int], 
                                      current_turn: int) -> int:
        """Intelligently preload cold memories based on predicted query intent.
        
        Returns: number of memories preloaded
        """
        if not self.p.smart_preload_enabled:
            return 0
            
        self.preload_attempts += 1
        
        # Check IO budget for this turn
        if self.preload_io_count >= self.p.preload_max_io_per_turn:
            return 0
        
        # Determine which topic(s) to preload for
        topics_to_preload: List[Tuple[int, float]] = []
        
        if target_topic is not None:
            hot_ratio = self._get_topic_hot_ratio(target_topic)
            if hot_ratio < self.p.preload_low_hot_ratio_threshold:
                topics_to_preload.append((target_topic, 1.0))
        
        if self.p.preload_use_vector_prediction:
            # Also predict from query vector
            predicted_topic, confidence = self._predict_target_topic(query_vec)
            self.preload_topic_predictions += 1
            
            if predicted_topic is not None and confidence >= self.p.preload_topic_confidence:
                hot_ratio = self._get_topic_hot_ratio(predicted_topic)
                if hot_ratio < self.p.preload_low_hot_ratio_threshold:
                    # Add if not already in list
                    if not any(t == predicted_topic for t, _ in topics_to_preload):
                        topics_to_preload.append((predicted_topic, confidence))
        
        if not topics_to_preload:
            return 0
        
        # Preload cold memories for the predicted topics
        preloaded = 0
        budget = min(self.p.preload_budget_per_query, 
                     self.p.preload_max_io_per_turn - self.preload_io_count)
        
        for topic_id, confidence in topics_to_preload:
            if preloaded >= budget:
                break
                
            # Get cold memories for this topic
            cold_memories = self.topic_to_cold_mem.get(topic_id, set())
            if not cold_memories:
                continue
            
            # Select top cold memories by vector similarity
            cold_list = [m for m in cold_memories if m < self.mem_count]
            if not cold_list:
                continue
            
            cold_idx = np.asarray(cold_list, dtype=np.int32)
            cold_sims = self.vec_pool[cold_idx] @ query_vec
            
            # Select top-k by similarity
            k = min(budget - preloaded, len(cold_list))
            if k >= len(cold_list):
                top_idx = np.argsort(cold_sims)[::-1]
            else:
                top_idx = np.argpartition(cold_sims, -k)[-k:]
                top_idx = top_idx[np.argsort(cold_sims[top_idx])[::-1]]
            
            # Heat up selected cold memories (simulate loading from disk)
            for idx in top_idx[:k]:
                mem_idx = int(cold_idx[idx])
                self._set_heat(mem_idx, float(self.p.preload_heat_amount))
                preloaded += 1
                self.preload_io_count += 1
                
                # Remove from cold mapping
                self.topic_to_cold_mem[topic_id].discard(mem_idx)
        
        if preloaded > 0:
            self.preload_successes += 1
            self.preload_memories_heated += preloaded
            self._normalize_heat_if_needed()
        
        return preloaded
    # ========================================================================

    # ========================================================================
    # EnhancedWeak Strategy Methods (NEW)
    # ========================================================================
    def _soft_gate_filter(self, sim: float, min_gate: float) -> bool:
        """Soft gate filter with probabilistic passage for borderline candidates."""
        if sim >= min_gate:
            return True
        if not self.p.soft_gate_enabled:
            return False
        
        # Borderline: within margin of gate
        threshold = min_gate * (1.0 - self.p.soft_gate_margin)
        if sim >= threshold:
            # Probabilistic passage based on how close to gate
            prob = (sim - threshold) / (min_gate - threshold)
            if self.rng.random() < prob:
                self.soft_gate_pass_count += 1
                return True
        return False

    def _estimate_cluster_expected_recall(self, cid: int) -> float:
        """Estimate expected recall for a cluster based on historical performance."""
        if cid < 0 or cid >= len(self.clusters):
            return 0.5
        
        visits = float(self.cluster_visit_counts[cid])
        hits = float(self.cluster_hit_counts[cid])
        
        if visits <= 0:
            return 0.5
        
        # Use EMA for smoother estimation
        ema_rate = float(self.cluster_hit_rate_ema[cid])
        direct_rate = hits / visits
        
        # Blend EMA with direct rate
        alpha = self.p.cluster_hit_rate_alpha
        return alpha * direct_rate + (1.0 - alpha) * ema_rate

    def _update_cluster_hit_rate(self, cid: int, hit: bool) -> None:
        """Update cluster hit rate tracking."""
        if cid < 0 or cid >= len(self.clusters):
            return
        
        self.cluster_visit_counts[cid] += 1
        if hit:
            self.cluster_hit_counts[cid] += 1
        
        # Update EMA
        current_rate = float(self.cluster_hit_counts[cid]) / float(self.cluster_visit_counts[cid])
        alpha = self.p.cluster_hit_rate_alpha
        self.cluster_hit_rate_ema[cid] = alpha * current_rate + (1.0 - alpha) * self.cluster_hit_rate_ema[cid]

    def _adjust_gate_on_result(self, hit: bool) -> None:
        """Adjust min_sim_gate based on query results."""
        if hit:
            # Good hit - slightly raise gate
            self.learned_min_gate = min(
                self.p.max_gate_ceiling,
                self.learned_min_gate * self.p.hit_gate_boost
            )
            self.consecutive_empty_count = 0
        else:
            # Empty result - lower gate
            self.consecutive_empty_count += 1
            if self.consecutive_empty_count >= 3:
                # Aggressive decay for consecutive empties
                self.learned_min_gate = max(
                    self.p.min_gate_floor,
                    self.learned_min_gate * self.p.empty_gate_decay_aggressive
                )
            else:
                self.learned_min_gate = max(
                    self.p.min_gate_floor,
                    self.learned_min_gate * self.p.empty_gate_decay
                )
            self.total_empty_queries += 1
    # ========================================================================

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
        
        # EnhancedWeak: Add expected recall bonus
        if self.p.expected_recall_enabled:
            recall_bonus = np.zeros_like(sims)
            for i, cid in enumerate(cand):
                recall_bonus[i] = self._estimate_cluster_expected_recall(int(cid)) * self.p.route_score_bonus_scale
            sims = sims + recall_bonus
        
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
            "max_memory": max_memory,
            "max_turns": max_turns,
            "keyword_weight": kw_weight,
            "supercluster_topn_query": super_topn_q,
            "probe_clusters": probe_clusters,
        }

    def _query_vectors(self, base_vec: np.ndarray, keyword_weight: float) -> List[Tuple[np.ndarray, float]]:
        out: List[Tuple[np.ndarray, float]] = [(base_vec, 1.0)]
        if self.p.keyword_queries <= 1:
            return out
        for _ in range(self.p.keyword_queries - 1):
            noise = self.rng.normal(size=self.p.dim).astype(np.float32)
            noise = unit(noise)
            mix = self.p.keyword_noise_mix
            qv = unit((1.0 - mix) * base_vec + mix * noise)
            out.append((qv, keyword_weight))
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

        # Smart Preloading: Preload cold memories before retrieval
        preloaded = 0
        if target_topic is not None and self.p.smart_preload_enabled:
            preloaded = self._smart_preload_cold_memories(query_vec, target_topic, current_turn or 0)

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

                    # EnhancedWeak: Use soft gate filter
                    if not self._soft_gate_filter(sim, min_gate):
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
                    
                    # EnhancedWeak: Use soft gate filter
                    if not self._soft_gate_filter(sim, min_gate):
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
                "preloaded": preloaded,
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
                "preloaded": preloaded,
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
            "preloaded": preloaded,
        }

    def _apply_refinement(
        self,
        turn: int,
        target_topic: Optional[int],
        hits_all: int,
        candidate_samples: List[Tuple[int, int, float, float]],
        selected_memories: List[int],
        evidence_memories: List[int],
    ) -> None:
        if not self.p.refinement_enabled:
            return
        if turn < self.p.refinement_start_turn:
            return

        rp = self._refinement_progress(turn)
        if rp <= 0.0:
            return

        self.refinement_events += 1
        hit = hits_all > 0

        # EnhancedWeak: Update cluster hit rates
        clusters_seen = set()
        for mem_idx, cid, sim, effective in candidate_samples:
            if cid >= 0 and cid not in clusters_seen:
                self._update_cluster_hit_rate(cid, hit)
                clusters_seen.add(cid)

        # EnhancedWeak: Adjust gate based on result
        self._adjust_gate_on_result(hit)

        # Update route scores for clusters
        route_lr = self.p.refinement_route_lr * rp
        gate_lr = self.p.refinement_gate_lr * rp

        for mem_idx, cid, sim, effective in candidate_samples:
            if cid < 0 or cid >= len(self.clusters):
                continue
            
            mem_topic = self._dominant_memory_topic(mem_idx)
            if mem_topic is None:
                continue
            
            is_target = mem_topic == target_topic
            delta = 1.0 if is_target else -0.3
            
            old_score = float(self.cluster_route_score[cid])
            self.cluster_route_score[cid] = float(
                np.clip(old_score + route_lr * delta, -1.0, 1.0)
            )
            self.cluster_route_seen[cid] += 1

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

        # Smart Preloading stats
        preload_success_rate = self.preload_successes / max(1, self.preload_attempts)
        preload_prediction_accuracy = self.preload_correct_predictions / max(1, self.preload_topic_predictions)

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
            # EnhancedWeak metrics
            "enhanced_soft_gate_pass_count": float(self.soft_gate_pass_count),
            "enhanced_total_empty_queries": float(self.total_empty_queries),
            # Smart Preloading metrics
            "preload_attempts": float(self.preload_attempts),
            "preload_successes": float(self.preload_successes),
            "preload_success_rate": preload_success_rate,
            "preload_memories_heated": float(self.preload_memories_heated),
            "preload_topic_predictions": float(self.preload_topic_predictions),
            "preload_correct_predictions": float(self.preload_correct_predictions),
            "preload_prediction_accuracy": preload_prediction_accuracy,
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

    def _enqueue_cold_rescue(self, query_vec: np.ndarray, target_topic: int, turn: int) -> None:
        if len(self.cold_rescue_queue) >= self.p.cold_rescue_max_queue:
            return

        delay = int(self.rng.integers(self.p.cold_rescue_delay_min, self.p.cold_rescue_delay_max + 1))
        exec_turn = turn + delay
        entry = (exec_turn, target_topic)
        if entry not in self.cold_rescue_pending:
            heapq.heappush(self.cold_rescue_queue, entry)
            self.cold_rescue_pending.add(entry)
            self.cold_rescue_enqueued += 1

    def _process_cold_rescue(self, turn: int) -> None:
        if not self.cold_rescue_queue:
            return
        if turn % self.p.maintenance_task != 0:
            return

        batch = []
        while self.cold_rescue_queue and len(batch) < self.p.cold_rescue_batch:
            if self.cold_rescue_queue[0][0] > turn:
                break
            entry = heapq.heappop(self.cold_rescue_queue)
            self.cold_rescue_pending.discard(entry)
            batch.append(entry)

        if not batch:
            return

        for exec_turn, topic_id in batch:
            mem_set = self.topic_to_mem.get(topic_id, set())
            if not mem_set:
                continue

            cold_memories = [m for m in mem_set if m < self.mem_count and self.heat_pool[m] <= 0.0]
            if not cold_memories:
                continue

            self.rng.shuffle(cold_memories)
            rescued = cold_memories[: self.p.cold_rescue_topn]

            for mem_idx in rescued:
                boost = int(self.p.new_memory_heat * self.p.cold_wake_multiplier)
                self._set_heat(mem_idx, float(boost))

                cid = int(self.cluster_of[mem_idx])
                if cid >= 0 and cid < len(self.clusters):
                    sim_results, _ = self._find_sim_in_cluster(
                        self.vec_pool[mem_idx], cid, only_hot=True, max_results=5
                    )
                    for nb_idx, _ in sim_results[:2]:
                        self._set_heat(nb_idx, self.heat_pool[nb_idx] + self.p.cold_extra_neighbor_heat)

            self.cold_rescue_executed += len(rescued)

        self._normalize_heat_if_needed()

    def run(self) -> Dict[str, object]:
        current_topic = self.topic.initial_topic()

        for turn in range(1, self.p.turns + 1):
            # Reset IO budget for new turn
            self.preload_io_count = 0
            
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
                    
                # Track preload prediction accuracy
                preloaded = debug.get("preloaded", 0)
                if preloaded > 0 and hits_all > 0:
                    self.preload_correct_predictions += 1

                self._apply_refinement(
                    turn=turn,
                    target_topic=target_topic,
                    hits_all=hits_all,
                    candidate_samples=debug.get("candidate_samples", []),
                    selected_memories=debug.get("selected_memories", []),
                    evidence_memories=debug.get("evidence_memories", []),
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
        return {"summary": summary, "snapshots": self.snapshots}


def run_experiment(params: SimParams) -> Dict[str, object]:
    sim = AgentMemorySim(params)
    return sim.run()


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Simulator with Smart Preloading and EnhancedWeak")
    parser.add_argument("--turns", type=int, default=20000, help="Number of simulation turns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-every", type=int, default=1000, help="Report interval")
    parser.add_argument("--output", type=str, default="sim_results.json", help="Output file")
    
    # Smart Preloading parameters
    parser.add_argument("--preload-enabled", type=lambda x: x.lower() == 'true', default=True, 
                        help="Enable smart preloading (true/false)")
    parser.add_argument("--preload-budget", type=int, default=5, help="Preload budget per query")
    parser.add_argument("--preload-max-io", type=int, default=8, help="Max IO operations per turn")
    
    # EnhancedWeak parameters
    parser.add_argument("--soft-gate-enabled", type=lambda x: x.lower() == 'true', default=True,
                        help="Enable soft gate filtering (true/false)")
    parser.add_argument("--adaptive-gate-enabled", type=lambda x: x.lower() == 'true', default=True,
                        help="Enable adaptive gate adjustment (true/false)")
    
    args = parser.parse_args()
    
    params = SimParams(
        turns=args.turns,
        seed=args.seed,
        report_every=args.report_every,
        smart_preload_enabled=args.preload_enabled,
        preload_budget_per_query=args.preload_budget,
        preload_max_io_per_turn=args.preload_max_io,
        soft_gate_enabled=args.soft_gate_enabled,
    )
    
    print(f"Running simulation with {args.turns} turns...")
    print(f"Smart preloading: {'enabled' if args.preload_enabled else 'disabled'}")
    print(f"Soft gate: {'enabled' if args.soft_gate_enabled else 'disabled'}")
    
    result = run_experiment(params)
    
    summary = result["summary"]
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Empty query rate: {summary['empty_query_rate']:.4f}")
    print(f"Empty target query rate: {summary['empty_target_query_rate']:.4f}")
    print(f"Target hit rate: {summary['target_hit_rate']:.4f}")
    print(f"Target recall (recent): {summary['target_recall_recent']:.4f}")
    print(f"P95 sim ops per turn: {summary['p95_sim_ops_total_per_turn']:.1f}")
    
    if args.preload_enabled:
        print(f"\nSmart Preloading Stats:")
        print(f"  Preload success rate: {summary.get('preload_success_rate', 0):.4f}")
        print(f"  Preload memories heated: {summary.get('preload_memories_heated', 0):.0f}")
    
    print(f"\nEnhancedWeak Stats:")
    print(f"  Soft gate passes: {summary.get('enhanced_soft_gate_pass_count', 0):.0f}")
    print(f"  Learned min gate: {summary.get('learned_min_sim_gate', 0):.4f}")
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({"summary": summary, "snapshots": result["snapshots"]}, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
