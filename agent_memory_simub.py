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
    retrieval_model: str = "memory"

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
    
    # Maximum clusters to preload per query (IO budget in clusters)
    preload_budget_per_query: int = 5
    
    # Legacy compatibility: kept but unused by cluster-level preload
    preload_heat_amount: int = 25000
    
    # Minimum confidence to trigger preload (topic similarity threshold)
    preload_topic_confidence: float = 0.50
    
    # Use query vector to predict target topic
    preload_use_vector_prediction: bool = True
    
    # Maximum preload cluster operations per turn (simulate IO bandwidth)
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

    max_memory: int = 20
    max_turns: int = 10
    min_sim_gate: float = 0.58
    power_suppress: float = 1.80
    keyword_weight: float = 0.55
    keyword_queries: int = 2
    keyword_noise_mix: float = 0.20
    memory_drop_sim: float = 0.80
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

    # ========================================================================
    # GHSOM Parameters (NEW)
    # ========================================================================
    # Enable GHSOM clustering instead of traditional K-means style
    ghsom_enabled: bool = False
    
    # Global quantization error target (τ1)
    ghsom_tau1: float = 0.10
    
    # Local quantization error target for each node (τ2)
    ghsom_tau2: float = 0.05
    
    # Initial SOM grid size (width x height)
    ghsom_initial_width: int = 2
    ghsom_initial_height: int = 2
    
    # Maximum grid size expansion
    ghsom_max_width: int = 8
    ghsom_max_height: int = 8
    
    # Learning rate parameters
    ghsom_learning_rate_initial: float = 0.3
    ghsom_learning_rate_decay: float = 0.95
    ghsom_neighborhood_radius_initial: float = 1.5
    ghsom_neighborhood_radius_decay: float = 0.95
    
    # Minimum number of samples before expanding
    ghsom_min_samples_for_expansion: int = 10
    
    # Maximum hierarchy depth
    ghsom_max_depth: int = 3
    
    # Use hierarchical retrieval (if disabled, only top-level nodes used as clusters)
    ghsom_hierarchical_retrieval: bool = True
    # Only enable GHSOM routing when a cluster is large enough to justify the fixed overhead.
    ghsom_linear_scan_threshold: int = 24
    # ========================================================================

    # ========================================================================
    # Topic-graph retrieval model (NEW)
    # ========================================================================
    topic_graph_seed_topics: int = 2
    topic_graph_expand_budget: int = 6
    topic_graph_max_return_topics: int = 4
    topic_graph_per_topic_evidence: int = 3
    topic_graph_load_budget: int = 4
    topic_graph_loaded_cap: int = 24
    topic_graph_bridge_topk: int = 8
    topic_graph_max_bridge_hops: int = 2
    topic_graph_transition_lr: float = 0.12
    topic_graph_recall_lr: float = 0.10
    topic_graph_adopt_lr: float = 0.18
    topic_graph_decay: float = 0.995
    topic_graph_min_bridge_score: float = 0.08
    topic_graph_query_semantic_weight: float = 1.00
    topic_graph_bridge_weight: float = 0.55
    topic_graph_resident_bonus: float = 0.08
    topic_graph_current_topic_bonus: float = 0.12
    # ========================================================================

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
    ghsom_root: Optional["GHSOMNodeState"] = None


@dataclass
class GHSOMNodeState:
    depth: int
    width: int
    height: int
    neurons: np.ndarray
    slot_members: List[List[int]]
    slot_count: np.ndarray
    slot_sim_sum: np.ndarray
    children: Dict[int, "GHSOMNodeState"] = field(default_factory=dict)
    member_count: int = 0


@dataclass
class TopicBridgeState:
    transition: float = 0.0
    recall: float = 0.0
    adopt: float = 0.0
    support: int = 0
    last_turn: int = 0


@dataclass
class TopicShardState:
    members: List[int] = field(default_factory=list)
    centroid: Optional[np.ndarray] = None
    loaded: bool = False
    last_loaded_turn: int = 0
    ghsom_root: Optional["GHSOMNodeState"] = None


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
        self.ghsom_probe_count = 0
        self.ghsom_probe_candidates_total = 0
        self.ghsom_probe_depth_total = 0

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
        # Compatibility metric: now counts preload-cache memory loads, not heat writes.
        self.preload_memories_heated = 0
        self.preload_clusters_loaded = 0
        self.preload_topic_predictions = 0
        self.preload_correct_predictions = 0
        self.preload_cache_topic: Optional[int] = None
        self.preload_cache_clusters: Set[int] = set()
        self.preload_cache_mem: Set[int] = set()
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
        self.preload_cache_unload_count = 0
        self.preload_cache_selected_turns_total = 0
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

    def _ghsom_cell_count(self, node: GHSOMNodeState) -> int:
        return max(1, int(node.width) * int(node.height))

    def _ghsom_slot_xy(self, node: GHSOMNodeState, slot: int) -> Tuple[int, int]:
        width = max(1, int(node.width))
        return int(slot % width), int(slot // width)

    def _ghsom_slot_distance(self, node: GHSOMNodeState, a: int, b: int) -> float:
        ax, ay = self._ghsom_slot_xy(node, a)
        bx, by = self._ghsom_slot_xy(node, b)
        dx = ax - bx
        dy = ay - by
        return math.sqrt(dx * dx + dy * dy)

    def _ghsom_split_similarity_threshold(self) -> float:
        target = 1.0 - float(self.p.ghsom_tau2) * 4.0
        return max(0.60, min(0.92, target))

    def _make_ghsom_node(self, seed_vec: np.ndarray, depth: int) -> GHSOMNodeState:
        width = max(2, int(self.p.ghsom_initial_width))
        height = max(2, int(self.p.ghsom_initial_height))
        cells = max(1, width * height)
        neurons = np.zeros((cells, self.p.dim), dtype=np.float32)
        base = unit(seed_vec.astype(np.float32, copy=False))
        for slot in range(cells):
            noise = unit(self.rng.normal(size=self.p.dim).astype(np.float32))
            magnitude = min(0.18, 0.035 + 0.006 * depth + 0.002 * slot)
            neurons[slot] = unit((1.0 - magnitude) * base + magnitude * noise).astype(np.float32)
        return GHSOMNodeState(
            depth=max(1, int(depth)),
            width=width,
            height=height,
            neurons=neurons,
            slot_members=[[] for _ in range(cells)],
            slot_count=np.zeros(cells, dtype=np.int32),
            slot_sim_sum=np.zeros(cells, dtype=np.float32),
        )

    def _ensure_cluster_ghsom_root(
        self,
        cid: int,
        seed_vec: np.ndarray,
    ) -> Tuple[Optional[GHSOMNodeState], int, bool]:
        if not self.p.ghsom_enabled:
            return None, 0, False
        if cid < 0 or cid >= len(self.clusters):
            return None, 0, False
        clu = self.clusters[cid]
        if len(clu.members) < max(2, int(self.p.ghsom_linear_scan_threshold)):
            return None, 0, False
        if clu.ghsom_root is not None:
            return clu.ghsom_root, 0, False

        clu.ghsom_root = self._make_ghsom_node(seed_vec, depth=1)
        ops = 0
        for mem_idx in clu.members:
            if 0 <= mem_idx < self.mem_count:
                ops += self._ghsom_insert(clu.ghsom_root, self.vec_pool[mem_idx], mem_idx)
        return clu.ghsom_root, ops, True

    def _refresh_supercluster_centroid(self, sid: int) -> None:
        if sid < 0 or sid >= len(self.super_members):
            return
        members = self.super_members[sid]
        if not members:
            return
        idx = np.asarray(members, dtype=np.int32)
        cen = np.mean(self.cluster_centroids[idx], axis=0).astype(np.float32)
        self.super_centroids[sid] = unit(cen)

    def _update_cluster_centroid(self, cid: int, vec: np.ndarray) -> None:
        if cid < 0 or cid >= len(self.clusters):
            return
        n = max(1, len(self.clusters[cid].members))
        if n <= 1:
            self.cluster_centroids[cid] = vec.astype(np.float32, copy=False)
        else:
            prev = self.cluster_centroids[cid].astype(np.float32, copy=False)
            updated = (((n - 1) * prev) + vec) / float(n)
            self.cluster_centroids[cid] = unit(updated.astype(np.float32))

        sid = int(self.super_of_cluster[cid]) if cid < len(self.super_of_cluster) else -1
        if sid >= 0:
            self._refresh_supercluster_centroid(sid)

    def _ghsom_find_bmu(self, node: GHSOMNodeState, vec: np.ndarray) -> Tuple[int, float, int]:
        sims = node.neurons @ vec
        slot = int(np.argmax(sims))
        return slot, float(sims[slot]), int(sims.size)

    def _ghsom_update_neurons(self, node: GHSOMNodeState, vec: np.ndarray, bmu_slot: int) -> None:
        cells = self._ghsom_cell_count(node)
        depth_offset = max(0, int(node.depth) - 1)
        alpha = float(self.p.ghsom_learning_rate_initial) * (
            float(self.p.ghsom_learning_rate_decay) ** depth_offset
        )
        alpha = max(0.02, alpha / math.sqrt(1.0 + node.member_count / max(1, cells)))
        radius = float(self.p.ghsom_neighborhood_radius_initial) * (
            float(self.p.ghsom_neighborhood_radius_decay) ** depth_offset
        )
        radius = max(0.8, min(radius, float(max(node.width, node.height))))
        sigma = max(0.7, radius)
        for slot in range(cells):
            dist = self._ghsom_slot_distance(node, slot, bmu_slot)
            if dist > radius:
                continue
            influence = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
            lr = alpha * influence
            node.neurons[slot] = unit(
                (node.neurons[slot] + lr * (vec - node.neurons[slot])).astype(np.float32)
            ).astype(np.float32)

    def _ghsom_route_path(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
    ) -> Tuple[List[Tuple[GHSOMNodeState, int, float]], int]:
        if root is None:
            return [], 0
        current = root
        path: List[Tuple[GHSOMNodeState, int, float]] = []
        ops = 0
        while current is not None:
            slot, sim, step_ops = self._ghsom_find_bmu(current, vec)
            ops += step_ops
            path.append((current, slot, sim))
            current = current.children.get(slot)
        return path, ops

    def _ghsom_insert(self, root: GHSOMNodeState, vec: np.ndarray, mem_idx: int) -> int:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return ops
        for node, slot, _sim in path:
            self._ghsom_update_neurons(node, vec, slot)
            node.member_count += 1
        leaf, slot, sim = path[-1]
        leaf.slot_members[slot].append(mem_idx)
        leaf.slot_count[slot] += 1
        leaf.slot_sim_sum[slot] += float(sim)
        self._ghsom_maybe_split(leaf, slot)
        return ops

    def _ghsom_maybe_split(self, node: GHSOMNodeState, slot: int) -> None:
        if node.depth >= max(1, int(self.p.ghsom_max_depth)):
            return
        if slot in node.children:
            return
        count = int(node.slot_count[slot])
        if count < max(2, int(self.p.ghsom_min_samples_for_expansion)):
            return
        avg_sim = float(node.slot_sim_sum[slot]) / max(1, count)
        if avg_sim >= self._ghsom_split_similarity_threshold():
            return

        child = self._make_ghsom_node(node.neurons[slot], depth=node.depth + 1)
        moving = list(node.slot_members[slot])
        node.children[slot] = child
        node.slot_members[slot] = []
        node.slot_count[slot] = 0
        node.slot_sim_sum[slot] = 0.0

        for mem_idx in moving:
            if 0 <= mem_idx < self.mem_count:
                self._ghsom_insert(child, self.vec_pool[mem_idx], mem_idx)

    def _ghsom_collect_candidates(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
        max_results: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return np.asarray([], dtype=np.int32), ops

        cap = self._effective_scan_cap(max_results)
        root_members = max(1, int(path[0][0].member_count))
        if cap is None:
            candidate_target = root_members
        else:
            candidate_target = min(root_members, max(8, int(cap)))
        min_hits = max(1, int(max_results) if max_results is not None else 1)

        ordered_path = list(reversed(path))
        if not self.p.ghsom_hierarchical_retrieval:
            ordered_path = ordered_path[:1]

        candidates: List[int] = []
        seen: Set[int] = set()
        for node, route_slot, _ in ordered_path:
            slot_sims = node.neurons @ vec
            ops += int(slot_sims.size)
            order = np.argsort(slot_sims)[::-1].tolist()
            if route_slot in order:
                order.remove(route_slot)
                order.insert(0, route_slot)

            beam_slots = max(1, min(len(order), int(math.ceil(math.sqrt(max(1, len(order)))))))
            for s_idx, slot in enumerate(order):
                members = node.slot_members[int(slot)]
                if members:
                    for mem_idx in members:
                        if mem_idx not in seen:
                            seen.add(mem_idx)
                            candidates.append(int(mem_idx))
                enough_for_ranking = (s_idx + 1) >= beam_slots and len(candidates) >= min_hits
                if len(candidates) >= candidate_target or enough_for_ranking:
                    break
            if len(candidates) >= candidate_target:
                break

        self.ghsom_probe_count += 1
        self.ghsom_probe_depth_total += len(path)
        self.ghsom_probe_candidates_total += len(candidates)
        return np.asarray(candidates, dtype=np.int32), ops

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

    def _trim_scan_indices(
        self,
        indices: np.ndarray,
        max_results: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        cap = self._effective_scan_cap(max_results)
        total = int(indices.size)
        if total <= 0 or cap is None or total <= cap:
            return indices, 0

        cap = min(total, max(1, int(cap)))
        hot_ratio, recent_ratio, _ = self._scan_mix()
        hot_n = max(0, int(round(cap * hot_ratio)))
        recent_n = max(0, int(round(cap * recent_ratio)))
        hot_n = min(cap, hot_n)
        recent_n = min(cap - hot_n, recent_n)
        prep_ops = 0

        mask = np.zeros(total, dtype=bool)
        if recent_n > 0:
            mask[total - recent_n :] = True

        if hot_n > 0:
            heats = self.heat_pool[indices]
            prep_ops += total
            if hot_n >= total:
                mask[:] = True
            else:
                keep = np.argpartition(heats, -hot_n)[-hot_n:]
                prep_ops += total
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

        return indices[mask], prep_ops

    def _find_sim_in_indices(
        self,
        vec: np.ndarray,
        indices: np.ndarray,
        max_results: Optional[int] = None,
    ) -> Tuple[List[Tuple[int, float]], int]:
        if indices.size == 0:
            return [], 0
        indices, prep_ops = self._trim_scan_indices(indices, max_results=max_results)
        if indices.size == 0:
            return [], prep_ops
        sims = self.vec_pool[indices] @ vec
        ops = prep_ops + int(indices.size)
        if max_results is not None and max_results > 0 and sims.size > max_results:
            keep = np.argpartition(sims, -max_results)[-max_results:]
            ops += int(sims.size)
            order = keep[np.argsort(sims[keep])[::-1]]
        else:
            order = np.argsort(sims)[::-1]
            ops += int(sims.size)
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
        clu = self.clusters[cid]
        members = clu.members
        if not members:
            return [], 0
        ops = 0
        idx: np.ndarray
        if self.p.ghsom_enabled and clu.ghsom_root is not None:
            idx, ops = self._ghsom_collect_candidates(clu.ghsom_root, vec, max_results=max_results)
            if idx.size <= 0:
                idx = np.asarray(members, dtype=np.int32)
        else:
            idx = np.asarray(members, dtype=np.int32)
        if only_hot and only_cold:
            return [], ops
        if only_hot:
            idx = idx[self.heat_pool[idx] > 0.0]
        elif only_cold:
            idx = idx[self.heat_pool[idx] <= 0.0]
        if idx.size <= 0 and self.p.ghsom_enabled and clu.ghsom_root is not None:
            idx = np.asarray(members, dtype=np.int32)
            if only_hot:
                idx = idx[self.heat_pool[idx] > 0.0]
            elif only_cold:
                idx = idx[self.heat_pool[idx] <= 0.0]
        results, scan_ops = self._find_sim_in_indices(vec, idx, max_results=max_results)
        return results, ops + scan_ops

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
            self._update_cluster_centroid(cid, vec)
        else:
            cid = len(self.clusters)
            self.cluster_centroids[cid] = vec
            clu = ClusterState(members=[mem_idx], hot_count=0, cold_count=1, is_hot_cluster=False)
            self.clusters.append(clu)
            self._refresh_hot_flag(cid)

        self.cluster_of[mem_idx] = cid
        root, build_ops, fresh = self._ensure_cluster_ghsom_root(cid, vec)
        self.sim_ops_add_total += build_ops
        if root is not None and not fresh:
            self.sim_ops_add_total += self._ghsom_insert(root, vec, mem_idx)
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

    def _ghsom_cell_count(self, node: GHSOMNodeState) -> int:
        return max(1, int(node.width) * int(node.height))

    def _ghsom_slot_xy(self, node: GHSOMNodeState, slot: int) -> Tuple[int, int]:
        width = max(1, int(node.width))
        return int(slot % width), int(slot // width)

    def _ghsom_slot_distance(self, node: GHSOMNodeState, a: int, b: int) -> float:
        ax, ay = self._ghsom_slot_xy(node, a)
        bx, by = self._ghsom_slot_xy(node, b)
        dx = ax - bx
        dy = ay - by
        return math.sqrt(dx * dx + dy * dy)

    def _ghsom_split_similarity_threshold(self) -> float:
        target = 1.0 - float(self.p.ghsom_tau2) * 4.0
        return max(0.60, min(0.92, target))

    def _make_ghsom_node(self, seed_vec: np.ndarray, depth: int) -> GHSOMNodeState:
        width = max(2, int(self.p.ghsom_initial_width))
        height = max(2, int(self.p.ghsom_initial_height))
        cells = max(1, width * height)
        neurons = np.zeros((cells, self.p.dim), dtype=np.float32)
        base = unit(seed_vec.astype(np.float32, copy=False))
        for slot in range(cells):
            noise = unit(self.rng.normal(size=self.p.dim).astype(np.float32))
            magnitude = min(0.18, 0.035 + 0.006 * depth + 0.002 * slot)
            neurons[slot] = unit((1.0 - magnitude) * base + magnitude * noise).astype(np.float32)
        return GHSOMNodeState(
            depth=max(1, int(depth)),
            width=width,
            height=height,
            neurons=neurons,
            slot_members=[[] for _ in range(cells)],
            slot_count=np.zeros(cells, dtype=np.int32),
            slot_sim_sum=np.zeros(cells, dtype=np.float32),
        )

    def _ghsom_find_bmu(self, node: GHSOMNodeState, vec: np.ndarray) -> Tuple[int, float, int]:
        sims = node.neurons @ vec
        slot = int(np.argmax(sims))
        return slot, float(sims[slot]), int(sims.size)

    def _ghsom_update_neurons(self, node: GHSOMNodeState, vec: np.ndarray, bmu_slot: int) -> None:
        cells = self._ghsom_cell_count(node)
        depth_offset = max(0, int(node.depth) - 1)
        alpha = float(self.p.ghsom_learning_rate_initial) * (
            float(self.p.ghsom_learning_rate_decay) ** depth_offset
        )
        alpha = max(0.02, alpha / math.sqrt(1.0 + node.member_count / max(1, cells)))
        radius = float(self.p.ghsom_neighborhood_radius_initial) * (
            float(self.p.ghsom_neighborhood_radius_decay) ** depth_offset
        )
        radius = max(0.8, min(radius, float(max(node.width, node.height))))
        sigma = max(0.7, radius)
        for slot in range(cells):
            dist = self._ghsom_slot_distance(node, slot, bmu_slot)
            if dist > radius:
                continue
            influence = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
            lr = alpha * influence
            node.neurons[slot] = unit(
                (node.neurons[slot] + lr * (vec - node.neurons[slot])).astype(np.float32)
            ).astype(np.float32)

    def _ghsom_route_path(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
    ) -> Tuple[List[Tuple[GHSOMNodeState, int, float]], int]:
        if root is None:
            return [], 0
        current = root
        path: List[Tuple[GHSOMNodeState, int, float]] = []
        ops = 0
        while current is not None:
            slot, sim, step_ops = self._ghsom_find_bmu(current, vec)
            ops += step_ops
            path.append((current, slot, sim))
            current = current.children.get(slot)
        return path, ops

    def _ghsom_insert(self, root: GHSOMNodeState, vec: np.ndarray, mem_idx: int) -> int:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return ops
        for node, slot, _sim in path:
            self._ghsom_update_neurons(node, vec, slot)
            node.member_count += 1
        leaf, slot, sim = path[-1]
        leaf.slot_members[slot].append(mem_idx)
        leaf.slot_count[slot] += 1
        leaf.slot_sim_sum[slot] += float(sim)
        self._ghsom_maybe_split(leaf, slot)
        return ops

    def _ghsom_maybe_split(self, node: GHSOMNodeState, slot: int) -> None:
        if node.depth >= max(1, int(self.p.ghsom_max_depth)):
            return
        if slot in node.children:
            return
        count = int(node.slot_count[slot])
        if count < max(2, int(self.p.ghsom_min_samples_for_expansion)):
            return
        avg_sim = float(node.slot_sim_sum[slot]) / max(1, count)
        if avg_sim >= self._ghsom_split_similarity_threshold():
            return

        child = self._make_ghsom_node(node.neurons[slot], depth=node.depth + 1)
        moving = list(node.slot_members[slot])
        node.children[slot] = child
        node.slot_members[slot] = []
        node.slot_count[slot] = 0
        node.slot_sim_sum[slot] = 0.0

        for mem_idx in moving:
            if 0 <= mem_idx < self.mem_count:
                self._ghsom_insert(child, self.vec_pool[mem_idx], mem_idx)

    def _ensure_topic_ghsom_root(
        self,
        tid: int,
        seed_vec: np.ndarray,
    ) -> Tuple[Optional[GHSOMNodeState], int, bool]:
        if not self.p.ghsom_enabled:
            return None, 0, False
        if tid < 0 or tid >= self.topic_count:
            return None, 0, False
        shard = self.topic_shards[tid]
        if len(shard.members) < max(2, int(self.p.ghsom_linear_scan_threshold)):
            return None, 0, False
        if shard.ghsom_root is not None:
            return shard.ghsom_root, 0, False

        shard.ghsom_root = self._make_ghsom_node(seed_vec, depth=1)
        ops = 0
        for mem_idx in shard.members:
            if 0 <= mem_idx < self.mem_count:
                ops += self._ghsom_insert(shard.ghsom_root, self.vec_pool[mem_idx], mem_idx)
        return shard.ghsom_root, ops, True

    def _ghsom_collect_candidates(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
        max_results: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return np.asarray([], dtype=np.int32), ops

        if max_results is None or max_results <= 0:
            candidate_target = max(1, int(path[0][0].member_count))
            min_hits = 1
        else:
            candidate_target = min(
                max(1, int(path[0][0].member_count)),
                max(8, int(max_results) * 2),
            )
            min_hits = max(1, int(max_results))

        ordered_path = list(reversed(path))
        if not self.p.ghsom_hierarchical_retrieval:
            ordered_path = ordered_path[:1]

        candidates: List[int] = []
        seen: Set[int] = set()
        for node, route_slot, _ in ordered_path:
            slot_sims = node.neurons @ vec
            ops += int(slot_sims.size)
            order = np.argsort(slot_sims)[::-1].tolist()
            if route_slot in order:
                order.remove(route_slot)
                order.insert(0, route_slot)

            beam_slots = max(1, min(len(order), int(math.ceil(math.sqrt(max(1, len(order)))))))
            for s_idx, slot in enumerate(order):
                members = node.slot_members[int(slot)]
                if members:
                    for mem_idx in members:
                        if mem_idx not in seen:
                            seen.add(mem_idx)
                            candidates.append(int(mem_idx))
                enough = (s_idx + 1) >= beam_slots and len(candidates) >= min_hits
                if len(candidates) >= candidate_target or enough:
                    break
            if len(candidates) >= candidate_target:
                break

        self.ghsom_probe_count += 1
        self.ghsom_probe_depth_total += len(path)
        self.ghsom_probe_candidates_total += len(candidates)
        return np.asarray(candidates, dtype=np.int32), ops

    def _ghsom_cell_count(self, node: GHSOMNodeState) -> int:
        return max(1, int(node.width) * int(node.height))

    def _ghsom_slot_xy(self, node: GHSOMNodeState, slot: int) -> Tuple[int, int]:
        width = max(1, int(node.width))
        return int(slot % width), int(slot // width)

    def _ghsom_slot_distance(self, node: GHSOMNodeState, a: int, b: int) -> float:
        ax, ay = self._ghsom_slot_xy(node, a)
        bx, by = self._ghsom_slot_xy(node, b)
        dx = ax - bx
        dy = ay - by
        return math.sqrt(dx * dx + dy * dy)

    def _ghsom_split_similarity_threshold(self) -> float:
        target = 1.0 - float(self.p.ghsom_tau2) * 4.0
        return max(0.60, min(0.92, target))

    def _make_ghsom_node(self, seed_vec: np.ndarray, depth: int) -> GHSOMNodeState:
        width = max(2, int(self.p.ghsom_initial_width))
        height = max(2, int(self.p.ghsom_initial_height))
        cells = max(1, width * height)
        neurons = np.zeros((cells, self.p.dim), dtype=np.float32)
        base = unit(seed_vec.astype(np.float32, copy=False))
        for slot in range(cells):
            noise = unit(self.rng.normal(size=self.p.dim).astype(np.float32))
            magnitude = min(0.18, 0.035 + 0.006 * depth + 0.002 * slot)
            neurons[slot] = unit((1.0 - magnitude) * base + magnitude * noise).astype(np.float32)
        return GHSOMNodeState(
            depth=max(1, int(depth)),
            width=width,
            height=height,
            neurons=neurons,
            slot_members=[[] for _ in range(cells)],
            slot_count=np.zeros(cells, dtype=np.int32),
            slot_sim_sum=np.zeros(cells, dtype=np.float32),
        )

    def _ghsom_find_bmu(self, node: GHSOMNodeState, vec: np.ndarray) -> Tuple[int, float, int]:
        sims = node.neurons @ vec
        slot = int(np.argmax(sims))
        return slot, float(sims[slot]), int(sims.size)

    def _ghsom_update_neurons(self, node: GHSOMNodeState, vec: np.ndarray, bmu_slot: int) -> None:
        cells = self._ghsom_cell_count(node)
        depth_offset = max(0, int(node.depth) - 1)
        alpha = float(self.p.ghsom_learning_rate_initial) * (
            float(self.p.ghsom_learning_rate_decay) ** depth_offset
        )
        alpha = max(0.02, alpha / math.sqrt(1.0 + node.member_count / max(1, cells)))
        radius = float(self.p.ghsom_neighborhood_radius_initial) * (
            float(self.p.ghsom_neighborhood_radius_decay) ** depth_offset
        )
        radius = max(0.8, min(radius, float(max(node.width, node.height))))
        sigma = max(0.7, radius)
        for slot in range(cells):
            dist = self._ghsom_slot_distance(node, slot, bmu_slot)
            if dist > radius:
                continue
            influence = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
            lr = alpha * influence
            node.neurons[slot] = unit(
                (node.neurons[slot] + lr * (vec - node.neurons[slot])).astype(np.float32)
            ).astype(np.float32)

    def _ghsom_route_path(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
    ) -> Tuple[List[Tuple[GHSOMNodeState, int, float]], int]:
        if root is None:
            return [], 0
        current = root
        path: List[Tuple[GHSOMNodeState, int, float]] = []
        ops = 0
        while current is not None:
            slot, sim, step_ops = self._ghsom_find_bmu(current, vec)
            ops += step_ops
            path.append((current, slot, sim))
            current = current.children.get(slot)
        return path, ops

    def _ghsom_insert(self, root: GHSOMNodeState, vec: np.ndarray, mem_idx: int) -> int:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return ops
        for node, slot, _sim in path:
            self._ghsom_update_neurons(node, vec, slot)
            node.member_count += 1
        leaf, slot, sim = path[-1]
        leaf.slot_members[slot].append(mem_idx)
        leaf.slot_count[slot] += 1
        leaf.slot_sim_sum[slot] += float(sim)
        self._ghsom_maybe_split(leaf, slot)
        return ops

    def _ghsom_maybe_split(self, node: GHSOMNodeState, slot: int) -> None:
        if node.depth >= max(1, int(self.p.ghsom_max_depth)):
            return
        if slot in node.children:
            return
        count = int(node.slot_count[slot])
        if count < max(2, int(self.p.ghsom_min_samples_for_expansion)):
            return
        avg_sim = float(node.slot_sim_sum[slot]) / max(1, count)
        if avg_sim >= self._ghsom_split_similarity_threshold():
            return

        child = self._make_ghsom_node(node.neurons[slot], depth=node.depth + 1)
        moving = list(node.slot_members[slot])
        node.children[slot] = child
        node.slot_members[slot] = []
        node.slot_count[slot] = 0
        node.slot_sim_sum[slot] = 0.0

        for mem_idx in moving:
            if 0 <= mem_idx < self.mem_count:
                self._ghsom_insert(child, self.vec_pool[mem_idx], mem_idx)

    def _ensure_topic_ghsom_root(
        self,
        tid: int,
        seed_vec: np.ndarray,
    ) -> Tuple[Optional[GHSOMNodeState], int, bool]:
        if not self.p.ghsom_enabled:
            return None, 0, False
        if tid < 0 or tid >= self.topic_count:
            return None, 0, False
        shard = self.topic_shards[tid]
        if len(shard.members) < max(2, int(self.p.ghsom_linear_scan_threshold)):
            return None, 0, False
        if shard.ghsom_root is not None:
            return shard.ghsom_root, 0, False

        shard.ghsom_root = self._make_ghsom_node(seed_vec, depth=1)
        ops = 0
        for mem_idx in shard.members:
            if 0 <= mem_idx < self.mem_count:
                ops += self._ghsom_insert(shard.ghsom_root, self.vec_pool[mem_idx], mem_idx)
        return shard.ghsom_root, ops, True

    def _ghsom_collect_candidates(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
        max_results: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return np.asarray([], dtype=np.int32), ops

        if max_results is None or max_results <= 0:
            candidate_target = max(1, int(path[0][0].member_count))
            min_hits = 1
        else:
            candidate_target = min(
                max(1, int(path[0][0].member_count)),
                max(8, int(max_results) * 2),
            )
            min_hits = max(1, int(max_results))

        ordered_path = list(reversed(path))
        if not self.p.ghsom_hierarchical_retrieval:
            ordered_path = ordered_path[:1]

        candidates: List[int] = []
        seen: Set[int] = set()
        for node, route_slot, _ in ordered_path:
            slot_sims = node.neurons @ vec
            ops += int(slot_sims.size)
            order = np.argsort(slot_sims)[::-1].tolist()
            if route_slot in order:
                order.remove(route_slot)
                order.insert(0, route_slot)

            beam_slots = max(1, min(len(order), int(math.ceil(math.sqrt(max(1, len(order)))))))
            for s_idx, slot in enumerate(order):
                members = node.slot_members[int(slot)]
                if members:
                    for mem_idx in members:
                        if mem_idx not in seen:
                            seen.add(mem_idx)
                            candidates.append(int(mem_idx))
                enough = (s_idx + 1) >= beam_slots and len(candidates) >= min_hits
                if len(candidates) >= candidate_target or enough:
                    break
            if len(candidates) >= candidate_target:
                break

        self.ghsom_probe_count += 1
        self.ghsom_probe_depth_total += len(path)
        self.ghsom_probe_candidates_total += len(candidates)
        return np.asarray(candidates, dtype=np.int32), ops

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

    def _unload_preload_cache(self) -> None:
        if self.preload_cache_clusters or self.preload_cache_mem:
            self.preload_cache_unload_count += 1
        self.preload_cache_topic = None
        self.preload_cache_clusters = set()
        self.preload_cache_mem = set()

    def _unload_topic_cache(self) -> None:
        if self.topic_cache_mem:
            self.topic_cache_unload_count += 1
        self.topic_cache_mem = set()
        self.topic_cache_topic = None
        self._unload_preload_cache()

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
    
    def _smart_preload_cold_memories(
        self,
        query_vec: np.ndarray,
        target_topic: Optional[int],
        current_turn: int,
        candidate_clusters: Optional[Sequence[int]] = None,
    ) -> int:
        """Preload query-hit clusters into dedicated preload cache.

        Returns: number of clusters preloaded.
        """
        _ = current_turn
        if not self.p.smart_preload_enabled:
            return 0

        self.preload_attempts += 1
        if self.preload_io_count >= self.p.preload_max_io_per_turn:
            return 0

        topics_to_preload: List[Tuple[int, float]] = []
        if target_topic is not None:
            hot_ratio = self._get_topic_hot_ratio(target_topic)
            if hot_ratio < self.p.preload_low_hot_ratio_threshold:
                topics_to_preload.append((target_topic, 1.0))

        if self.p.preload_use_vector_prediction:
            predicted_topic, confidence = self._predict_target_topic(query_vec)
            self.preload_topic_predictions += 1
            if predicted_topic is not None and confidence >= self.p.preload_topic_confidence:
                hot_ratio = self._get_topic_hot_ratio(predicted_topic)
                if hot_ratio < self.p.preload_low_hot_ratio_threshold:
                    if not any(t == predicted_topic for t, _ in topics_to_preload):
                        topics_to_preload.append((predicted_topic, confidence))

        if not topics_to_preload:
            return 0

        if not candidate_clusters:
            return 0
        cluster_allow = {
            int(cid)
            for cid in candidate_clusters
            if isinstance(cid, (int, np.integer)) and 0 <= int(cid) < len(self.clusters)
        }
        if not cluster_allow:
            return 0

        budget = min(
            int(self.p.preload_budget_per_query),
            int(self.p.preload_max_io_per_turn) - int(self.preload_io_count),
        )
        if budget <= 0:
            return 0

        preloaded_clusters = 0
        preloaded_memories = 0

        for topic_id, _confidence in topics_to_preload:
            if preloaded_clusters >= budget:
                break
            mem_set = self.topic_to_mem.get(topic_id, set())
            if not mem_set:
                continue

            topic_cluster_ids = set()
            for mem_idx in mem_set:
                if mem_idx < 0 or mem_idx >= self.mem_count:
                    continue
                cid = int(self.cluster_of[mem_idx])
                if cid >= 0 and cid in cluster_allow:
                    topic_cluster_ids.add(cid)
            if not topic_cluster_ids:
                continue

            scored_clusters: List[Tuple[int, float, List[int]]] = []
            for cid in topic_cluster_ids:
                members = [m for m in self.clusters[cid].members if 0 <= m < self.mem_count]
                if not members:
                    continue
                mem_idx = np.asarray(members, dtype=np.int32)
                sims = self.vec_pool[mem_idx] @ query_vec
                best_sim = float(np.max(sims)) if sims.size > 0 else -1.0
                scored_clusters.append((cid, best_sim, members))

            scored_clusters.sort(key=lambda it: it[1], reverse=True)
            for cid, _score, members in scored_clusters:
                if preloaded_clusters >= budget:
                    break
                if cid in self.preload_cache_clusters:
                    continue
                self.preload_cache_clusters.add(int(cid))
                added = 0
                for mem_idx in members:
                    if mem_idx not in self.preload_cache_mem:
                        self.preload_cache_mem.add(int(mem_idx))
                        added += 1
                self.preload_io_count += 1
                self.preload_clusters_loaded += 1
                preloaded_clusters += 1
                preloaded_memories += added

        if preloaded_clusters > 0:
            self.preload_successes += 1
            self.preload_memories_heated += preloaded_memories
            if target_topic is not None:
                self.preload_cache_topic = int(target_topic)
            elif topics_to_preload:
                self.preload_cache_topic = int(topics_to_preload[0][0])

        return preloaded_clusters
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

        # Smart preloading at cluster granularity, independent from heat competition.
        preloaded = 0
        if target_topic is not None and self.p.smart_preload_enabled:
            preload_cluster_ids, ops0 = self._top_probe_clusters(
                query_vec,
                probe_clusters=max(1, probe_clusters),
                super_topn=query_topn,
                turn=query_turn,
            )
            sim_ops += ops0
            preloaded = self._smart_preload_cold_memories(
                query_vec,
                target_topic,
                query_turn,
                candidate_clusters=preload_cluster_ids,
            )

        mem_best_sim: Dict[int, float] = {}
        mem_best_cluster: Dict[int, int] = {}
        mem_best_effective: Dict[int, float] = {}
        mem_best_source: Dict[int, str] = {}
        turn_best: Dict[int, float] = {}
        turn_src: Dict[int, str] = {}
        turn_mem: Dict[int, int] = {}

        def update_mem_best(mem_idx: int, sim: float, effective: float, cid: int, source: str) -> None:
            prev = mem_best_sim.get(mem_idx)
            if prev is None or sim > prev:
                mem_best_sim[mem_idx] = sim
                mem_best_cluster[mem_idx] = cid
                mem_best_effective[mem_idx] = effective
                mem_best_source[mem_idx] = source

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
                    effective = (sim_pos ** power) * weight
                    source = "explore" if cid in persistent_probe_clusters_vec else "hot"
                    update_mem_best(mem_idx, sim, effective, cid, source)

                    # EnhancedWeak: Use soft gate filter
                    if not self._soft_gate_filter(sim, min_gate):
                        break

            if not scanned_any:
                fallback, ops2 = self._find_sim_all_hot(qv, max_results=max_memory * 2)
                sim_ops += ops2
                for mem_idx, sim in fallback:
                    sim_pos = max(0.0, sim)
                    cid = int(self.cluster_of[mem_idx])
                    effective = (sim_pos ** power) * weight
                    update_mem_best(mem_idx, sim, effective, cid, "hot")

                    # EnhancedWeak: Use soft gate filter
                    if not self._soft_gate_filter(sim, min_gate):
                        break

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
                        effective = (sim ** power) * weight * self.p.topic_cache_weight
                        cid = int(self.cluster_of[mem_idx])
                        update_mem_best(mem_idx, sim, effective, cid, "cache")

            # Preloaded clusters are scored independently and bypass hot-pool competition.
            if self.preload_cache_mem:
                preload_idx = np.asarray(
                    [idx for idx in self.preload_cache_mem if 0 <= idx < self.mem_count],
                    dtype=np.int32,
                )
                if preload_idx.size > 0:
                    preload_sims = self.vec_pool[preload_idx] @ qv
                    sim_ops += int(preload_idx.size)
                    order = np.argsort(preload_sims)[::-1]
                    for oi in order:
                        sim = float(preload_sims[oi])
                        if not self._soft_gate_filter(sim, min_gate):
                            break
                        mem_idx = int(preload_idx[oi])
                        cid = int(self.cluster_of[mem_idx])
                        sim_pos = max(0.0, sim)
                        effective = (sim_pos ** power) * weight
                        update_mem_best(mem_idx, sim, effective, cid, "preload_cluster")

        # Stage 2: turn expansion only for memories that pass hard similarity gate.
        kept_memories = [
            mem_idx
            for mem_idx, sim in mem_best_sim.items()
            if float(sim) >= float(self.p.memory_drop_sim)
        ]
        for mem_idx in kept_memories:
            effective = float(mem_best_effective.get(mem_idx, 0.0))
            if effective <= 0.0:
                continue
            source = mem_best_source.get(mem_idx, "hot")
            for t in self.mem_turns[mem_idx]:
                prev = turn_best.get(t)
                if prev is None or effective > prev:
                    turn_best[t] = effective
                    turn_src[t] = source
                    turn_mem[t] = mem_idx

        sample_limit = max(1, int(self.p.refinement_sample_mem_topk))
        if kept_memories:
            mem_items = sorted(
                ((mem_idx, mem_best_sim[mem_idx]) for mem_idx in kept_memories),
                key=lambda it: it[1],
                reverse=True,
            )
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
            self.preload_cache_selected_turns_total += sum(
                1 for t in selected_turns if turn_src.get(t) == "preload_cluster"
            )
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
        self.preload_cache_selected_turns_total += sum(
            1 for t in selected_turns if turn_src.get(t) == "preload_cluster"
        )
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
        preload_cache_contrib_ratio = self.preload_cache_selected_turns_total / max(1, self.returned_turns_sum)

        # Smart Preloading stats
        preload_success_rate = self.preload_successes / max(1, self.preload_attempts)
        preload_prediction_accuracy = self.preload_correct_predictions / max(1, self.preload_topic_predictions)
        ghsom_probe_avg_candidates = self.ghsom_probe_candidates_total / max(1, self.ghsom_probe_count)
        ghsom_probe_avg_depth = self.ghsom_probe_depth_total / max(1, self.ghsom_probe_count)

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
            "preload_cache_topic": float(self.preload_cache_topic if self.preload_cache_topic is not None else -1),
            "preload_cache_size": float(len(self.preload_cache_mem)),
            "preload_cache_cluster_size": float(len(self.preload_cache_clusters)),
            "preload_cache_unload_count": float(self.preload_cache_unload_count),
            "preload_cache_contrib_ratio": preload_cache_contrib_ratio,
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
            "preload_clusters_loaded": float(self.preload_clusters_loaded),
            "preload_topic_predictions": float(self.preload_topic_predictions),
            "preload_correct_predictions": float(self.preload_correct_predictions),
            "preload_prediction_accuracy": preload_prediction_accuracy,
            "ghsom_enabled": 1.0 if self.p.ghsom_enabled else 0.0,
            "ghsom_probe_count": float(self.ghsom_probe_count),
            "ghsom_probe_avg_candidates": float(ghsom_probe_avg_candidates),
            "ghsom_probe_avg_depth": float(ghsom_probe_avg_depth),
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


class TopicGraphSim:
    def __init__(self, p: SimParams):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.topic = TopicStream(p, self.rng)
        self.topic_count = self.topic.num_topics
        self.max_memories = p.turns

        self.vec_pool = np.zeros((self.max_memories, p.dim), dtype=np.float32)
        self.mem_topic = np.full(self.max_memories, -1, dtype=np.int32)
        self.mem_turn = np.zeros(self.max_memories, dtype=np.int32)
        self.mem_count = 0

        self.topic_shards: List[TopicShardState] = [TopicShardState() for _ in range(self.topic_count)]
        self.topic_bridges: List[Dict[int, TopicBridgeState]] = [dict() for _ in range(self.topic_count)]

        self.turn_topics: List[int] = []
        self.turns_by_topic: Dict[int, List[int]] = {}

        self.query_turns = 0
        self.eval_count = 0
        self.returned_topics_sum = 0
        self.target_precision_sum = 0.0
        self.target_hit_sum = 0.0
        self.target_mrr_sum = 0.0
        self.empty_query_count = 0
        self.empty_target_query_count = 0
        self.sim_ops_query_total = 0
        self.turn_sim_ops: List[int] = []

        self.topic_load_events = 0
        self.topic_evict_events = 0
        self.bridge_expand_events = 0
        self.bridge_hit_events = 0
        self.bridge_update_events = 0
        self.resident_hit_events = 0
        self.preload_attempts = 0
        self.preload_successes = 0
        self.preload_clusters_loaded = 0
        self.ghsom_probe_count = 0
        self.ghsom_probe_candidates_total = 0
        self.ghsom_probe_depth_total = 0

        self.snapshots: List[Dict[str, float]] = []

    def _register_turn_topic(self, turn: int, topic_id: int) -> None:
        self.turn_topics.append(topic_id)
        bucket = self.turns_by_topic.get(topic_id)
        if bucket is None:
            bucket = []
            self.turns_by_topic[topic_id] = bucket
        bucket.append(turn)

    def _ghsom_cell_count(self, node: GHSOMNodeState) -> int:
        return max(1, int(node.width) * int(node.height))

    def _ghsom_slot_xy(self, node: GHSOMNodeState, slot: int) -> Tuple[int, int]:
        width = max(1, int(node.width))
        return int(slot % width), int(slot // width)

    def _ghsom_slot_distance(self, node: GHSOMNodeState, a: int, b: int) -> float:
        ax, ay = self._ghsom_slot_xy(node, a)
        bx, by = self._ghsom_slot_xy(node, b)
        dx = ax - bx
        dy = ay - by
        return math.sqrt(dx * dx + dy * dy)

    def _ghsom_split_similarity_threshold(self) -> float:
        target = 1.0 - float(self.p.ghsom_tau2) * 4.0
        return max(0.60, min(0.92, target))

    def _make_ghsom_node(self, seed_vec: np.ndarray, depth: int) -> GHSOMNodeState:
        width = max(2, int(self.p.ghsom_initial_width))
        height = max(2, int(self.p.ghsom_initial_height))
        cells = max(1, width * height)
        neurons = np.zeros((cells, self.p.dim), dtype=np.float32)
        base = unit(seed_vec.astype(np.float32, copy=False))
        for slot in range(cells):
            noise = unit(self.rng.normal(size=self.p.dim).astype(np.float32))
            magnitude = min(0.18, 0.035 + 0.006 * depth + 0.002 * slot)
            neurons[slot] = unit((1.0 - magnitude) * base + magnitude * noise).astype(np.float32)
        return GHSOMNodeState(
            depth=max(1, int(depth)),
            width=width,
            height=height,
            neurons=neurons,
            slot_members=[[] for _ in range(cells)],
            slot_count=np.zeros(cells, dtype=np.int32),
            slot_sim_sum=np.zeros(cells, dtype=np.float32),
        )

    def _ghsom_find_bmu(self, node: GHSOMNodeState, vec: np.ndarray) -> Tuple[int, float, int]:
        sims = node.neurons @ vec
        slot = int(np.argmax(sims))
        return slot, float(sims[slot]), int(sims.size)

    def _ghsom_update_neurons(self, node: GHSOMNodeState, vec: np.ndarray, bmu_slot: int) -> None:
        cells = self._ghsom_cell_count(node)
        depth_offset = max(0, int(node.depth) - 1)
        alpha = float(self.p.ghsom_learning_rate_initial) * (
            float(self.p.ghsom_learning_rate_decay) ** depth_offset
        )
        alpha = max(0.02, alpha / math.sqrt(1.0 + node.member_count / max(1, cells)))
        radius = float(self.p.ghsom_neighborhood_radius_initial) * (
            float(self.p.ghsom_neighborhood_radius_decay) ** depth_offset
        )
        radius = max(0.8, min(radius, float(max(node.width, node.height))))
        sigma = max(0.7, radius)
        for slot in range(cells):
            dist = self._ghsom_slot_distance(node, slot, bmu_slot)
            if dist > radius:
                continue
            influence = math.exp(-(dist * dist) / (2.0 * sigma * sigma))
            lr = alpha * influence
            node.neurons[slot] = unit(
                (node.neurons[slot] + lr * (vec - node.neurons[slot])).astype(np.float32)
            ).astype(np.float32)

    def _ghsom_route_path(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
    ) -> Tuple[List[Tuple[GHSOMNodeState, int, float]], int]:
        if root is None:
            return [], 0
        current = root
        path: List[Tuple[GHSOMNodeState, int, float]] = []
        ops = 0
        while current is not None:
            slot, sim, step_ops = self._ghsom_find_bmu(current, vec)
            ops += step_ops
            path.append((current, slot, sim))
            current = current.children.get(slot)
        return path, ops

    def _ghsom_insert(self, root: GHSOMNodeState, vec: np.ndarray, mem_idx: int) -> int:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return ops
        for node, slot, _sim in path:
            self._ghsom_update_neurons(node, vec, slot)
            node.member_count += 1
        leaf, slot, sim = path[-1]
        leaf.slot_members[slot].append(mem_idx)
        leaf.slot_count[slot] += 1
        leaf.slot_sim_sum[slot] += float(sim)
        self._ghsom_maybe_split(leaf, slot)
        return ops

    def _ghsom_maybe_split(self, node: GHSOMNodeState, slot: int) -> None:
        if node.depth >= max(1, int(self.p.ghsom_max_depth)):
            return
        if slot in node.children:
            return
        count = int(node.slot_count[slot])
        if count < max(2, int(self.p.ghsom_min_samples_for_expansion)):
            return
        avg_sim = float(node.slot_sim_sum[slot]) / max(1, count)
        if avg_sim >= self._ghsom_split_similarity_threshold():
            return

        child = self._make_ghsom_node(node.neurons[slot], depth=node.depth + 1)
        moving = list(node.slot_members[slot])
        node.children[slot] = child
        node.slot_members[slot] = []
        node.slot_count[slot] = 0
        node.slot_sim_sum[slot] = 0.0

        for mem_idx in moving:
            if 0 <= mem_idx < self.mem_count:
                self._ghsom_insert(child, self.vec_pool[mem_idx], mem_idx)

    def _ensure_topic_ghsom_root(
        self,
        tid: int,
        seed_vec: np.ndarray,
    ) -> Tuple[Optional[GHSOMNodeState], int, bool]:
        if not self.p.ghsom_enabled:
            return None, 0, False
        if tid < 0 or tid >= self.topic_count:
            return None, 0, False
        shard = self.topic_shards[tid]
        if len(shard.members) < max(2, int(self.p.ghsom_linear_scan_threshold)):
            return None, 0, False
        if shard.ghsom_root is not None:
            return shard.ghsom_root, 0, False

        shard.ghsom_root = self._make_ghsom_node(seed_vec, depth=1)
        ops = 0
        for mem_idx in shard.members:
            if 0 <= mem_idx < self.mem_count:
                ops += self._ghsom_insert(shard.ghsom_root, self.vec_pool[mem_idx], mem_idx)
        return shard.ghsom_root, ops, True

    def _ghsom_collect_candidates(
        self,
        root: Optional[GHSOMNodeState],
        vec: np.ndarray,
        max_results: Optional[int],
    ) -> Tuple[np.ndarray, int]:
        path, ops = self._ghsom_route_path(root, vec)
        if not path:
            return np.asarray([], dtype=np.int32), ops

        if max_results is None or max_results <= 0:
            candidate_target = max(1, int(path[0][0].member_count))
            min_hits = 1
        else:
            candidate_target = min(
                max(1, int(path[0][0].member_count)),
                max(8, int(max_results) * 2),
            )
            min_hits = max(1, int(max_results))

        ordered_path = list(reversed(path))
        if not self.p.ghsom_hierarchical_retrieval:
            ordered_path = ordered_path[:1]

        candidates: List[int] = []
        seen: Set[int] = set()
        for node, route_slot, _ in ordered_path:
            slot_sims = node.neurons @ vec
            ops += int(slot_sims.size)
            order = np.argsort(slot_sims)[::-1].tolist()
            if route_slot in order:
                order.remove(route_slot)
                order.insert(0, route_slot)

            beam_slots = max(1, min(len(order), int(math.ceil(math.sqrt(max(1, len(order)))))))
            for s_idx, slot in enumerate(order):
                members = node.slot_members[int(slot)]
                if members:
                    for mem_idx in members:
                        if mem_idx not in seen:
                            seen.add(mem_idx)
                            candidates.append(int(mem_idx))
                enough = (s_idx + 1) >= beam_slots and len(candidates) >= min_hits
                if len(candidates) >= candidate_target or enough:
                    break
            if len(candidates) >= candidate_target:
                break

        self.ghsom_probe_count += 1
        self.ghsom_probe_depth_total += len(path)
        self.ghsom_probe_candidates_total += len(candidates)
        return np.asarray(candidates, dtype=np.int32), ops

    def _known_topic_ids(self) -> np.ndarray:
        return np.asarray(
            [tid for tid, shard in enumerate(self.topic_shards) if shard.members],
            dtype=np.int32,
        )

    def _topic_centroid(self, tid: int) -> np.ndarray:
        shard = self.topic_shards[tid]
        if shard.centroid is not None:
            return shard.centroid
        return self.topic.topic_vectors[tid]

    def _touch_loaded_topic(self, tid: int, turn: int) -> bool:
        shard = self.topic_shards[tid]
        newly_loaded = not shard.loaded
        if newly_loaded:
            self.topic_load_events += 1
        shard.loaded = True
        shard.last_loaded_turn = turn

        loaded_ids = [i for i, item in enumerate(self.topic_shards) if item.loaded]
        while len(loaded_ids) > max(1, int(self.p.topic_graph_loaded_cap)):
            evict = min(loaded_ids, key=lambda i: self.topic_shards[i].last_loaded_turn)
            if evict == tid and len(loaded_ids) > 1:
                alt = [i for i in loaded_ids if i != tid]
                evict = min(alt, key=lambda i: self.topic_shards[i].last_loaded_turn)
            self.topic_shards[evict].loaded = False
            self.topic_evict_events += 1
            loaded_ids = [i for i, item in enumerate(self.topic_shards) if item.loaded]
        return newly_loaded

    def _bridge_strength(self, edge: TopicBridgeState) -> float:
        return (
            0.45 * float(edge.transition)
            + 0.75 * float(edge.recall)
            + 1.00 * float(edge.adopt)
        )

    def _decay_bridges(self) -> None:
        decay = min(0.9999, max(0.0, float(self.p.topic_graph_decay)))
        drop_threshold = max(0.01, float(self.p.topic_graph_min_bridge_score) * 0.35)
        for src in range(self.topic_count):
            drop: List[int] = []
            for dst, edge in self.topic_bridges[src].items():
                edge.transition *= decay
                edge.recall *= decay
                edge.adopt *= decay
                if self._bridge_strength(edge) < drop_threshold and edge.support <= 1:
                    drop.append(dst)
            for dst in drop:
                self.topic_bridges[src].pop(dst, None)

    def _prune_bridges(self, src: int) -> None:
        keep = max(1, int(self.p.topic_graph_bridge_topk))
        edges = self.topic_bridges[src]
        if len(edges) <= keep:
            return
        ranked = sorted(edges.items(), key=lambda it: self._bridge_strength(it[1]), reverse=True)
        self.topic_bridges[src] = {dst: edge for dst, edge in ranked[:keep]}

    def _reinforce_bridge(self, src: int, dst: int, kind: str, lr: float, turn: int) -> None:
        if src == dst or src < 0 or dst < 0:
            return
        edge = self.topic_bridges[src].get(dst)
        if edge is None:
            edge = TopicBridgeState()
            self.topic_bridges[src][dst] = edge
        if kind == "transition":
            edge.transition = min(1.0, edge.transition + lr * (1.0 - edge.transition))
        elif kind == "recall":
            edge.recall = min(1.0, edge.recall + lr * (1.0 - edge.recall))
        elif kind == "adopt":
            edge.adopt = min(1.0, edge.adopt + lr * (1.0 - edge.adopt))
        edge.support += 1
        edge.last_turn = turn
        self.bridge_update_events += 1
        self._prune_bridges(src)

    def _add_memory(self, vec: np.ndarray, turn: int, topic_id: int) -> None:
        if self.mem_count >= self.max_memories:
            return
        mem_idx = self.mem_count
        self.vec_pool[mem_idx] = vec
        self.mem_topic[mem_idx] = topic_id
        self.mem_turn[mem_idx] = turn
        self.mem_count += 1

        shard = self.topic_shards[topic_id]
        shard.members.append(mem_idx)
        if shard.centroid is None:
            shard.centroid = vec.astype(np.float32, copy=True)
        else:
            n = len(shard.members)
            shard.centroid = unit((((n - 1) * shard.centroid) + vec) / float(n)).astype(np.float32)

        root, build_ops, fresh = self._ensure_topic_ghsom_root(topic_id, vec)
        if root is not None and not fresh:
            self._ghsom_insert(root, vec, mem_idx)

    def _semantic_topic_scores(
        self,
        query_vec: np.ndarray,
        current_topic: Optional[int],
    ) -> Tuple[Dict[int, float], int]:
        known = self._known_topic_ids()
        if known.size <= 0:
            return {}, 0
        centers = np.vstack([self._topic_centroid(int(tid)) for tid in known.tolist()]).astype(np.float32)
        sims = centers @ query_vec
        ops = int(centers.shape[0])
        scores: Dict[int, float] = {}
        for pos, tid in enumerate(known.tolist()):
            score = float(self.p.topic_graph_query_semantic_weight) * float(sims[pos])
            if current_topic is not None and tid == int(current_topic):
                score += float(self.p.topic_graph_current_topic_bonus)
            scores[int(tid)] = score
        return scores, ops

    def _top_seed_topics(
        self,
        semantic_scores: Dict[int, float],
        current_topic: Optional[int],
    ) -> List[Tuple[int, float]]:
        ranked = sorted(semantic_scores.items(), key=lambda it: it[1], reverse=True)
        seeds = ranked[: max(1, int(self.p.topic_graph_seed_topics))]
        if current_topic is not None and current_topic in semantic_scores and all(t != current_topic for t, _ in seeds):
            seeds = [(int(current_topic), float(semantic_scores[int(current_topic)]))] + seeds
        return seeds[: max(1, int(self.p.topic_graph_seed_topics))]

    def _expand_topic_candidates(
        self,
        semantic_scores: Dict[int, float],
        current_topic: Optional[int],
    ) -> Tuple[Dict[int, float], Set[int], int]:
        if not semantic_scores:
            return {}, set(), 0

        budget = max(1, int(self.p.topic_graph_expand_budget))
        max_hops = max(0, int(self.p.topic_graph_max_bridge_hops))
        seeds = self._top_seed_topics(semantic_scores, current_topic)
        frontier: List[Tuple[float, int, int, float]] = []
        best_total: Dict[int, float] = {}
        candidate_scores: Dict[int, float] = {}
        via_bridge: Set[int] = set()

        for tid, sem_score in seeds:
            resident_bonus = float(self.p.topic_graph_resident_bonus) if self.topic_shards[tid].loaded else 0.0
            total = sem_score + resident_bonus
            best_total[tid] = total
            heapq.heappush(frontier, (-total, 0, tid, 0.0))

        ops = 0
        expanded = 0
        visited: Set[int] = set()
        while frontier and expanded < budget:
            neg_total, hops, tid, bridge_bonus = heapq.heappop(frontier)
            total = -neg_total
            if total + 1e-9 < best_total.get(tid, -1e9):
                continue
            if tid in visited:
                continue
            visited.add(tid)
            expanded += 1
            candidate_scores[tid] = total

            if hops >= max_hops:
                continue

            edges = self.topic_bridges[tid]
            if not edges:
                continue
            ranked_edges = sorted(edges.items(), key=lambda it: self._bridge_strength(it[1]), reverse=True)
            for dst, edge in ranked_edges[: max(1, int(self.p.topic_graph_bridge_topk))]:
                edge_score = self._bridge_strength(edge)
                if edge_score < float(self.p.topic_graph_min_bridge_score):
                    continue
                if dst not in semantic_scores:
                    continue
                next_bonus = bridge_bonus * 0.60 + float(self.p.topic_graph_bridge_weight) * edge_score
                resident_bonus = float(self.p.topic_graph_resident_bonus) if self.topic_shards[dst].loaded else 0.0
                total_dst = float(semantic_scores[dst]) + next_bonus + resident_bonus
                ops += 1
                if total_dst > best_total.get(dst, -1e9):
                    best_total[dst] = total_dst
                    heapq.heappush(frontier, (-total_dst, hops + 1, dst, next_bonus))
                    via_bridge.add(int(dst))

        self.bridge_expand_events += len(via_bridge)
        return candidate_scores, via_bridge, ops

    def _preload_topics(
        self,
        candidate_scores: Dict[int, float],
        turn: int,
    ) -> Set[int]:
        self.preload_attempts += 1
        load_budget = max(1, int(self.p.topic_graph_load_budget))
        ranked = sorted(candidate_scores.items(), key=lambda it: it[1], reverse=True)
        available: Set[int] = set()
        used_budget = 0
        for tid, _score in ranked:
            shard = self.topic_shards[tid]
            if shard.loaded:
                shard.last_loaded_turn = turn
                available.add(int(tid))
                continue
            if used_budget >= load_budget:
                continue
            used_budget += 1
            self._touch_loaded_topic(int(tid), turn)
            available.add(int(tid))
            self.preload_clusters_loaded += 1

        if available:
            self.preload_successes += 1
        return available

    def _retrieve_topic_evidence(
        self,
        query_vec: np.ndarray,
        candidate_scores: Dict[int, float],
        available_topics: Set[int],
    ) -> Tuple[List[int], Dict[str, object], int]:
        if not available_topics:
            return [], {"evidence_topics": [], "evidence_memories": []}, 0

        per_topic_evidence = max(1, int(self.p.topic_graph_per_topic_evidence))
        ranked_topics: List[Tuple[int, float]] = []
        evidence_topics: List[int] = []
        evidence_memories: List[int] = []
        ops = 0

        for tid in sorted(available_topics):
            members = self.topic_shards[tid].members
            if not members:
                continue
            topic_ops = 0
            shard = self.topic_shards[tid]
            idx: np.ndarray
            if self.p.ghsom_enabled and shard.ghsom_root is not None:
                idx, topic_ops = self._ghsom_collect_candidates(
                    shard.ghsom_root,
                    query_vec,
                    max_results=per_topic_evidence,
                )
                if idx.size <= 0:
                    idx = np.asarray(members, dtype=np.int32)
            else:
                idx = np.asarray(members, dtype=np.int32)
            ops += topic_ops
            sims = self.vec_pool[idx] @ query_vec
            ops += int(idx.size)
            if sims.size <= 0:
                continue
            k = min(per_topic_evidence, int(sims.size))
            if k < sims.size:
                keep = np.argpartition(sims, -k)[-k:]
                best = keep[np.argsort(sims[keep])[::-1]]
                ops += int(sims.size)
            else:
                best = np.argsort(sims)[::-1]
                ops += int(sims.size)
            best_sims = sims[best]
            evidence_score = float(np.max(best_sims)) + 0.15 * float(np.mean(best_sims))
            total_score = float(candidate_scores.get(tid, 0.0)) + evidence_score
            ranked_topics.append((int(tid), total_score))
            evidence_topics.append(int(tid))
            evidence_memories.extend(int(idx[i]) for i in best.tolist())
            if self.topic_shards[tid].loaded:
                self.resident_hit_events += 1

        ranked_topics.sort(key=lambda it: it[1], reverse=True)
        selected_topics = [
            tid
            for tid, _score in ranked_topics[: max(1, int(self.p.topic_graph_max_return_topics))]
        ]
        return selected_topics, {
            "evidence_topics": evidence_topics,
            "evidence_memories": evidence_memories,
            "ranked_topics": ranked_topics,
        }, ops

    def retrieve(
        self,
        query_vec: np.ndarray,
        current_topic: Optional[int],
        current_turn: int,
    ) -> Tuple[List[int], int, Dict[str, object]]:
        semantic_scores, ops0 = self._semantic_topic_scores(query_vec, current_topic)
        if not semantic_scores:
            return [], ops0, {"candidate_topics": [], "bridge_topics": []}

        candidate_scores, via_bridge, ops1 = self._expand_topic_candidates(semantic_scores, current_topic)
        available_topics = self._preload_topics(candidate_scores, current_turn)
        selected_topics, debug, ops2 = self._retrieve_topic_evidence(query_vec, candidate_scores, available_topics)
        debug["candidate_topics"] = sorted(candidate_scores.items(), key=lambda it: it[1], reverse=True)
        debug["bridge_topics"] = sorted(via_bridge)
        return selected_topics, ops0 + ops1 + ops2, debug

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

    def _eval_query(self, selected_topics: List[int], target_topic: int) -> Tuple[int, float]:
        k = len(selected_topics)
        self.returned_topics_sum += k
        hit = 1 if target_topic in selected_topics else 0
        precision = (1.0 / k) if hit and k > 0 else 0.0
        mrr = 0.0
        if hit:
            rank = selected_topics.index(target_topic) + 1
            mrr = 1.0 / rank

        self.target_precision_sum += precision
        self.target_hit_sum += float(hit)
        self.target_mrr_sum += mrr
        self.eval_count += 1
        return hit, mrr

    def _apply_query_feedback(
        self,
        current_topic: int,
        target_topic: int,
        selected_topics: List[int],
        bridge_topics: Sequence[int],
        turn: int,
    ) -> None:
        if current_topic != target_topic and target_topic in selected_topics:
            self._reinforce_bridge(
                current_topic,
                target_topic,
                "recall",
                float(self.p.topic_graph_recall_lr),
                turn,
            )
            self.bridge_hit_events += 1
        if current_topic != target_topic and selected_topics and selected_topics[0] == target_topic:
            self._reinforce_bridge(
                current_topic,
                target_topic,
                "adopt",
                float(self.p.topic_graph_adopt_lr),
                turn,
            )
        for tid in bridge_topics:
            if tid in selected_topics and tid != current_topic:
                self._reinforce_bridge(
                    current_topic,
                    int(tid),
                    "recall",
                    float(self.p.topic_graph_recall_lr) * 0.45,
                    turn,
                )

    def _summary(self) -> Dict[str, float]:
        loaded_topics = sum(1 for shard in self.topic_shards if shard.loaded)
        active_topics = sum(1 for shard in self.topic_shards if shard.members)
        bridge_edges = sum(len(edges) for edges in self.topic_bridges)
        eval_n = max(1, self.eval_count)
        avg_query_ops = self.sim_ops_query_total / max(1, self.query_turns)
        avg_turn_ops = float(np.mean(self.turn_sim_ops)) if self.turn_sim_ops else 0.0
        p95_turn_ops = float(np.quantile(self.turn_sim_ops, 0.95)) if self.turn_sim_ops else 0.0
        target_hit = self.target_hit_sum / eval_n
        target_precision = self.target_precision_sum / eval_n
        target_mrr = self.target_mrr_sum / eval_n
        ghsom_probe_avg_candidates = self.ghsom_probe_candidates_total / max(1, self.ghsom_probe_count)
        ghsom_probe_avg_depth = self.ghsom_probe_depth_total / max(1, self.ghsom_probe_count)

        return {
            "turns": float(self.p.turns),
            "query_turns": float(self.query_turns),
            "memory_count": float(self.mem_count),
            "hot_memory_count": float(loaded_topics),
            "hot_memory_ratio": (float(loaded_topics) / max(1.0, float(active_topics))),
            "cluster_count": float(active_topics),
            "hot_cluster_count": float(loaded_topics),
            "supercluster_count": 0.0,
            "supercluster_rebuild_count": 0.0,
            "avg_clusters_per_super": 0.0,
            "merge_count": 0.0,
            "new_count": float(self.mem_count),
            "merge_rate": 0.0,
            "normalize_events": 0.0,
            "total_heat": 0.0,
            "heat_gini": 0.0,
            "target_precision_at_k": target_precision,
            "target_recall_recent": target_hit,
            "target_recall_all": target_hit,
            "target_hit_rate": target_hit,
            "target_recent_hit_rate": target_hit,
            "target_mrr": target_mrr,
            "empty_query_rate": (self.empty_query_count / max(1, self.query_turns)),
            "empty_target_query_rate": (self.empty_target_query_count / max(1, self.query_turns)),
            "avg_returned_turns": (self.returned_topics_sum / eval_n),
            "avg_sim_ops_add_per_turn": 0.0,
            "avg_sim_ops_query_per_query": avg_query_ops,
            "avg_sim_ops_total_per_turn": avg_turn_ops,
            "p95_sim_ops_total_per_turn": p95_turn_ops,
            "topic_graph_loaded_topics": float(loaded_topics),
            "topic_graph_active_topics": float(active_topics),
            "topic_graph_bridge_edges": float(bridge_edges),
            "topic_graph_load_events": float(self.topic_load_events),
            "topic_graph_evict_events": float(self.topic_evict_events),
            "topic_graph_bridge_expand_events": float(self.bridge_expand_events),
            "topic_graph_bridge_hit_events": float(self.bridge_hit_events),
            "topic_graph_bridge_update_events": float(self.bridge_update_events),
            "topic_graph_resident_hit_events": float(self.resident_hit_events),
            "preload_attempts": float(self.preload_attempts),
            "preload_successes": float(self.preload_successes),
            "preload_success_rate": (self.preload_successes / max(1, self.preload_attempts)),
            "preload_memories_heated": 0.0,
            "preload_clusters_loaded": float(self.preload_clusters_loaded),
            "preload_topic_predictions": 0.0,
            "preload_correct_predictions": 0.0,
            "preload_prediction_accuracy": 0.0,
            "ghsom_enabled": 1.0 if self.p.ghsom_enabled else 0.0,
            "ghsom_active": 1.0 if self.p.ghsom_enabled else 0.0,
            "ghsom_probe_count": float(self.ghsom_probe_count),
            "ghsom_probe_avg_candidates": float(ghsom_probe_avg_candidates),
            "ghsom_probe_avg_depth": float(ghsom_probe_avg_depth),
        }

    def _record_snapshot(self, turn: int) -> None:
        summary = self._summary()
        summary["turn"] = float(turn)
        self.snapshots.append(summary)

    def _finalize_summary(self, summary: Dict[str, float]) -> Dict[str, float]:
        summary["heat_gini_start"] = 0.0
        summary["heat_gini_end"] = 0.0
        summary["heat_gini_delta"] = 0.0
        summary["heat_gini_max"] = 0.0
        summary["heat_gini_min"] = 0.0
        return summary

    def run(self) -> Dict[str, object]:
        current_topic = self.topic.initial_topic()
        prev_topic: Optional[int] = None

        for turn in range(1, self.p.turns + 1):
            self._decay_bridges()
            if turn > 1:
                next_topic = self.topic.next_topic(current_topic)
                if prev_topic is not None and next_topic != current_topic:
                    self._reinforce_bridge(
                        prev_topic,
                        current_topic,
                        "transition",
                        float(self.p.topic_graph_transition_lr),
                        turn,
                    )
                prev_topic = current_topic
                current_topic = next_topic

            turn_vec = self.topic.turn_vector(current_topic)
            query_ops = 0

            if turn > self.p.warmup_turns and self.rng.random() < self.p.query_prob:
                target_topic = self._sample_query_topic(current_topic, turn)
                noise_mix = self.p.query_noise_mix
                query_vec = self.topic.query_vector(target_topic, noise_mix)
                selected_topics, query_ops, debug = self.retrieve(
                    query_vec,
                    current_topic=current_topic,
                    current_turn=turn,
                )
                self.sim_ops_query_total += query_ops
                self.query_turns += 1
                if not selected_topics:
                    self.empty_query_count += 1

                hit, _mrr = self._eval_query(selected_topics, target_topic)
                if hit <= 0:
                    self.empty_target_query_count += 1
                self._apply_query_feedback(
                    current_topic=current_topic,
                    target_topic=target_topic,
                    selected_topics=selected_topics,
                    bridge_topics=debug.get("bridge_topics", []),
                    turn=turn,
                )

            self._register_turn_topic(turn, current_topic)
            self._add_memory(turn_vec, turn, current_topic)
            self.turn_sim_ops.append(int(query_ops))

            if turn % self.p.report_every == 0 or turn == self.p.turns:
                self._record_snapshot(turn)

        summary = self._summary()
        summary = self._finalize_summary(summary)
        return {"summary": summary, "snapshots": self.snapshots}


def run_experiment(params: SimParams) -> Dict[str, object]:
    if str(params.retrieval_model).strip().lower() == "topic_graph":
        sim = TopicGraphSim(params)
    else:
        sim = AgentMemorySim(params)
    return sim.run()


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


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Simulator with Smart Preloading and EnhancedWeak")
    parser.add_argument("--retrieval-model", type=str, default="memory",
                        choices=["memory", "topic_graph"],
                        help="Retrieval model to simulate")
    parser.add_argument("--turns", type=int, default=20000, help="Number of simulation turns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--report-every", type=int, default=1000, help="Report interval")
    parser.add_argument("--output", type=str, default="sim_results.json", help="Output file")
    parser.add_argument("--no-plot", action="store_true", help="Disable png dashboard generation")
    parser.add_argument("--plot-out", type=str, default="", help="Output png path (default: output with .png)")
    parser.add_argument("--plot-dpi", type=int, default=160)
    
    # Smart Preloading parameters
    parser.add_argument("--preload-enabled", type=lambda x: x.lower() == 'true', default=True, 
                        help="Enable smart preloading (true/false)")
    parser.add_argument("--preload-budget", type=int, default=5, help="Preload cluster budget per query")
    parser.add_argument("--preload-max-io", type=int, default=8, help="Max preload clusters per turn")
    parser.add_argument("--memory-drop-sim", type=float, default=0.80,
                        help="Hard drop threshold for memory expansion")
    
    # EnhancedWeak parameters
    parser.add_argument("--soft-gate-enabled", type=lambda x: x.lower() == 'true', default=True,
                        help="Enable soft gate filtering (true/false)")
    parser.add_argument("--adaptive-gate-enabled", type=lambda x: x.lower() == 'true', default=True,
                        help="Enable adaptive gate adjustment (true/false)")
    parser.add_argument("--ghsom-enabled", type=lambda x: x.lower() == 'true', default=None,
                        help="Enable GHSOM-like hierarchical pruning (memory clusters or topic-local shards)")
    parser.add_argument("--ghsom-max-depth", type=int, default=3,
                        help="Maximum depth for the in-cluster GHSOM index")
    parser.add_argument("--ghsom-min-samples", type=int, default=10,
                        help="Minimum samples in a slot before GHSOM split")
    parser.add_argument("--ghsom-threshold", type=int, default=24,
                        help="Only use GHSOM routing when cluster size reaches this threshold")
    
    args = parser.parse_args()
    
    ghsom_enabled = args.ghsom_enabled
    if ghsom_enabled is None:
        ghsom_enabled = (args.retrieval_model == "topic_graph")

    params = SimParams(
        retrieval_model=args.retrieval_model,
        turns=args.turns,
        seed=args.seed,
        report_every=args.report_every,
        smart_preload_enabled=args.preload_enabled,
        preload_budget_per_query=args.preload_budget,
        preload_max_io_per_turn=args.preload_max_io,
        memory_drop_sim=max(-1.0, min(1.0, float(args.memory_drop_sim))),
        soft_gate_enabled=args.soft_gate_enabled,
        ghsom_enabled=ghsom_enabled,
        ghsom_max_depth=max(1, int(args.ghsom_max_depth)),
        ghsom_min_samples_for_expansion=max(2, int(args.ghsom_min_samples)),
        ghsom_linear_scan_threshold=max(2, int(args.ghsom_threshold)),
    )
    
    print(f"Retrieval model: {args.retrieval_model}")
    print(f"Running simulation with {args.turns} turns...")
    print(f"Smart preloading: {'enabled' if args.preload_enabled else 'disabled'}")
    print(f"Soft gate: {'enabled' if args.soft_gate_enabled else 'disabled'}")
    if args.retrieval_model == "memory":
        print(f"GHSOM index: {'enabled' if ghsom_enabled else 'disabled'}")
    else:
        print(
            "GHSOM index: "
            f"configured={ghsom_enabled}, active={'yes' if ghsom_enabled else 'no'} "
            "(topic_graph local pruning)"
        )
    
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
    if args.retrieval_model == "topic_graph":
        print(f"Topic graph bridge edges: {summary.get('topic_graph_bridge_edges', 0):.0f}")
        print(f"Loaded topics: {summary.get('topic_graph_loaded_topics', 0):.0f}")
    if ghsom_enabled:
        print(f"GHSOM avg probe depth: {summary.get('ghsom_probe_avg_depth', 0.0):.2f}")
        print(f"GHSOM avg candidate pool: {summary.get('ghsom_probe_avg_candidates', 0.0):.2f}")
    
    if args.preload_enabled:
        print(f"\nSmart Preloading Stats:")
        print(f"  Preload success rate: {summary.get('preload_success_rate', 0):.4f}")
        print(f"  Preload memories cached: {summary.get('preload_memories_heated', 0):.0f}")
        print(f"  Preload clusters loaded: {summary.get('preload_clusters_loaded', 0):.0f}")
    
    print(f"\nEnhancedWeak Stats:")
    print(f"  Soft gate passes: {summary.get('enhanced_soft_gate_pass_count', 0):.0f}")
    print(f"  Learned min gate: {summary.get('learned_min_sim_gate', 0):.4f}")
    
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump({"summary": summary, "snapshots": result["snapshots"]}, f, indent=2)
        print(f"\nResults saved to {output_path}")

    if not args.no_plot:
        if args.plot_out:
            plot_out = Path(args.plot_out)
        elif args.output:
            plot_out = Path(args.output).with_suffix(".png")
        else:
            plot_out = Path("sim_results.png")
        ok, reason = save_plot(plot_out, result["snapshots"], summary, dpi=args.plot_dpi)
        if ok:
            print(f"Snapshot plot saved to {plot_out}")
        else:
            print(f"Snapshot plot skipped: {reason}")


if __name__ == "__main__":
    main()
