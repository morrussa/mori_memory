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

try:
    from mori_hnsw import HNSWIndex
except Exception:
    HNSWIndex = None


TOPIC_FAMILY_DYNAMIC_REBUILD_INTERVAL = 128


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
    topic_graph_anchor_graph_enabled: bool = False
    topic_graph_anchor_seed_k: int = 2
    topic_graph_anchor_neighbor_topk: int = 2
    topic_graph_anchor_max_hops: int = 1
    topic_graph_anchor_edge_decay: float = 0.65
    topic_graph_momentum_probe_enabled: bool = False
    topic_graph_momentum_probe_cap: int = 1
    topic_graph_momentum_probe_min_sim: float = 0.58
    topic_graph_momentum_probe_topic_margin: float = 0.03
    topic_graph_family_topk: int = 3
    topic_graph_family_similarity: float = 0.82
    topic_graph_family_member_limit: int = 2
    topic_graph_self_excite_penalty: float = 0.18
    topic_graph_family_revisit_penalty: float = 0.10
    topic_graph_family_escape_bonus: float = 0.06
    topic_graph_local_index: str = "topoart"
    topic_graph_topoart_min_members: int = 8
    topic_graph_topoart_vigilance: float = 0.88
    topic_graph_topoart_secondary_vigilance: float = 0.80
    topic_graph_topoart_beta: float = 0.28
    topic_graph_topoart_beta_secondary: float = 0.10
    topic_graph_topoart_query_categories: int = 4
    topic_graph_topoart_neighbor_topk: int = 3
    topic_graph_topoart_exemplars: int = 12
    topic_graph_topoart_prune_interval: int = 256
    topic_graph_topoart_prune_min_support: int = 2
    topic_graph_topoart_match_slack: float = 0.050
    topic_graph_topoart_category_capacity: int = 6
    topic_graph_topoart_capacity_boost: float = 0.030
    topic_graph_topoart_link_margin: float = 0.045
    topic_graph_topoart_query_margin: float = 0.080
    topic_graph_topoart_temporal_link_window: int = 6
    topic_graph_deep_artmap_min_members: int = 4
    topic_graph_deep_artmap_bundle_vigilance: float = 0.78
    topic_graph_deep_artmap_bundle_beta: float = 0.18
    topic_graph_deep_artmap_query_bundles: int = 3
    topic_graph_deep_artmap_query_margin: float = 0.10
    topic_graph_deep_artmap_neighbor_topk: int = 2
    topic_graph_deep_artmap_temporal_link_window: int = 8
    topic_graph_deep_artmap_bundle_prior_weight: float = 0.18
    topic_graph_deep_artmap_bundle_recall_lr: float = 0.10
    topic_graph_deep_artmap_bundle_adopt_lr: float = 0.18
    topic_graph_deep_artmap_bundle_edge_feedback: float = 0.14
    topic_graph_deep_artmap_bundle_decay: float = 0.996
    topic_graph_topic_hnsw_enabled: bool = False
    topic_graph_topic_hnsw_k: int = 48
    topic_graph_topic_hnsw_m: int = 16
    topic_graph_topic_hnsw_ef_construction: int = 80
    topic_graph_topic_hnsw_ef_search: int = 32
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
    topic_facets_per_topic: int = 6
    topic_bundles_per_topic: int = 3
    topic_bundle_facet_min: int = 2
    topic_bundle_facet_max: int = 3
    topic_bundle_variant_mix: float = 0.16
    topic_temporal_state_enabled: bool = False
    topic_temporal_velocity_mix: float = 0.32
    topic_temporal_state_momentum: float = 0.78
    topic_temporal_state_reversion: float = 0.06
    topic_temporal_state_noise: float = 0.07
    topic_temporal_focus_stickiness: float = 0.72
    topic_temporal_turn_state_mix: float = 0.78
    topic_temporal_query_state_mix: float = 0.68
    turn_facet_min: int = 1
    turn_facet_max: int = 2
    turn_bundle_min: int = 1
    turn_bundle_max: int = 1
    turn_bundle_mix: float = 0.30
    topic_atomic_memories_min: int = 1
    topic_atomic_memories_max: int = 3
    topic_atomic_facet_min: int = 1
    topic_atomic_facet_max: int = 2
    topic_atomic_secondary_mix: float = 0.16
    topic_atomic_max_self_similarity: float = 0.93
    topic_atomic_anchor_deflate: float = 0.08
    query_facet_min: int = 2
    query_facet_max: int = 3
    query_bundle_min: int = 1
    query_bundle_max: int = 2
    query_bundle_mix: float = 0.38
    topic_facet_variant_mix: float = 0.32
    turn_facet_mix: float = 0.28
    query_facet_mix: float = 0.42
    context_redundancy_weight: float = 0.35
    context_irrelevance_weight: float = 0.20
    context_association_weight: float = 0.16
    context_saturation_weight: float = 0.22
    context_similarity_saturation_threshold: float = 0.95
    context_min_marginal_gain: float = 0.02


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
    topoart: Optional["TopoARTState"] = None
    deep_artmap: Optional["DeepARTMAPState"] = None


@dataclass
class TopoARTCategoryState:
    prototype: np.ndarray
    vector_sum: np.ndarray  # unnormalized sum of member vectors
    support: int = 0
    member_count: int = 0
    exemplars: List[int] = field(default_factory=list)
    exemplar_scores: List[float] = field(default_factory=list)
    neighbors: Dict[int, float] = field(default_factory=dict)
    last_update_turn: int = 0
    match_ema: float = 1.0
    match_min: float = 1.0
    active: bool = True


@dataclass
class TopoARTState:
    categories: List[TopoARTCategoryState] = field(default_factory=list)
    update_count: int = 0
    last_winner_id: int = -1
    last_winner_turn: int = 0


@dataclass
class TopoARTInsertResult:
    ops: int = 0
    winner_id: int = -1
    second_id: int = -1
    created: bool = False


@dataclass
class DeepARTMAPBundleState:
    prototype: np.ndarray
    vector_sum: np.ndarray  # unnormalized sum of category prototypes (or member vectors)
    category_ids: List[int] = field(default_factory=list)
    support: int = 0
    last_update_turn: int = 0
    neighbors: Dict[int, float] = field(default_factory=dict)
    recall_score: float = 0.0
    adopt_score: float = 0.0
    active: bool = True


@dataclass
class DeepARTMAPState:
    bundles: List[DeepARTMAPBundleState] = field(default_factory=list)
    category_to_bundle: Dict[int, int] = field(default_factory=dict)
    last_bundle_id: int = -1
    last_bundle_turn: int = 0


@dataclass
class TopicLocalQueryResult:
    indices: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=np.int32))
    score: float = 0.0
    ops: int = 0
    bundle_ids: List[int] = field(default_factory=list)
    category_ids: List[int] = field(default_factory=list)
    memory_scores: List[float] = field(default_factory=list)
    memory_bundle_ids: List[int] = field(default_factory=list)


class TopicLocalIndex:
    name = "exact"

    def add_memory(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        vec: np.ndarray,
        mem_idx: int,
        turn: int,
    ) -> int:
        return 0

    def collect_candidates(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        query_vec: np.ndarray,
        max_results: int,
        query_facet_ids: Optional[np.ndarray] = None,
        query_facet_weights: Optional[np.ndarray] = None,
    ) -> TopicLocalQueryResult:
        return TopicLocalQueryResult()

    def summary(self, sim: "TopicGraphSim", active_topics: int) -> Dict[str, float]:
        return {
            "topic_graph_local_index_is_topoart": 0.0,
            "topic_graph_local_index_is_deep_artmap": 0.0,
            "topic_graph_local_index_is_ghsom": 0.0,
            "deep_artmap_scaffold": 0.0,
            "deep_artmap_layer_count": 0.0,
            "deep_artmap_bundle_count": 0.0,
            "deep_artmap_avg_categories_per_bundle": 0.0,
            "deep_artmap_probe_count": 0.0,
            "deep_artmap_probe_avg_bundles": 0.0,
            "deep_artmap_probe_avg_categories": 0.0,
            "deep_artmap_bundle_avg_recall_prior": 0.0,
            "deep_artmap_bundle_avg_adopt_prior": 0.0,
            "topoart_category_count": 0.0,
            "topoart_edge_count": 0.0,
            "topoart_probe_count": 0.0,
            "topoart_probe_avg_candidates": 0.0,
            "topoart_probe_avg_categories": 0.0,
            "topoart_avg_categories_per_active_topic": 0.0,
            "ghsom_probe_count": 0.0,
            "ghsom_probe_avg_candidates": 0.0,
            "ghsom_probe_avg_depth": 0.0,
        }

    def ghsom_active(self, sim: "TopicGraphSim") -> bool:
        return False


class ExactTopicLocalIndex(TopicLocalIndex):
    name = "exact"


class GHSOMTopicLocalIndex(TopicLocalIndex):
    name = "ghsom"

    def add_memory(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        vec: np.ndarray,
        mem_idx: int,
        turn: int,
    ) -> int:
        if not bool(sim.p.ghsom_enabled):
            return 0
        root, build_ops, fresh = sim._ensure_topic_ghsom_root(topic_id, vec)
        if root is None:
            return build_ops
        if fresh:
            return build_ops
        return build_ops + sim._ghsom_insert(root, vec, mem_idx)

    def collect_candidates(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        query_vec: np.ndarray,
        max_results: int,
        query_facet_ids: Optional[np.ndarray] = None,
        query_facet_weights: Optional[np.ndarray] = None,
    ) -> TopicLocalQueryResult:
        if not bool(sim.p.ghsom_enabled):
            return TopicLocalQueryResult()
        shard = sim.topic_shards[topic_id]
        if shard.ghsom_root is None:
            return TopicLocalQueryResult()
        idx, ops = sim._ghsom_collect_candidates(
            shard.ghsom_root,
            query_vec,
            max_results=max_results,
        )
        return TopicLocalQueryResult(indices=idx, score=0.0, ops=ops)

    def summary(self, sim: "TopicGraphSim", active_topics: int) -> Dict[str, float]:
        data = super().summary(sim, active_topics)
        ghsom_probe_avg_candidates = sim.ghsom_probe_candidates_total / max(1, sim.ghsom_probe_count)
        ghsom_probe_avg_depth = sim.ghsom_probe_depth_total / max(1, sim.ghsom_probe_count)
        data.update(
            {
                "topic_graph_local_index_is_ghsom": 1.0,
                "ghsom_probe_count": float(sim.ghsom_probe_count),
                "ghsom_probe_avg_candidates": float(ghsom_probe_avg_candidates),
                "ghsom_probe_avg_depth": float(ghsom_probe_avg_depth),
            }
        )
        return data

    def ghsom_active(self, sim: "TopicGraphSim") -> bool:
        return bool(sim.p.ghsom_enabled)


class TopoARTTopicLocalIndex(TopicLocalIndex):
    name = "topoart"

    def add_memory(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        vec: np.ndarray,
        mem_idx: int,
        turn: int,
    ) -> int:
        return sim._topoart_insert(topic_id, vec, mem_idx, turn)

    def collect_candidates(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        query_vec: np.ndarray,
        max_results: int,
        query_facet_ids: Optional[np.ndarray] = None,
        query_facet_weights: Optional[np.ndarray] = None,
    ) -> TopicLocalQueryResult:
        shard = sim.topic_shards[topic_id]
        if len(shard.members) < max(2, int(sim.p.topic_graph_topoart_min_members)):
            return TopicLocalQueryResult()
        idx, score, ops = sim._topoart_collect_candidates(
            topic_id,
            query_vec,
            max_results=max_results,
        )
        return TopicLocalQueryResult(indices=idx, score=score, ops=ops)

    def summary(self, sim: "TopicGraphSim", active_topics: int) -> Dict[str, float]:
        data = super().summary(sim, active_topics)
        topoart_probe_avg_candidates = sim.topoart_probe_candidates_total / max(1, sim.topoart_probe_count)
        topoart_probe_avg_categories = sim.topoart_probe_categories_total / max(1, sim.topoart_probe_count)
        topoart_active_categories = 0
        topoart_active_edges = 0
        for shard in sim.topic_shards:
            state = shard.topoart
            if state is None:
                continue
            for idx, cat in enumerate(state.categories):
                if not cat.active:
                    continue
                topoart_active_categories += 1
                topoart_active_edges += sum(
                    1
                    for nid in cat.neighbors
                    if nid > idx and state.categories[nid].active
                )
        data.update(
            {
                "topic_graph_local_index_is_topoart": 1.0,
                "topoart_category_count": float(topoart_active_categories),
                "topoart_edge_count": float(topoart_active_edges),
                "topoart_probe_count": float(sim.topoart_probe_count),
                "topoart_probe_avg_candidates": float(topoart_probe_avg_candidates),
                "topoart_probe_avg_categories": float(topoart_probe_avg_categories),
                "topoart_avg_categories_per_active_topic": float(
                    topoart_active_categories / max(1, active_topics)
                ),
            }
        )
        return data


class DeepARTMAPTopicLocalIndex(TopoARTTopicLocalIndex):
    name = "deep_artmap"

    def add_memory(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        vec: np.ndarray,
        mem_idx: int,
        turn: int,
    ) -> int:
        return sim._deep_artmap_insert(topic_id, vec, mem_idx, turn)

    def collect_candidates(
        self,
        sim: "TopicGraphSim",
        topic_id: int,
        query_vec: np.ndarray,
        max_results: int,
        query_facet_ids: Optional[np.ndarray] = None,
        query_facet_weights: Optional[np.ndarray] = None,
    ) -> TopicLocalQueryResult:
        return sim._deep_artmap_collect_candidates(
            topic_id,
            query_vec,
            max_results,
            query_facet_ids=query_facet_ids,
            query_facet_weights=query_facet_weights,
        )

    def summary(self, sim: "TopicGraphSim", active_topics: int) -> Dict[str, float]:
        data = super().summary(sim, active_topics)
        bundle_count = 0
        bundle_category_total = 0
        bundle_recall_total = 0.0
        bundle_adopt_total = 0.0
        # Use global deep artmap state
        state = sim.global_deep_artmap
        for bundle in state.bundles:
            if not bundle.active:
                continue
            bundle_count += 1
            bundle_category_total += len(bundle.category_ids)
        # Sum per-topic priors across all topics
        for tid in range(sim.topic_count):
            for prior in sim.bundle_recall_prior[tid].values():
                bundle_recall_total += prior
            for prior in sim.bundle_adopt_prior[tid].values():
                bundle_adopt_total += prior
        data.update(
            {
                "topic_graph_local_index_is_topoart": 0.0,
                "topic_graph_local_index_is_deep_artmap": 1.0,
                "deep_artmap_scaffold": 0.0,
                "deep_artmap_layer_count": 2.0,
                "deep_artmap_bundle_count": float(bundle_count),
                "deep_artmap_avg_categories_per_bundle": float(
                    bundle_category_total / max(1, bundle_count)
                ),
                "deep_artmap_probe_count": float(sim.deep_artmap_probe_count),
                "deep_artmap_probe_avg_bundles": float(
                    sim.deep_artmap_probe_bundle_total / max(1, sim.deep_artmap_probe_count)
                ),
                "deep_artmap_probe_avg_categories": float(
                    sim.deep_artmap_probe_category_total / max(1, sim.deep_artmap_probe_count)
                ),
                "deep_artmap_bundle_avg_recall_prior": float(
                    bundle_recall_total / max(1, bundle_count)
                ),
                "deep_artmap_bundle_avg_adopt_prior": float(
                    bundle_adopt_total / max(1, bundle_count)
                ),
            }
        )
        return data


class TopicStream:
    def __init__(self, p: SimParams, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.topic_vectors, self.topic_groups = self._build_topics()
        self.num_topics = self.topic_vectors.shape[0]
        self.topic_facet_vectors = self._build_topic_facets()
        self.topic_bundle_facet_weights, self.topic_bundle_vectors = self._build_topic_bundles()
        self.group_to_topics: List[List[int]] = [[] for _ in range(p.topic_groups)]
        for tid, gid in enumerate(self.topic_groups):
            self.group_to_topics[gid].append(tid)
        self.topic_sim = self.topic_vectors @ self.topic_vectors.T
        self._active_topic: Optional[int] = None
        self._flow_anchor: Optional[np.ndarray] = None
        self._init_temporal_topic_state()

    def _normalize_simplex(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1).astype(np.float32)
        arr = np.maximum(arr, 1e-6).astype(np.float32)
        total = float(np.sum(arr))
        if total <= 0.0:
            return np.full(arr.shape[0], 1.0 / max(1, arr.shape[0]), dtype=np.float32)
        return (arr / total).astype(np.float32)

    def _init_temporal_topic_state(self) -> None:
        bundle_count = max(1, int(self.p.topic_bundles_per_topic))
        salience = np.sum(np.maximum(self.topic_bundle_facet_weights, 0.0), axis=2).astype(np.float32)
        salience += 0.05 * self.rng.random(salience.shape).astype(np.float32)
        self.topic_bundle_base = np.zeros((self.num_topics, bundle_count), dtype=np.float32)
        self.topic_bundle_state = np.zeros((self.num_topics, bundle_count), dtype=np.float32)
        self.topic_bundle_velocity = np.zeros((self.num_topics, bundle_count), dtype=np.float32)
        self.topic_focus_bundle = np.full(self.num_topics, -1, dtype=np.int32)
        for tid in range(self.num_topics):
            base = self._normalize_simplex(salience[tid])
            self.topic_bundle_base[tid] = base
            self.topic_bundle_state[tid] = base

    def _advance_temporal_topic_state(self, topic_id: int) -> None:
        if not bool(self.p.topic_temporal_state_enabled):
            return
        tid = int(topic_id)
        if tid < 0 or tid >= self.num_topics:
            return

        base = self.topic_bundle_base[tid]
        state = self.topic_bundle_state[tid]
        velocity = self.topic_bundle_velocity[tid]
        bundle_count = max(1, int(base.shape[0]))
        focus_stickiness = min(0.995, max(0.0, float(self.p.topic_temporal_focus_stickiness)))
        prev_focus = int(self.topic_focus_bundle[tid])
        if 0 <= prev_focus < bundle_count and self.rng.random() < focus_stickiness:
            focus = prev_focus
        else:
            focus_prior = self._normalize_simplex(0.70 * state + 0.30 * base)
            focus = int(self.rng.choice(bundle_count, p=focus_prior))

        drive = np.array(base, dtype=np.float32, copy=True)
        drive[focus] += 1.0
        drive = self._normalize_simplex(drive)
        momentum = min(0.999, max(0.0, float(self.p.topic_temporal_state_momentum)))
        velocity_mix = max(0.0, float(self.p.topic_temporal_velocity_mix))
        reversion = max(0.0, float(self.p.topic_temporal_state_reversion))
        state_noise = max(0.0, float(self.p.topic_temporal_state_noise))
        noise = self.rng.normal(size=bundle_count).astype(np.float32)
        noise = noise / max(1.0, float(np.linalg.norm(noise)))
        velocity = (
            momentum * velocity
            + velocity_mix * (drive - state)
            + state_noise * noise
        ).astype(np.float32)
        state = self._normalize_simplex(state + velocity + reversion * (base - state))
        self.topic_bundle_velocity[tid] = velocity
        self.topic_bundle_state[tid] = state
        self.topic_focus_bundle[tid] = int(focus)

    def _bundle_prior(self, topic_id: int, for_query: bool) -> Optional[np.ndarray]:
        if not bool(self.p.topic_temporal_state_enabled):
            return None
        tid = int(topic_id)
        if tid < 0 or tid >= self.num_topics:
            return None
        base = self.topic_bundle_base[tid]
        state = self.topic_bundle_state[tid]
        velocity = self.topic_bundle_velocity[tid]
        mix = (
            float(self.p.topic_temporal_query_state_mix)
            if for_query
            else float(self.p.topic_temporal_turn_state_mix)
        )
        mix = min(1.0, max(0.0, mix))
        velocity_mix = max(0.0, float(self.p.topic_temporal_velocity_mix))
        predicted = self._normalize_simplex(np.maximum(1e-6, state + 0.50 * velocity_mix * velocity))
        return self._normalize_simplex((1.0 - mix) * base + mix * predicted)

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

    def _build_topic_facets(self) -> np.ndarray:
        mix = min(0.9, max(0.05, float(self.p.topic_facet_variant_mix)))
        facets = np.zeros((self.num_topics, max(1, int(self.p.topic_facets_per_topic)), self.p.dim), dtype=np.float32)
        for tid in range(self.num_topics):
            base = self.topic_vectors[tid]
            for fid in range(facets.shape[1]):
                noise = self._rand_unit(self.p.dim)
                facets[tid, fid] = unit((1.0 - mix) * base + mix * noise).astype(np.float32)
        return facets

    def _build_topic_bundles(self) -> Tuple[np.ndarray, np.ndarray]:
        bundle_count = max(1, int(self.p.topic_bundles_per_topic))
        facet_count = max(1, int(self.p.topic_facets_per_topic))
        lo = max(1, min(int(self.p.topic_bundle_facet_min), facet_count))
        hi = max(lo, min(int(self.p.topic_bundle_facet_max), facet_count))
        variant_mix = min(0.6, max(0.0, float(self.p.topic_bundle_variant_mix)))
        bundle_weights = np.zeros((self.num_topics, bundle_count, facet_count), dtype=np.float32)
        bundle_vectors = np.zeros((self.num_topics, bundle_count, self.p.dim), dtype=np.float32)
        for tid in range(self.num_topics):
            for bid in range(bundle_count):
                take = int(self.rng.integers(lo, hi + 1))
                local_ids = self.rng.choice(facet_count, size=take, replace=False).astype(np.int32)
                raw = self.rng.random(take).astype(np.float32)
                raw += np.linspace(0.15, 0.0, num=take, dtype=np.float32)
                weights = raw / max(1e-6, float(np.sum(raw)))
                for pos, local_id in enumerate(local_ids.tolist()):
                    bundle_weights[tid, bid, int(local_id)] = float(weights[pos])
                bundle_vec = np.zeros(self.p.dim, dtype=np.float32)
                for pos, local_id in enumerate(local_ids.tolist()):
                    bundle_vec += float(weights[pos]) * self.topic_facet_vectors[tid, int(local_id)]
                noise = self._rand_unit(self.p.dim)
                bundle_vectors[tid, bid] = unit((1.0 - variant_mix) * bundle_vec + variant_mix * noise).astype(np.float32)
        return bundle_weights, bundle_vectors

    def _sample_bundle_mix(
        self,
        topic_id: int,
        min_count: int,
        max_count: int,
        prior: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        bundle_count = max(1, int(self.p.topic_bundles_per_topic))
        lo = max(1, min(int(min_count), bundle_count))
        hi = max(lo, min(int(max_count), bundle_count))
        take = int(self.rng.integers(lo, hi + 1))
        if prior is not None:
            probs = self._normalize_simplex(np.asarray(prior, dtype=np.float32))
            if take >= bundle_count:
                bundle_ids = np.arange(bundle_count, dtype=np.int32)
            else:
                bundle_ids = self.rng.choice(bundle_count, size=take, replace=False, p=probs).astype(np.int32)
            raw = probs[bundle_ids].astype(np.float32) + 0.05 * self.rng.random(len(bundle_ids)).astype(np.float32)
            raw += np.linspace(0.12, 0.0, num=len(bundle_ids), dtype=np.float32)
            weights = raw / max(1e-6, float(np.sum(raw)))
        else:
            bundle_ids = self.rng.choice(bundle_count, size=take, replace=False).astype(np.int32)
            raw = self.rng.random(take).astype(np.float32)
            raw += np.linspace(0.20, 0.0, num=take, dtype=np.float32)
            weights = raw / max(1e-6, float(np.sum(raw)))
        return (int(topic_id) * bundle_count + bundle_ids).astype(np.int32), weights

    def _sample_facets_from_bundles(
        self,
        topic_id: int,
        bundle_ids: np.ndarray,
        bundle_weights: np.ndarray,
        min_count: int,
        max_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        facet_count = max(1, int(self.p.topic_facets_per_topic))
        lo = max(1, min(int(min_count), facet_count))
        hi = max(lo, min(int(max_count), facet_count))
        take = int(self.rng.integers(lo, hi + 1))
        probs = np.full(facet_count, 1e-4, dtype=np.float32)
        local_bundle_ids = (np.asarray(bundle_ids, dtype=np.int32) - int(topic_id) * max(1, int(self.p.topic_bundles_per_topic))).astype(np.int32)
        for pos, local_bid in enumerate(local_bundle_ids.tolist()):
            if local_bid < 0 or local_bid >= self.topic_bundle_facet_weights.shape[1]:
                continue
            probs += float(bundle_weights[pos]) * self.topic_bundle_facet_weights[int(topic_id), int(local_bid)]
        probs = probs / max(1e-6, float(np.sum(probs)))
        local_ids = self.rng.choice(facet_count, size=take, replace=False, p=probs).astype(np.int32)
        local_weights = probs[local_ids].astype(np.float32)
        local_weights /= max(1e-6, float(np.sum(local_weights)))
        global_ids = (int(topic_id) * facet_count + local_ids).astype(np.int32)
        return global_ids, local_weights

    def _sample_facets(
        self,
        topic_id: int,
        min_count: int,
        max_count: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        facet_count = max(1, int(self.p.topic_facets_per_topic))
        lo = max(1, min(int(min_count), facet_count))
        hi = max(lo, min(int(max_count), facet_count))
        take = int(self.rng.integers(lo, hi + 1))
        local_ids = self.rng.choice(facet_count, size=take, replace=False).astype(np.int32)
        raw = self.rng.random(take).astype(np.float32)
        weights = raw / max(1e-6, float(np.sum(raw)))
        global_ids = (int(topic_id) * facet_count + local_ids).astype(np.int32)
        return global_ids, weights

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

    def turn_sample(self, topic_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        base = self.topic_vectors[topic_id]
        drift = min(0.8, max(0.0, float(self.p.topic_flow_drift)))
        anchor_mix = min(0.8, max(0.0, float(self.p.topic_flow_anchor_mix)))
        switch_jolt = min(0.9, max(0.0, float(self.p.topic_flow_switch_jolt)))
        noise_mix = min(0.95, max(0.01, float(self.p.turn_noise_mix)))
        bundle_mix = min(0.7, max(0.0, float(self.p.turn_bundle_mix)))
        facet_mix = min(0.75, max(0.0, float(self.p.turn_facet_mix)))

        if self._active_topic != topic_id or self._flow_anchor is None:
            self._active_topic = topic_id
            j = self._rand_unit(self.p.dim)
            self._flow_anchor = unit((1.0 - switch_jolt) * base + switch_jolt * j)
        else:
            d = self._rand_unit(self.p.dim)
            self._flow_anchor = unit((1.0 - drift) * self._flow_anchor + drift * d)

        self._advance_temporal_topic_state(topic_id)
        noise = self._rand_unit(self.p.dim)
        bundle_prior = self._bundle_prior(topic_id, for_query=False)
        bundle_ids, bundle_weights = self._sample_bundle_mix(
            topic_id,
            int(self.p.turn_bundle_min),
            int(self.p.turn_bundle_max),
            prior=bundle_prior,
        )
        local_bundle_ids = (bundle_ids - int(topic_id) * max(1, int(self.p.topic_bundles_per_topic))).astype(np.int32)
        bundle_vec = np.zeros(self.p.dim, dtype=np.float32)
        for pos, local_bid in enumerate(local_bundle_ids.tolist()):
            bundle_vec += float(bundle_weights[pos]) * self.topic_bundle_vectors[int(topic_id), int(local_bid)]
        facet_ids, facet_weights = self._sample_facets_from_bundles(
            topic_id,
            bundle_ids,
            bundle_weights,
            int(self.p.turn_facet_min),
            int(self.p.turn_facet_max),
        )
        local_ids = (facet_ids - int(topic_id) * max(1, int(self.p.topic_facets_per_topic))).astype(np.int32)
        facet_vec = np.zeros(self.p.dim, dtype=np.float32)
        for pos, local_id in enumerate(local_ids.tolist()):
            facet_vec += float(facet_weights[pos]) * self.topic_facet_vectors[int(topic_id), int(local_id)]
        base_w = max(0.01, 1.0 - noise_mix - anchor_mix - bundle_mix - facet_mix)
        vec = unit(
            base_w * base
            + anchor_mix * self._flow_anchor
            + bundle_mix * bundle_vec
            + facet_mix * facet_vec
            + noise_mix * noise
        )
        return vec.astype(np.float32), facet_ids.astype(np.int32), facet_weights.astype(np.float32)

    def turn_vector(self, topic_id: int) -> np.ndarray:
        vec, _facet_ids, _facet_weights = self.turn_sample(topic_id)
        return vec

    def query_sample(self, topic_id: int, noise_mix: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        noise = self._rand_unit(self.p.dim)
        nm = min(0.95, max(0.01, float(noise_mix)))
        bundle_mix = min(0.8, max(0.0, float(self.p.query_bundle_mix)))
        facet_mix = min(0.8, max(0.0, float(self.p.query_facet_mix)))
        bundle_prior = self._bundle_prior(topic_id, for_query=True)
        bundle_ids, bundle_weights = self._sample_bundle_mix(
            topic_id,
            int(self.p.query_bundle_min),
            int(self.p.query_bundle_max),
            prior=bundle_prior,
        )
        local_bundle_ids = (bundle_ids - int(topic_id) * max(1, int(self.p.topic_bundles_per_topic))).astype(np.int32)
        bundle_vec = np.zeros(self.p.dim, dtype=np.float32)
        for pos, local_bid in enumerate(local_bundle_ids.tolist()):
            bundle_vec += float(bundle_weights[pos]) * self.topic_bundle_vectors[int(topic_id), int(local_bid)]
        facet_ids, facet_weights = self._sample_facets_from_bundles(
            topic_id,
            bundle_ids,
            bundle_weights,
            int(self.p.query_facet_min),
            int(self.p.query_facet_max),
        )
        local_ids = (facet_ids - int(topic_id) * max(1, int(self.p.topic_facets_per_topic))).astype(np.int32)
        facet_vec = np.zeros(self.p.dim, dtype=np.float32)
        for pos, local_id in enumerate(local_ids.tolist()):
            facet_vec += float(facet_weights[pos]) * self.topic_facet_vectors[int(topic_id), int(local_id)]
        base_w = max(0.01, 1.0 - nm - bundle_mix - facet_mix)
        vec = unit(
            base_w * self.topic_vectors[topic_id]
            + bundle_mix * bundle_vec
            + facet_mix * facet_vec
            + nm * noise
        )
        return vec.astype(np.float32), facet_ids.astype(np.int32), facet_weights.astype(np.float32)

    def query_vector(self, topic_id: int, noise_mix: float) -> np.ndarray:
        vec, _facet_ids, _facet_weights = self.query_sample(topic_id, noise_mix)
        return vec

    def turn_sample_with_atomics(
        self,
        topic_id: int,
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        tid = int(topic_id)
        min_atomic = max(1, int(self.p.topic_atomic_memories_min))
        max_atomic = max(min_atomic, int(self.p.topic_atomic_memories_max))
        atomic_count = int(self.rng.integers(min_atomic, max_atomic + 1))
        bundle_count = max(1, int(self.p.topic_bundles_per_topic))
        facet_count = max(1, int(self.p.topic_facets_per_topic))

        base = self.topic_vectors[tid]
        drift = min(0.8, max(0.0, float(self.p.topic_flow_drift)))
        anchor_mix = min(0.8, max(0.0, float(self.p.topic_flow_anchor_mix)))
        switch_jolt = min(0.9, max(0.0, float(self.p.topic_flow_switch_jolt)))
        noise_mix = min(0.95, max(0.01, float(self.p.turn_noise_mix)))
        bundle_mix = min(0.7, max(0.0, float(self.p.turn_bundle_mix)))
        facet_mix = min(0.75, max(0.0, float(self.p.turn_facet_mix)))

        if self._active_topic != tid or self._flow_anchor is None:
            self._active_topic = tid
            j = self._rand_unit(self.p.dim)
            self._flow_anchor = unit((1.0 - switch_jolt) * base + switch_jolt * j)
        else:
            d = self._rand_unit(self.p.dim)
            self._flow_anchor = unit((1.0 - drift) * self._flow_anchor + drift * d)

        self._advance_temporal_topic_state(tid)
        bundle_prior = self._bundle_prior(tid, for_query=False)
        anchor_bundle_cap = min(bundle_count, max(int(self.p.turn_bundle_max), atomic_count))
        anchor_bundle_min = min(anchor_bundle_cap, max(1, int(self.p.turn_bundle_min)))
        anchor_bundle_ids, anchor_bundle_weights = self._sample_bundle_mix(
            tid,
            anchor_bundle_min,
            anchor_bundle_cap,
            prior=bundle_prior,
        )
        local_bundle_ids = (
            anchor_bundle_ids - tid * bundle_count
        ).astype(np.int32)
        bundle_vec = np.zeros(self.p.dim, dtype=np.float32)
        for pos, local_bid in enumerate(local_bundle_ids.tolist()):
            bundle_vec += float(anchor_bundle_weights[pos]) * self.topic_bundle_vectors[tid, int(local_bid)]

        facet_probs = np.full(facet_count, 1e-4, dtype=np.float32)
        for pos, local_bid in enumerate(local_bundle_ids.tolist()):
            if local_bid < 0 or local_bid >= self.topic_bundle_facet_weights.shape[1]:
                continue
            facet_probs += float(anchor_bundle_weights[pos]) * self.topic_bundle_facet_weights[tid, int(local_bid)]
        facet_probs = self._normalize_simplex(facet_probs)
        source_take = min(
            facet_count,
            max(
                atomic_count,
                int(self.p.topic_atomic_facet_max) + 1,
                int(self.p.turn_facet_max),
            ),
        )
        source_local_ids = self.rng.choice(
            facet_count,
            size=source_take,
            replace=False,
            p=facet_probs,
        ).astype(np.int32)
        source_local_weights = facet_probs[source_local_ids].astype(np.float32)
        source_local_weights = self._normalize_simplex(source_local_weights)
        source_global_ids = (tid * facet_count + source_local_ids).astype(np.int32)

        facet_pool_vec = np.zeros(self.p.dim, dtype=np.float32)
        for pos, local_id in enumerate(source_local_ids.tolist()):
            facet_pool_vec += float(source_local_weights[pos]) * self.topic_facet_vectors[tid, int(local_id)]
        noise = self._rand_unit(self.p.dim)
        base_w = max(0.01, 1.0 - noise_mix - anchor_mix - bundle_mix - facet_mix)
        turn_vec = unit(
            base_w * base
            + anchor_mix * self._flow_anchor
            + bundle_mix * bundle_vec
            + facet_mix * facet_pool_vec
            + noise_mix * noise
        ).astype(np.float32)

        sorted_pos = np.argsort(source_local_weights)[::-1].tolist()
        max_self_sim = min(0.999, max(-1.0, float(self.p.topic_atomic_max_self_similarity)))
        secondary_mix = max(0.0, min(0.45, float(self.p.topic_atomic_secondary_mix)))
        atomic_records: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        atomic_facet_min = max(1, int(self.p.topic_atomic_facet_min))
        atomic_facet_max = max(atomic_facet_min, int(self.p.topic_atomic_facet_max))
        source_vecs = self.topic_facet_vectors[tid, source_local_ids]
        for atomic_idx in range(atomic_count):
            primary_pos = int(sorted_pos[atomic_idx % len(sorted_pos)])
            primary_vec = source_vecs[primary_pos]
            remaining = [pos for pos in sorted_pos if pos != primary_pos]
            secondary_order = sorted(
                remaining,
                key=lambda pos: float(primary_vec @ source_vecs[pos]),
            )
            take = min(2, source_take, int(self.rng.integers(atomic_facet_min, atomic_facet_max + 1)))
            best_record: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
            best_sim = 1e9
            for attempt in range(6):
                chosen_pos = [primary_pos]
                if take > 1 and secondary_order:
                    if attempt < len(secondary_order):
                        sec_pos = int(secondary_order[attempt])
                    else:
                        sec_pos = int(self.rng.choice(secondary_order))
                    chosen_pos.append(sec_pos)

                global_ids = source_global_ids[np.asarray(chosen_pos, dtype=np.int32)].astype(np.int32)
                if len(chosen_pos) > 1:
                    raw_weights = np.asarray(
                        [
                            1.0,
                            secondary_mix * (0.85 + 0.30 * float(self.rng.random())),
                        ],
                        dtype=np.float32,
                    )
                else:
                    raw_weights = np.asarray([1.0], dtype=np.float32)
                weights = self._normalize_simplex(raw_weights)
                vec = self.atomic_memory_vector(tid, global_ids, weights, anchor_vec=turn_vec)
                max_prev_sim = max(
                    (float(vec @ prev_vec) for prev_vec, _prev_ids, _prev_weights in atomic_records),
                    default=-1.0,
                )
                if max_prev_sim < best_sim:
                    best_sim = max_prev_sim
                    best_record = (vec, global_ids, weights)
                if max_prev_sim <= max_self_sim:
                    best_record = (vec, global_ids, weights)
                    break

            if best_record is not None:
                atomic_records.append(best_record)

        return turn_vec, atomic_records

    def atomic_memory_vector(
        self,
        topic_id: int,
        facet_ids: np.ndarray,
        facet_weights: np.ndarray,
        anchor_vec: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        base = self.topic_vectors[int(topic_id)]
        local_ids = (
            np.asarray(facet_ids, dtype=np.int32)
            - int(topic_id) * max(1, int(self.p.topic_facets_per_topic))
        ).astype(np.int32)
        weights = self._normalize_simplex(np.asarray(facet_weights, dtype=np.float32))
        facet_vec = np.zeros(self.p.dim, dtype=np.float32)
        for pos, local_id in enumerate(local_ids.tolist()):
            if local_id < 0 or local_id >= self.topic_facet_vectors.shape[1]:
                continue
            facet_vec += float(weights[pos]) * self.topic_facet_vectors[int(topic_id), int(local_id)]
        noise = self._rand_unit(self.p.dim)
        noise = unit((noise - float(noise @ facet_vec) * facet_vec).astype(np.float32))
        anchor_deflate = 0.0
        if anchor_vec is not None:
            anchor_deflate = max(0.0, min(0.20, float(self.p.topic_atomic_anchor_deflate)))
        vec = unit(
            0.10 * base
            + 0.84 * facet_vec
            + 0.06 * noise
            - anchor_deflate * np.asarray(anchor_vec, dtype=np.float32)
        )
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
        self.max_memories = p.turns * max(1, int(self.p.topic_atomic_memories_max))
        self.max_anchors = p.turns
        self.topic_facet_count = max(1, int(self.p.topic_facets_per_topic))
        self.max_facet_slots = max(
            1,
            int(max(self.p.turn_facet_max, self.p.query_facet_max)),
        )

        self.vec_pool = np.zeros((self.max_memories, p.dim), dtype=np.float32)
        self.mem_topic = np.full(self.max_memories, -1, dtype=np.int32)
        self.mem_turn = np.zeros(self.max_memories, dtype=np.int32)
        self.mem_facet_ids = np.full((self.max_memories, self.max_facet_slots), -1, dtype=np.int32)
        self.mem_facet_weights = np.zeros((self.max_memories, self.max_facet_slots), dtype=np.float32)
        self.mem_origin_topic_counts: List[Dict[int, int]] = []
        self.mem_count = 0
        self.anchor_vec_pool = np.zeros((self.max_anchors, p.dim), dtype=np.float32)
        self.anchor_topic = np.full(self.max_anchors, -1, dtype=np.int32)
        self.anchor_turn = np.zeros(self.max_anchors, dtype=np.int32)
        self.anchor_count = 0
        self.anchor_members: List[List[int]] = []
        self.anchor_neighbors: List[Dict[int, float]] = []
        self.topic_anchor_ids: List[List[int]] = [list() for _ in range(self.topic_count)]
        self.last_anchor_id = -1
        self.last_anchor_by_topic: Dict[int, int] = {}

        # Global DeepARTMAP state (category and bundle are globally unique)
        self.global_topoart: TopoARTState = TopoARTState()
        self.global_deep_artmap: DeepARTMAPState = DeepARTMAPState()
        # Mapping from topic to categories/bundles
        self.topic_to_categories: List[Set[int]] = [set() for _ in range(self.topic_count)]
        self.topic_to_bundles: List[Set[int]] = [set() for _ in range(self.topic_count)]
        # Prior scores per (topic, bundle)
        self.bundle_recall_prior: List[Dict[int, float]] = [{} for _ in range(self.topic_count)]
        self.bundle_adopt_prior: List[Dict[int, float]] = [{} for _ in range(self.topic_count)]
        # Category to bundle mapping (global)
        self.global_category_to_bundle: Dict[int, int] = {}

        self.topic_shards: List[TopicShardState] = [TopicShardState() for _ in range(self.topic_count)]
        self.topic_member_sets: List[Set[int]] = [set() for _ in range(self.topic_count)]
        self.topic_bridges: List[Dict[int, TopicBridgeState]] = [dict() for _ in range(self.topic_count)]
        self.topic_hnsw_index: Optional["HNSWIndex"] = None
        self.topic_family_of = np.full(self.topic_count, -1, dtype=np.int32)
        self.topic_family_members: List[List[int]] = []

        self.turn_topics: List[int] = []
        self.turns_by_topic: Dict[int, List[int]] = {}

        self.query_turns = 0
        self.eval_count = 0
        self.context_eval_count = 0
        self.returned_topics_sum = 0
        self.target_precision_sum = 0.0
        self.target_hit_sum = 0.0
        self.target_mrr_sum = 0.0
        self.context_quality_sum = 0.0
        self.context_coverage_sum = 0.0
        self.context_association_sum = 0.0
        self.context_saturation_sum = 0.0
        self.context_redundancy_sum = 0.0
        self.context_irrelevance_sum = 0.0
        self.context_fill_ratio_sum = 0.0
        self.empty_query_count = 0
        self.empty_target_query_count = 0
        self.new_count = 0
        self.merge_count = 0
        self.sim_ops_add_total = 0
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
        self.topoart_probe_count = 0
        self.topoart_probe_candidates_total = 0
        self.topoart_probe_categories_total = 0
        self.topoart_edge_count_total = 0
        self.deep_artmap_probe_count = 0
        self.deep_artmap_probe_bundle_total = 0
        self.deep_artmap_probe_category_total = 0
        self.topic_hnsw_build_events = 0
        self.topic_hnsw_query_events = 0
        self.topic_hnsw_failure_events = 0
        self.topic_family_rebuild_count = 0
        self.self_excite_penalty_total = 0.0
        self.self_excite_event_count = 0
        self.self_excite_skip_events = 0
        self.self_excite_escape_bonus_total = 0.0
        self.self_excite_escape_event_count = 0
        self.momentum_probe_events = 0
        self.momentum_probe_hits = 0
        self.momentum_probe_total_candidates = 0
        self.momentum_last_turn = 0
        self.momentum_last_topic: Optional[int] = None
        self.momentum_last_memories: List[int] = []
        self.anchor_probe_count = 0
        self.anchor_probe_seed_total = 0
        self.anchor_probe_candidate_total = 0

        self.snapshots: List[Dict[str, float]] = []
        self._build_topic_families()
        self._init_topic_hnsw()
        self.topic_local_index: TopicLocalIndex = self._build_topic_local_index()

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

    def _build_topic_families(self) -> None:
        n = self.topic_count
        if n <= 0:
            self.topic_family_members = []
            return

        self.topic_family_of.fill(-1)
        topk = max(1, min(n - 1, int(self.p.topic_graph_family_topk))) if n > 1 else 0
        thresh = float(self.p.topic_graph_family_similarity)
        if self._topic_family_mode() == "deep_artmap":
            neighbors = self._topic_bundle_overlap_neighbors(topk=topk)
        else:
            sims = self.topic.topic_sim
            neighbors = self._topic_family_neighbors_from_similarity(sims, topk=topk, thresh=thresh)

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for a in range(n):
            for b in neighbors[a]:
                if a in neighbors[int(b)]:
                    union(int(a), int(b))

        roots: Dict[int, List[int]] = {}
        for tid in range(n):
            root = find(tid)
            roots.setdefault(root, []).append(int(tid))

        self.topic_family_members = []
        for family_id, members in enumerate(roots.values()):
            self.topic_family_members.append(list(members))
            for tid in members:
                self.topic_family_of[int(tid)] = int(family_id)
        self.topic_family_rebuild_count += 1

    def _topic_family_neighbors_from_similarity(
        self,
        sims: np.ndarray,
        topk: int,
        thresh: float,
    ) -> List[Set[int]]:
        n = self.topic_count
        neighbors: List[Set[int]] = []
        for tid in range(n):
            if topk <= 0:
                neighbors.append(set())
                continue
            row = np.asarray(sims[tid], dtype=np.float32)
            if topk >= n - 1:
                ordered = np.argsort(row)[::-1].tolist()
            else:
                keep = np.argpartition(row, -(topk + 1))[-(topk + 1):]
                ordered = keep[np.argsort(row[keep])[::-1]].tolist()
            picked: List[int] = []
            for nid in ordered:
                if int(nid) == tid:
                    continue
                if float(row[int(nid)]) < thresh:
                    continue
                picked.append(int(nid))
                if len(picked) >= topk:
                    break
            neighbors.append(set(picked))
        return neighbors

    def _deep_artmap_bundle_signature(
        self,
        tid: int,
    ) -> List[Tuple[np.ndarray, float]]:
        if tid < 0 or tid >= self.topic_count:
            return []
        state = self.global_deep_artmap
        signature: List[Tuple[np.ndarray, float]] = []
        for bundle_id, bundle in enumerate(state.bundles):
            if not bundle.active:
                continue
            if bundle_id not in self.topic_to_bundles[tid]:
                continue
            weight = float(max(1, int(bundle.support)))
            weight += 0.25 * float(self.bundle_recall_prior[tid].get(bundle_id, 0.0))
            weight += 0.35 * float(self.bundle_adopt_prior[tid].get(bundle_id, 0.0))
            if weight <= 0.0:
                continue
            signature.append((bundle.prototype, weight))
        if not signature:
            return []
        total = max(1e-6, float(sum(weight for _proto, weight in signature)))
        return [(proto, float(weight) / total) for proto, weight in signature]

    def _bundle_similarity_matrix(
        self,
        a_tid: int,
        b_tid: int,
    ) -> Tuple[List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float]], Optional[np.ndarray]]:
        sig_a = self._deep_artmap_bundle_signature(a_tid)
        sig_b = self._deep_artmap_bundle_signature(b_tid)
        if not sig_a or not sig_b:
            return sig_a, sig_b, None
        mat = np.zeros((len(sig_a), len(sig_b)), dtype=np.float32)
        for a_pos, (proto_a, _weight_a) in enumerate(sig_a):
            for b_pos, (proto_b, _weight_b) in enumerate(sig_b):
                mat[a_pos, b_pos] = float(proto_a @ proto_b)
        return sig_a, sig_b, mat

    def _pair_bundle_overlap_graph_score(
        self,
        a_tid: int,
        b_tid: int,
    ) -> float:
        if a_tid == b_tid:
            return 1.0
        sig_a, sig_b, sim_mat = self._bundle_similarity_matrix(a_tid, b_tid)
        if sim_mat is None:
            return 0.0
        strong_thresh = max(0.55, float(self.p.topic_graph_family_similarity) - 0.20)
        row_best = np.max(sim_mat, axis=1)
        col_best = np.max(sim_mat, axis=0)
        row_weights = np.asarray([weight for _proto, weight in sig_a], dtype=np.float32)
        col_weights = np.asarray([weight for _proto, weight in sig_b], dtype=np.float32)
        coverage_a = float(np.sum(row_weights[row_best >= strong_thresh]))
        coverage_b = float(np.sum(col_weights[col_best >= strong_thresh]))
        overlap = 0.0
        match_count = 0
        for a_pos, (_proto_a, weight_a) in enumerate(sig_a):
            for b_pos, (_proto_b, weight_b) in enumerate(sig_b):
                sim = float(sim_mat[a_pos, b_pos])
                if sim < strong_thresh:
                    continue
                match_count += 1
                scaled = (sim - strong_thresh) / max(1e-6, 1.0 - strong_thresh)
                overlap += min(float(weight_a), float(weight_b)) * max(0.0, scaled)
        if match_count <= 0:
            return 0.0
        density = min(1.0, float(match_count) / float(max(1, len(sig_a) + len(sig_b) - 1)))
        return float(
            0.55 * math.sqrt(max(0.0, coverage_a) * max(0.0, coverage_b))
            + 0.30 * min(1.0, overlap)
            + 0.15 * density
        )

    def _topic_bundle_overlap_neighbors(
        self,
        topk: int,
    ) -> List[Set[int]]:
        n = self.topic_count
        edge_threshold = max(0.16, float(self.p.topic_graph_family_similarity) - 0.56)
        score_mat = np.zeros((n, n), dtype=np.float32)
        for a_tid in range(n):
            score_mat[a_tid, a_tid] = 1.0
            for b_tid in range(a_tid + 1, n):
                score = self._pair_bundle_overlap_graph_score(a_tid, b_tid)
                score_mat[a_tid, b_tid] = float(score)
                score_mat[b_tid, a_tid] = float(score)
        return self._topic_family_neighbors_from_similarity(score_mat, topk=topk, thresh=edge_threshold)

    def _maybe_rebuild_topic_families(self, turn: int) -> None:
        if self._topic_family_mode() != "deep_artmap":
            return
        interval = max(0, int(TOPIC_FAMILY_DYNAMIC_REBUILD_INTERVAL))
        if interval <= 0 or (int(turn) % interval) != 0:
            return
        self._build_topic_families()

    def _topic_family_id(self, tid: Optional[int]) -> int:
        if tid is None:
            return -1
        topic_id = int(tid)
        if topic_id < 0 or topic_id >= self.topic_count:
            return -1
        return int(self.topic_family_of[topic_id])

    def _self_excitation_adjustment(
        self,
        src_tid: int,
        dst_tid: int,
        current_topic: Optional[int],
        visited_families: Set[int],
    ) -> Tuple[float, float]:
        dst_family = self._topic_family_id(dst_tid)
        if dst_family < 0:
            return 0.0, 0.0

        cur_family = self._topic_family_id(current_topic)
        src_family = self._topic_family_id(src_tid)
        penalty = 0.0
        if current_topic is not None and int(dst_tid) != int(current_topic) and dst_family == cur_family:
            penalty += float(self.p.topic_graph_self_excite_penalty)
        if int(dst_tid) != int(src_tid) and dst_family == src_family:
            penalty += 0.55 * float(self.p.topic_graph_self_excite_penalty)
        if dst_family in visited_families:
            penalty += float(self.p.topic_graph_family_revisit_penalty)

        bonus = 0.0
        if cur_family >= 0 and dst_family != cur_family and dst_family not in visited_families:
            bonus += float(self.p.topic_graph_family_escape_bonus)
        return penalty, bonus

    def _topic_centroid(self, tid: int) -> np.ndarray:
        shard = self.topic_shards[tid]
        if shard.centroid is not None:
            return shard.centroid
        return self.topic.topic_vectors[tid]

    def _build_topic_local_index(self) -> TopicLocalIndex:
        mode = str(getattr(self.p, "topic_graph_local_index", "topoart")).strip().lower()
        if mode == "ghsom":
            return GHSOMTopicLocalIndex()
        if mode == "deep_artmap":
            return DeepARTMAPTopicLocalIndex()
        if mode == "topoart":
            return TopoARTTopicLocalIndex()
        return ExactTopicLocalIndex()

    def _topic_family_mode(self) -> str:
        return "deep_artmap" if self._topic_local_index_mode() == "deep_artmap" else "centroid"

    def _topic_local_index_mode(self) -> str:
        topic_local_index = getattr(self, "topic_local_index", None)
        if topic_local_index is not None:
            return topic_local_index.name
        return str(getattr(self.p, "topic_graph_local_index", "topoart")).strip().lower()

    def _topic_ghsom_active(self) -> bool:
        return self.topic_local_index.ghsom_active(self)

    def _ensure_topic_topoart(self, tid: int) -> TopoARTState:
        # Global topology: all categories are shared across topics
        return self.global_topoart

    def _ensure_topic_deep_artmap(self, tid: int) -> DeepARTMAPState:
        # Global deep ARTMAP: all bundles are shared across topics
        return self.global_deep_artmap

    def _topoart_active_category_ids(self, state: Optional[TopoARTState]) -> List[int]:
        if state is None:
            return []
        return [idx for idx, cat in enumerate(state.categories) if cat.active]

    def _topoart_update_exemplars(
        self,
        cat: TopoARTCategoryState,
        mem_idx: int,
        score: float,
    ) -> None:
        limit = max(1, int(self.p.topic_graph_topoart_exemplars))
        if mem_idx in cat.exemplars:
            pos = cat.exemplars.index(mem_idx)
            cat.exemplar_scores[pos] = max(float(cat.exemplar_scores[pos]), float(score))
            return
        if len(cat.exemplars) < limit:
            cat.exemplars.append(int(mem_idx))
            cat.exemplar_scores.append(float(score))
            return
        worst_pos = int(np.argmin(np.asarray(cat.exemplar_scores, dtype=np.float32)))
        if float(score) > float(cat.exemplar_scores[worst_pos]):
            cat.exemplars[worst_pos] = int(mem_idx)
            cat.exemplar_scores[worst_pos] = float(score)

    def _topoart_link_categories(
        self,
        state: TopoARTState,
        a_idx: int,
        b_idx: int,
    ) -> None:
        if a_idx == b_idx:
            return
        cat_a = state.categories[a_idx]
        cat_b = state.categories[b_idx]
        cat_a.neighbors[b_idx] = float(cat_a.neighbors.get(b_idx, 0.0)) + 1.0
        cat_b.neighbors[a_idx] = float(cat_b.neighbors.get(a_idx, 0.0)) + 1.0

    def _topoart_prune(self, state: TopoARTState, turn: int) -> None:
        interval = max(0, int(self.p.topic_graph_topoart_prune_interval))
        if interval <= 0 or state.update_count <= 0 or (state.update_count % interval) != 0:
            return
        min_support = max(1, int(self.p.topic_graph_topoart_prune_min_support))
        stale_before = int(turn) - interval
        retired: List[int] = []
        for idx, cat in enumerate(state.categories):
            if not cat.active:
                continue
            weak = int(cat.support) < min_support and int(cat.member_count) <= min_support
            stale = int(cat.last_update_turn) <= stale_before
            if weak and stale:
                cat.active = False
                retired.append(idx)
        if not retired:
            return
        retired_set = set(retired)
        for cat in state.categories:
            if not cat.active or not cat.neighbors:
                continue
            for rid in retired:
                cat.neighbors.pop(rid, None)
            dead = [nid for nid in cat.neighbors if nid in retired_set]
            for nid in dead:
                cat.neighbors.pop(nid, None)

    def _topoart_category_vigilance(self, cat: TopoARTCategoryState) -> float:
        base = float(self.p.topic_graph_topoart_vigilance)
        if int(cat.member_count) < 4:
            return base
        slack = max(0.0, float(self.p.topic_graph_topoart_match_slack))
        adaptive = max(base, float(cat.match_ema) - slack)
        cap = max(1, int(self.p.topic_graph_topoart_category_capacity))
        if int(cat.member_count) > cap:
            over = float(int(cat.member_count) - cap) / float(cap)
            adaptive += float(self.p.topic_graph_topoart_capacity_boost) * min(2.0, over)
        return min(0.995, adaptive)

    def _topoart_insert_with_result(
        self,
        tid: int,
        vec: np.ndarray,
        mem_idx: int,
        turn: int,
    ) -> TopoARTInsertResult:
        state = self._ensure_topic_topoart(tid)
        state.update_count += 1
        active_ids = self._topoart_active_category_ids(state)
        result = TopoARTInsertResult()
        vigilance = float(self.p.topic_graph_topoart_vigilance)
        secondary_vigilance = min(vigilance, float(self.p.topic_graph_topoart_secondary_vigilance))
        beta = min(1.0, max(0.01, float(self.p.topic_graph_topoart_beta)))
        beta_secondary = min(1.0, max(0.0, float(self.p.topic_graph_topoart_beta_secondary)))
        match_alpha = min(0.5, max(0.05, 0.5 * beta))
        link_margin = max(0.0, float(self.p.topic_graph_topoart_link_margin))
        temporal_link_window = max(0, int(self.p.topic_graph_topoart_temporal_link_window))

        def create_category() -> int:
            state.categories.append(
                TopoARTCategoryState(
                    prototype=vec.astype(np.float32, copy=True),
                    vector_sum=vec.astype(np.float32, copy=True),
                    support=1,
                    member_count=1,
                    exemplars=[int(mem_idx)],
                    exemplar_scores=[1.0],
                    last_update_turn=int(turn),
                    match_ema=1.0,
                    match_min=1.0,
                )
            )
            new_id = len(state.categories) - 1
            self.topic_to_categories[tid].add(new_id)
            prev_id = int(state.last_winner_id)
            if (
                temporal_link_window > 0
                and prev_id >= 0
                and prev_id != new_id
                and (int(turn) - int(state.last_winner_turn)) <= temporal_link_window
                and prev_id < len(state.categories)
                and state.categories[prev_id].active
            ):
                self._topoart_link_categories(state, prev_id, new_id)
            state.last_winner_id = new_id
            state.last_winner_turn = int(turn)
            self._topoart_prune(state, turn)
            return new_id

        if not active_ids:
            result.winner_id = create_category()
            result.created = True
            return result

        protos = np.vstack([state.categories[idx].prototype for idx in active_ids]).astype(np.float32)
        sims = protos @ vec
        result.ops += int(protos.shape[0])
        order = np.argsort(sims)[::-1].tolist()

        winner_id: Optional[int] = None
        second_id: Optional[int] = None
        winner_sim = -1.0
        runner_up_id: Optional[int] = None
        runner_up_sim = -1.0
        for pos in order:
            sim = float(sims[pos])
            cat_id = int(active_ids[pos])
            cat = state.categories[cat_id]
            effective_vigilance = self._topoart_category_vigilance(cat)
            if winner_id is None and sim >= effective_vigilance:
                winner_id = cat_id
                winner_sim = sim
                continue
            if winner_id is None:
                continue
            if cat_id == winner_id:
                continue
            if runner_up_id is None:
                runner_up_id = cat_id
                runner_up_sim = sim
            if sim >= secondary_vigilance:
                second_id = cat_id
                break

        if winner_id is None:
            result.winner_id = create_category()
            result.created = True
            return result

        winner = state.categories[winner_id]
        winner.vector_sum += vec
        winner.prototype = unit(winner.vector_sum)
        winner.support += 1
        winner.member_count += 1
        winner.last_update_turn = int(turn)
        winner.match_ema = (1.0 - match_alpha) * float(winner.match_ema) + match_alpha * float(winner_sim)
        winner.match_min = min(float(winner.match_min), float(winner_sim))
        self._topoart_update_exemplars(winner, mem_idx, max(winner_sim, float(winner.prototype @ vec)))

        if second_id is None and runner_up_id is not None and (winner_sim - runner_up_sim) <= link_margin:
            second_id = runner_up_id
        if second_id is not None:
            second = state.categories[second_id]
            # centroid update removed for second category (global geometry)
            # second.prototype remains unchanged
            second.last_update_turn = int(turn)
            second.match_ema = (1.0 - match_alpha) * float(second.match_ema) + match_alpha * float(max(0.0, runner_up_sim))
            second.match_min = min(float(second.match_min), float(max(0.0, runner_up_sim)))
            self._topoart_link_categories(state, winner_id, second_id)

        prev_id = int(state.last_winner_id)
        if (
            temporal_link_window > 0
            and prev_id >= 0
            and prev_id != winner_id
            and (int(turn) - int(state.last_winner_turn)) <= temporal_link_window
            and prev_id < len(state.categories)
            and state.categories[prev_id].active
        ):
            self._topoart_link_categories(state, prev_id, winner_id)
        state.last_winner_id = winner_id
        state.last_winner_turn = int(turn)
        self._topoart_prune(state, turn)
        result.winner_id = int(winner_id)
        result.second_id = int(second_id) if second_id is not None else -1
        return result

    def _topoart_insert(
        self,
        tid: int,
        vec: np.ndarray,
        mem_idx: int,
        turn: int,
    ) -> int:
        return self._topoart_insert_with_result(tid, vec, mem_idx, turn).ops

    def _topoart_collect_candidates(
        self,
        tid: int,
        query_vec: np.ndarray,
        max_results: Optional[int],
    ) -> Tuple[np.ndarray, float, int]:
        if tid < 0 or tid >= self.topic_count:
            return np.asarray([], dtype=np.int32), 0.0, 0
        state = self._ensure_topic_topoart(tid)
        active_ids = self._topoart_active_category_ids(state)
        if not active_ids:
            return np.asarray([], dtype=np.int32), 0.0, 0

        protos = np.vstack([state.categories[idx].prototype for idx in active_ids]).astype(np.float32)
        sims = protos @ query_vec
        ops = int(protos.shape[0])
        order = np.argsort(sims)[::-1].tolist()
        if not order:
            return np.asarray([], dtype=np.int32), 0.0, ops

        best_sim = float(sims[order[0]])
        if best_sim < float(self.p.topic_graph_topoart_secondary_vigilance):
            return np.asarray([], dtype=np.int32), best_sim, ops

        query_categories = max(1, int(self.p.topic_graph_topoart_query_categories))
        neighbor_topk = max(0, int(self.p.topic_graph_topoart_neighbor_topk))
        query_margin = max(0.0, float(self.p.topic_graph_topoart_query_margin))
        selected: List[int] = [int(active_ids[order[0]])]
        selected_set: Set[int] = {int(active_ids[order[0]])}

        for pos in order[1:]:
            cat_id = int(active_ids[pos])
            sim = float(sims[pos])
            if sim < float(self.p.topic_graph_topoart_vigilance):
                break
            selected.append(cat_id)
            selected_set.add(cat_id)
            if len(selected) >= query_categories:
                break

        neighbor_pool: List[Tuple[float, int]] = []
        for cat_id in list(selected):
            cat = state.categories[cat_id]
            if not cat.neighbors:
                continue
            ranked_neighbors = sorted(cat.neighbors.items(), key=lambda it: it[1], reverse=True)
            for nid, strength in ranked_neighbors[:neighbor_topk]:
                neighbor = state.categories[int(nid)]
                if not neighbor.active or int(nid) in selected_set:
                    continue
                neighbor_score = float(strength) + 0.35 * float(neighbor.prototype @ query_vec)
                neighbor_pool.append((neighbor_score, int(nid)))
                self.topoart_edge_count_total += 1

        neighbor_pool.sort(key=lambda it: it[0], reverse=True)
        for _score, nid in neighbor_pool:
            if len(selected) >= query_categories:
                break
            if nid in selected_set:
                continue
            selected.append(int(nid))
            selected_set.add(int(nid))
            if len(selected) >= query_categories:
                break

        query_floor = max(
            float(self.p.topic_graph_topoart_secondary_vigilance),
            best_sim - query_margin,
        )
        for pos in order:
            if len(selected) >= query_categories:
                break
            cat_id = int(active_ids[pos])
            sim = float(sims[pos])
            if sim < query_floor:
                break
            if cat_id in selected_set:
                continue
            selected.append(cat_id)
            selected_set.add(cat_id)
            if len(selected) >= query_categories:
                break

        candidate_target = max(4, int(max_results or 1) * 4)
        candidates: List[int] = []
        seen: Set[int] = set()
        proto_score = 0.0
        for cat_id in selected:
            cat = state.categories[cat_id]
            proto_score = max(proto_score, float(cat.prototype @ query_vec))
            ranked_exemplars = sorted(
                zip(cat.exemplars, cat.exemplar_scores),
                key=lambda it: it[1],
                reverse=True,
            )
            for mem_idx, _score in ranked_exemplars:
                if int(mem_idx) in seen:
                    continue
                seen.add(int(mem_idx))
                candidates.append(int(mem_idx))
                if len(candidates) >= candidate_target:
                    break
            if len(candidates) >= candidate_target:
                break

        self.topoart_probe_count += 1
        self.topoart_probe_categories_total += len(selected)
        self.topoart_probe_candidates_total += len(candidates)
        return np.asarray(candidates, dtype=np.int32), proto_score, ops

    def _deep_artmap_active_bundle_ids(self, state: Optional[DeepARTMAPState]) -> List[int]:
        if state is None:
            return []
        return [idx for idx, bundle in enumerate(state.bundles) if bundle.active]

    def _deep_artmap_link_bundles(
        self,
        state: DeepARTMAPState,
        a_idx: int,
        b_idx: int,
    ) -> None:
        if a_idx == b_idx or a_idx < 0 or b_idx < 0:
            return
        bundle_a = state.bundles[a_idx]
        bundle_b = state.bundles[b_idx]
        bundle_a.neighbors[b_idx] = float(bundle_a.neighbors.get(b_idx, 0.0)) + 1.0
        bundle_b.neighbors[a_idx] = float(bundle_b.neighbors.get(a_idx, 0.0)) + 1.0

    def _deep_artmap_reinforce_bundle(
        self,
        tid: int,
        bundle_id: int,
        kind: str,
        lr: float,
    ) -> None:
        if tid < 0 or tid >= self.topic_count or bundle_id < 0:
            return
        # Check bundle exists in global deep artmap state
        if bundle_id >= len(self.global_deep_artmap.bundles):
            return
        bundle = self.global_deep_artmap.bundles[bundle_id]
        if not bundle.active:
            return
        rate = max(0.0, float(lr))
        if kind == "recall":
            prior = self.bundle_recall_prior[tid].get(bundle_id, 0.0)
            new_prior = min(1.0, prior + rate * (1.0 - prior))
            self.bundle_recall_prior[tid][bundle_id] = new_prior
        elif kind == "adopt":
            prior = self.bundle_adopt_prior[tid].get(bundle_id, 0.0)
            new_prior = min(1.0, prior + rate * (1.0 - prior))
            self.bundle_adopt_prior[tid][bundle_id] = new_prior

    def _decay_deep_artmap_feedback(self) -> None:
        decay = min(0.9999, max(0.0, float(self.p.topic_graph_deep_artmap_bundle_decay)))
        # Decay per-topic prior maps
        for tid in range(self.topic_count):
            for bundle_id, prior in list(self.bundle_recall_prior[tid].items()):
                new_prior = prior * decay
                if new_prior < 0.001:
                    del self.bundle_recall_prior[tid][bundle_id]
                else:
                    self.bundle_recall_prior[tid][bundle_id] = new_prior
            for bundle_id, prior in list(self.bundle_adopt_prior[tid].items()):
                new_prior = prior * decay
                if new_prior < 0.001:
                    del self.bundle_adopt_prior[tid][bundle_id]
                else:
                    self.bundle_adopt_prior[tid][bundle_id] = new_prior
        # Decay global bundle neighbor strengths
        for bundle in self.global_deep_artmap.bundles:
            if not bundle.active:
                continue
            if bundle.neighbors:
                drop: List[int] = []
                for nid, strength in bundle.neighbors.items():
                    new_strength = float(strength) * decay
                    bundle.neighbors[nid] = new_strength
                    if new_strength < 0.02:
                        drop.append(int(nid))
                for nid in drop:
                    bundle.neighbors.pop(nid, None)

    def _deep_artmap_assign_bundle(
        self,
        tid: int,
        category_id: int,
        turn: int,
    ) -> Tuple[int, int]:
        topo_state = self._ensure_topic_topoart(tid)
        if category_id < 0 or category_id >= len(topo_state.categories):
            return -1, 0
        deep_state = self._ensure_topic_deep_artmap(tid)
        category = topo_state.categories[category_id]
        bundle_vigilance = float(self.p.topic_graph_deep_artmap_bundle_vigilance)
        bundle_beta = min(1.0, max(0.01, float(self.p.topic_graph_deep_artmap_bundle_beta)))
        temporal_window = max(0, int(self.p.topic_graph_deep_artmap_temporal_link_window))
        ops = 0

        existing = deep_state.category_to_bundle.get(int(category_id), -1)
        assigned = -1
        if 0 <= existing < len(deep_state.bundles) and deep_state.bundles[existing].active:
            assigned = int(existing)
            self.topic_to_bundles[tid].add(assigned)
        else:
            active_bundle_ids = self._deep_artmap_active_bundle_ids(deep_state)
            if active_bundle_ids:
                protos = np.vstack([deep_state.bundles[idx].prototype for idx in active_bundle_ids]).astype(np.float32)
                sims = protos @ category.prototype
                ops += int(protos.shape[0])
                best_pos = int(np.argmax(sims))
                best_bundle_id = int(active_bundle_ids[best_pos])
                best_sim = float(sims[best_pos])
                if best_sim >= bundle_vigilance:
                    assigned = best_bundle_id
                    self.topic_to_bundles[tid].add(assigned)
            if assigned < 0:
                deep_state.bundles.append(
                    DeepARTMAPBundleState(
                        prototype=category.prototype.astype(np.float32, copy=True),
                        vector_sum=category.prototype.astype(np.float32, copy=True),
                        category_ids=[int(category_id)],
                        support=0,
                        last_update_turn=int(turn),
                    )
                )
                assigned = len(deep_state.bundles) - 1
                self.topic_to_bundles[tid].add(assigned)

        bundle = deep_state.bundles[assigned]
        if int(category_id) not in bundle.category_ids:
            bundle.category_ids.append(int(category_id))
        bundle.vector_sum += category.prototype
        bundle.prototype = unit(bundle.vector_sum)
        bundle.support += 1
        bundle.last_update_turn = int(turn)
        deep_state.category_to_bundle[int(category_id)] = int(assigned)
        self.global_category_to_bundle[int(category_id)] = int(assigned)

        prev_bundle = int(deep_state.last_bundle_id)
        if (
            temporal_window > 0
            and prev_bundle >= 0
            and prev_bundle != assigned
            and prev_bundle < len(deep_state.bundles)
            and deep_state.bundles[prev_bundle].active
            and (int(turn) - int(deep_state.last_bundle_turn)) <= temporal_window
        ):
            self._deep_artmap_link_bundles(deep_state, prev_bundle, assigned)
        deep_state.last_bundle_id = int(assigned)
        deep_state.last_bundle_turn = int(turn)
        return int(assigned), ops

    def _deep_artmap_insert(
        self,
        tid: int,
        vec: np.ndarray,
        mem_idx: int,
        turn: int,
    ) -> int:
        insert_result = self._topoart_insert_with_result(tid, vec, mem_idx, turn)
        total_ops = int(insert_result.ops)
        if insert_result.winner_id < 0:
            return total_ops
        winner_bundle, bundle_ops = self._deep_artmap_assign_bundle(tid, int(insert_result.winner_id), turn)
        total_ops += bundle_ops
        if insert_result.second_id >= 0:
            second_bundle, second_ops = self._deep_artmap_assign_bundle(tid, int(insert_result.second_id), turn)
            total_ops += second_ops
            if winner_bundle >= 0 and second_bundle >= 0 and winner_bundle != second_bundle:
                deep_state = self._ensure_topic_deep_artmap(tid)
                self._deep_artmap_link_bundles(deep_state, winner_bundle, second_bundle)
        return total_ops

    def _deep_artmap_collect_candidates(
        self,
        tid: int,
        query_vec: np.ndarray,
        max_results: int,
        query_facet_ids: Optional[np.ndarray] = None,
        query_facet_weights: Optional[np.ndarray] = None,
    ) -> TopicLocalQueryResult:
        if tid < 0 or tid >= self.topic_count:
            return TopicLocalQueryResult()
        shard = self.topic_shards[tid]
        if len(shard.members) < max(2, int(self.p.topic_graph_deep_artmap_min_members)):
            return TopicLocalQueryResult()
        topo_state = self._ensure_topic_topoart(tid)
        deep_state = self._ensure_topic_deep_artmap(tid)

        active_bundle_ids = self._deep_artmap_active_bundle_ids(deep_state)
        if not active_bundle_ids:
            idx, score, ops = self._topoart_collect_candidates(tid, query_vec, max_results=max_results)
            return TopicLocalQueryResult(indices=idx, score=score, ops=ops)

        protos = np.vstack([deep_state.bundles[idx].prototype for idx in active_bundle_ids]).astype(np.float32)
        sims = protos @ query_vec
        ops = int(protos.shape[0])
        order = np.argsort(sims)[::-1].tolist()
        if not order:
            return TopicLocalQueryResult()

        prior_weight = max(0.0, float(self.p.topic_graph_deep_artmap_bundle_prior_weight))
        bundle_scores = np.asarray(
            [
                float(sims[pos])
                + prior_weight
                * (
                    0.65 * float(self.bundle_recall_prior[tid].get(int(active_bundle_ids[pos]), 0.0))
                    + 1.00 * float(self.bundle_adopt_prior[tid].get(int(active_bundle_ids[pos]), 0.0))
                )
                for pos in range(len(active_bundle_ids))
            ],
            dtype=np.float32,
        )
        order = np.argsort(bundle_scores)[::-1].tolist()
        best_sim = float(bundle_scores[order[0]])
        bundle_vigilance = float(self.p.topic_graph_deep_artmap_bundle_vigilance)
        if best_sim < bundle_vigilance:
            idx, score, topo_ops = self._topoart_collect_candidates(tid, query_vec, max_results=max_results)
            return TopicLocalQueryResult(indices=idx, score=score, ops=ops + topo_ops)

        query_bundles = max(1, int(self.p.topic_graph_deep_artmap_query_bundles))
        bundle_margin = max(0.0, float(self.p.topic_graph_deep_artmap_query_margin))
        neighbor_topk = max(0, int(self.p.topic_graph_deep_artmap_neighbor_topk))
        query_map = self._build_query_facet_map(query_facet_ids, query_facet_weights)

        selected_bundles: List[int] = [int(active_bundle_ids[order[0]])]
        selected_bundle_set: Set[int] = {int(active_bundle_ids[order[0]])}
        bundle_floor = max(bundle_vigilance, best_sim - bundle_margin)
        for pos in order[1:]:
            if len(selected_bundles) >= query_bundles:
                break
            bundle_id = int(active_bundle_ids[pos])
            if float(bundle_scores[pos]) < bundle_floor:
                break
            selected_bundles.append(bundle_id)
            selected_bundle_set.add(bundle_id)

        neighbor_pool: List[Tuple[float, int]] = []
        for bundle_id in list(selected_bundles):
            bundle = deep_state.bundles[bundle_id]
            ranked_neighbors = sorted(bundle.neighbors.items(), key=lambda it: it[1], reverse=True)
            for nid, strength in ranked_neighbors[:neighbor_topk]:
                if nid in selected_bundle_set:
                    continue
                neighbor = deep_state.bundles[int(nid)]
                if not neighbor.active:
                    continue
                prior = prior_weight * (0.65 * float(self.bundle_recall_prior[tid].get(int(nid), 0.0)) + 1.00 * float(self.bundle_adopt_prior[tid].get(int(nid), 0.0)))
                score = float(strength) + 0.35 * float(neighbor.prototype @ query_vec) + prior
                neighbor_pool.append((score, int(nid)))

        neighbor_pool.sort(key=lambda it: it[0], reverse=True)
        for _score, bundle_id in neighbor_pool:
            if len(selected_bundles) >= query_bundles:
                break
            if bundle_id in selected_bundle_set:
                continue
            selected_bundles.append(int(bundle_id))
            selected_bundle_set.add(int(bundle_id))

        query_categories = max(1, int(self.p.topic_graph_topoart_query_categories))
        category_budget = max(query_categories, len(selected_bundles))
        bundle_category_rows: Dict[int, List[Tuple[float, int]]] = {}
        candidate_categories: List[Tuple[float, int, int]] = []
        spill_categories: List[Tuple[float, int, int]] = []
        seen_categories: Set[int] = set()
        bundle_query_scores = {
            int(active_bundle_ids[pos]): float(bundle_scores[pos]) for pos in range(len(active_bundle_ids))
        }
        bundle_score = 0.0
        for bundle_id in selected_bundles:
            bundle = deep_state.bundles[bundle_id]
            bundle_query_score = max(
                float(bundle.prototype @ query_vec),
                float(bundle_query_scores.get(int(bundle_id), float(bundle.prototype @ query_vec))),
            )
            bundle_score = max(bundle_score, bundle_query_score)
            active_categories = [
                cid for cid in bundle.category_ids
                if 0 <= cid < len(topo_state.categories) and topo_state.categories[cid].active
            ]
            if not active_categories:
                continue
            cat_protos = np.vstack([topo_state.categories[cid].prototype for cid in active_categories]).astype(np.float32)
            cat_sims = cat_protos @ query_vec
            ops += int(cat_protos.shape[0])
            rows = [(float(cat_sims[pos]), int(active_categories[pos])) for pos in np.argsort(cat_sims)[::-1].tolist()]
            bundle_category_rows[int(bundle_id)] = rows

        if bundle_category_rows:
            per_bundle_budget = max(1, (category_budget + len(selected_bundles) - 1) // max(1, len(selected_bundles)))
            for bundle_id in selected_bundles:
                rows = bundle_category_rows.get(int(bundle_id), [])
                kept = 0
                for cat_score, cid in rows:
                    if cid in seen_categories:
                        continue
                    row = (float(cat_score), int(cid), int(bundle_id))
                    if kept < per_bundle_budget:
                        candidate_categories.append(row)
                        seen_categories.add(int(cid))
                        kept += 1
                    else:
                        spill_categories.append(row)

            if len(candidate_categories) < category_budget:
                for row in sorted(spill_categories, key=lambda it: it[0], reverse=True):
                    _cat_score, cid, _bundle_id = row
                    if int(cid) in seen_categories:
                        continue
                    candidate_categories.append(row)
                    seen_categories.add(int(cid))
                    if len(candidate_categories) >= category_budget:
                        break

        selected_categories = [int(cid) for _score, cid, _bundle_id in candidate_categories[:category_budget]]

        candidate_target = max(4, int(max_results or 1) * 4)
        candidate_local_scores: Dict[int, float] = {}
        candidate_bundle_ids: Dict[int, int] = {}
        semantic_scores: Dict[int, float] = {}
        proto_score = bundle_score
        for cat_score, cid, bundle_id in candidate_categories[:category_budget]:
            cat = topo_state.categories[cid]
            proto_score = max(proto_score, float(cat.prototype @ query_vec))
            ranked_exemplars = sorted(
                zip(cat.exemplars, cat.exemplar_scores),
                key=lambda it: it[1],
                reverse=True,
            )
            bundle_query_score = float(bundle_query_scores.get(int(bundle_id), float(cat_score)))
            for mem_idx, exemplar_score in ranked_exemplars:
                mem_id = int(mem_idx)
                if mem_id < 0 or mem_id >= self.mem_count:
                    continue
                semantic = float(self.vec_pool[mem_id] @ query_vec)
                local_score = (
                    0.55 * max(0.0, float(cat_score))
                    + 0.25 * max(0.0, bundle_query_score)
                    + 0.20 * max(0.0, float(exemplar_score))
                )
                if local_score > float(candidate_local_scores.get(mem_id, -1e9)):
                    candidate_local_scores[mem_id] = float(local_score)
                    candidate_bundle_ids[mem_id] = int(bundle_id)
                if semantic > float(semantic_scores.get(mem_id, -1e9)):
                    semantic_scores[mem_id] = float(semantic)
                ops += 1

        ranked_memories = self._select_memories_for_context(
            memory_ids=list(candidate_local_scores.keys()),
            query_vec=query_vec,
            query_map=query_map,
            budget=candidate_target,
            semantic_scores=semantic_scores,
            local_scores=candidate_local_scores,
            group_ids=candidate_bundle_ids,
        )
        candidates = [int(mem_id) for mem_id, _score in ranked_memories]
        memory_scores = [float(score) for _mem_id, score in ranked_memories]
        memory_bundle_ids = [int(candidate_bundle_ids.get(int(mem_id), -1)) for mem_id in candidates]

        self.deep_artmap_probe_count += 1
        self.deep_artmap_probe_bundle_total += len(selected_bundles)
        self.deep_artmap_probe_category_total += len(selected_categories)
        return TopicLocalQueryResult(
            indices=np.asarray(candidates, dtype=np.int32),
            score=proto_score,
            ops=ops,
            bundle_ids=[int(bid) for bid in selected_bundles],
            category_ids=[int(cid) for cid in selected_categories],
            memory_scores=memory_scores,
            memory_bundle_ids=memory_bundle_ids,
        )

    def _topic_hnsw_ready(self) -> bool:
        return bool(self.p.topic_graph_topic_hnsw_enabled and HNSWIndex is not None)

    def _init_topic_hnsw(self) -> None:
        if not self._topic_hnsw_ready():
            return
        try:
            index = HNSWIndex.create(
                dim=self.p.dim,
                max_elements=max(8, self.topic_count),
                space="cosine",
                m=max(4, int(self.p.topic_graph_topic_hnsw_m)),
                ef_construction=max(16, int(self.p.topic_graph_topic_hnsw_ef_construction)),
                ef_search=max(8, int(self.p.topic_graph_topic_hnsw_ef_search)),
                random_seed=int(self.p.seed),
            )
            for tid in range(self.topic_count):
                index.add(int(tid), self.topic.topic_vectors[tid])
        except Exception:
            self.topic_hnsw_failure_events += 1
            self.topic_hnsw_index = None
            return
        self.topic_hnsw_index = index
        self.topic_hnsw_build_events += 1

    def _semantic_topic_score(
        self,
        tid: int,
        query_vec: np.ndarray,
        current_topic: Optional[int],
    ) -> float:
        score = float(self.p.topic_graph_query_semantic_weight) * float(self._topic_centroid(tid) @ query_vec)
        if current_topic is not None and tid == int(current_topic):
            score += float(self.p.topic_graph_current_topic_bonus)
        return score

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

    def _add_memory(
        self,
        vec: np.ndarray,
        turn: int,
        topic_id: int,
        facet_ids: Optional[np.ndarray] = None,
        facet_weights: Optional[np.ndarray] = None,
    ) -> Tuple[int, List[int]]:
        merge_target, merge_sim, add_ops = self._find_merge_target(vec, topic_id)
        if merge_target is not None and merge_sim >= float(self.p.merge_limit):
            self._merge_memory_facets(merge_target, facet_ids, facet_weights)
            add_ops += self._bind_memory_topic(
                merge_target,
                topic_id,
                self.vec_pool[merge_target],
                turn,
                amount=1,
            )
            self.merge_count += 1
            return int(add_ops), [int(merge_target)]

        if self.mem_count >= self.max_memories:
            return int(add_ops), []
        mem_idx = self.mem_count
        self.vec_pool[mem_idx] = vec
        self.mem_topic[mem_idx] = topic_id
        self.mem_turn[mem_idx] = turn
        self.mem_facet_ids[mem_idx].fill(-1)
        self.mem_facet_weights[mem_idx].fill(0.0)
        if facet_ids is not None and facet_weights is not None:
            limit = min(
                self.max_facet_slots,
                int(len(facet_ids)),
                int(len(facet_weights)),
            )
            if limit > 0:
                self.mem_facet_ids[mem_idx, :limit] = np.asarray(facet_ids[:limit], dtype=np.int32)
                self.mem_facet_weights[mem_idx, :limit] = np.asarray(facet_weights[:limit], dtype=np.float32)
        self.mem_count += 1
        self.new_count += 1
        self.mem_origin_topic_counts.append({})
        add_ops += self._bind_memory_topic(mem_idx, topic_id, vec, turn, amount=1)
        return int(add_ops), [int(mem_idx)]

    def _update_one_turn_momentum(
        self,
        turn: int,
        topic_id: int,
        memory_ids: Sequence[int],
    ) -> None:
        cleaned: List[int] = []
        seen: Set[int] = set()
        cap = max(1, int(self.p.topic_graph_momentum_probe_cap))
        for mem_idx in memory_ids:
            mem_id = int(mem_idx)
            if mem_id < 0 or mem_id >= self.mem_count or mem_id in seen:
                continue
            seen.add(mem_id)
            cleaned.append(mem_id)
            if len(cleaned) >= cap:
                break

        self.momentum_last_turn = int(turn)
        self.momentum_last_topic = int(topic_id)
        self.momentum_last_memories = cleaned

    def _link_anchor_nodes(self, a: int, b: int, weight: float) -> None:
        if a < 0 or b < 0 or a == b:
            return
        w = max(0.0, float(weight))
        if w <= 0.0:
            return
        while len(self.anchor_neighbors) <= max(a, b):
            self.anchor_neighbors.append({})
        self.anchor_neighbors[a][b] = max(w, float(self.anchor_neighbors[a].get(b, 0.0)))
        self.anchor_neighbors[b][a] = max(w, float(self.anchor_neighbors[b].get(a, 0.0)))

    def _add_turn_anchor(
        self,
        vec: np.ndarray,
        turn: int,
        topic_id: int,
        memory_ids: Sequence[int],
    ) -> int:
        # Retired experiment: turn-anchor routing consistently widened the
        # candidate pool without improving recall or context assembly quality.
        _ = (vec, turn, topic_id, memory_ids)
        return -1

    def _anchor_graph_evidence(
        self,
        query_vec: np.ndarray,
        available_topics: Set[int],
    ) -> Tuple[Dict[int, List[int]], Dict[str, object], int]:
        _ = (query_vec, available_topics)
        return {}, {"anchor_seed_ids": [], "anchor_visit_ids": [], "anchor_memory_ids": []}, 0

    def _one_turn_momentum_probe(
        self,
        query_vec: np.ndarray,
        current_topic: Optional[int],
        current_turn: int,
        semantic_scores: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[Tuple[int, float]], int]:
        if not self.p.topic_graph_momentum_probe_enabled:
            return [], 0
        if current_topic is None or self.momentum_last_topic is None:
            return [], 0
        if int(current_turn) != int(self.momentum_last_turn) + 1:
            return [], 0
        if int(current_topic) != int(self.momentum_last_topic):
            return [], 0
        if not self.momentum_last_memories:
            return [], 0
        if semantic_scores is None or int(current_topic) not in semantic_scores:
            return [], 0
        current_score = float(semantic_scores.get(int(current_topic), -1e9))
        current_score -= float(self.p.topic_graph_current_topic_bonus)
        best_score = max((float(score) for score in semantic_scores.values()), default=-1e9)
        margin = max(0.0, float(self.p.topic_graph_momentum_probe_topic_margin))
        if current_score + margin < best_score:
            return [], 0

        self.momentum_probe_events += 1
        min_sim = max(-1.0, min(1.0, float(self.p.topic_graph_momentum_probe_min_sim)))
        scored: List[Tuple[int, float]] = []
        ops = 0
        for mem_idx in self.momentum_last_memories:
            mem_id = int(mem_idx)
            if mem_id < 0 or mem_id >= self.mem_count:
                continue
            ops += 1
            sim = float(self.vec_pool[mem_id] @ query_vec)
            if sim < min_sim:
                continue
            scored.append((mem_id, sim))

        scored.sort(key=lambda it: it[1], reverse=True)
        cap = max(1, int(self.p.topic_graph_momentum_probe_cap))
        picked = scored[:cap]
        if picked:
            self.momentum_probe_hits += 1
            self.momentum_probe_total_candidates += len(picked)
        return picked, ops

    def _semantic_topic_scores(
        self,
        query_vec: np.ndarray,
        current_topic: Optional[int],
    ) -> Tuple[Dict[int, float], int]:
        known = self._known_topic_ids()
        if known.size <= 0:
            return {}, 0
        scores: Dict[int, float] = {}
        if self._topic_hnsw_ready() and self.topic_hnsw_index is not None:
            want = min(
                self.topic_count,
                max(
                    int(self.p.topic_graph_topic_hnsw_k),
                    int(self.p.topic_graph_seed_topics) * 4,
                    int(self.p.topic_graph_load_budget) * 3,
                ),
            )
            try:
                hits = self.topic_hnsw_index.search(query_vec, want)
                self.topic_hnsw_query_events += 1
            except Exception:
                self.topic_hnsw_failure_events += 1
                hits = []
            active_ids = set(int(tid) for tid in known.tolist())
            for hit in hits:
                tid = int(hit.label)
                if tid not in active_ids:
                    continue
                score = float(self.p.topic_graph_query_semantic_weight) * float(hit.similarity)
                if current_topic is not None and tid == int(current_topic):
                    score += float(self.p.topic_graph_current_topic_bonus)
                scores[tid] = score
            if current_topic is not None and current_topic in active_ids and int(current_topic) not in scores:
                scores[int(current_topic)] = self._semantic_topic_score(int(current_topic), query_vec, current_topic)
            if len(scores) >= max(1, int(self.p.topic_graph_seed_topics)):
                ops = min(
                    int(known.size),
                    max(want, int(self.p.topic_graph_topic_hnsw_ef_search)),
                )
                return scores, ops

        centers = np.vstack([self._topic_centroid(int(tid)) for tid in known.tolist()]).astype(np.float32)
        sims = centers @ query_vec
        ops = int(centers.shape[0])
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
        query_vec: np.ndarray,
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
        visited_families: Set[int] = set()
        kept_family_counts: Dict[int, int] = {}
        family_limit = max(1, int(self.p.topic_graph_family_member_limit))

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
            family_id = self._topic_family_id(tid)
            if (
                tid != current_topic
                and family_id >= 0
                and kept_family_counts.get(family_id, 0) >= family_limit
            ):
                self.self_excite_skip_events += 1
                continue

            visited.add(tid)
            expanded += 1
            candidate_scores[tid] = total
            if tid != current_topic and family_id >= 0:
                kept_family_counts[family_id] = kept_family_counts.get(family_id, 0) + 1
            if family_id >= 0:
                visited_families.add(family_id)

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
                if not self.topic_shards[dst].members:
                    continue
                if dst not in semantic_scores:
                    semantic_scores[dst] = self._semantic_topic_score(int(dst), query_vec, current_topic)
                    ops += 1
                next_bonus = bridge_bonus * 0.60 + float(self.p.topic_graph_bridge_weight) * edge_score
                resident_bonus = float(self.p.topic_graph_resident_bonus) if self.topic_shards[dst].loaded else 0.0
                risk_penalty, escape_bonus = self._self_excitation_adjustment(
                    src_tid=int(tid),
                    dst_tid=int(dst),
                    current_topic=current_topic,
                    visited_families=visited_families,
                )
                total_dst = float(semantic_scores[dst]) + next_bonus + resident_bonus + escape_bonus - risk_penalty
                if risk_penalty > 0.0:
                    self.self_excite_penalty_total += float(risk_penalty)
                    self.self_excite_event_count += 1
                if escape_bonus > 0.0:
                    self.self_excite_escape_bonus_total += float(escape_bonus)
                    self.self_excite_escape_event_count += 1
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
        ranked = sorted(candidate_scores.items(), key=lambda it: it[1], reverse=True)
        available: Set[int] = set()
        if not bool(self.p.smart_preload_enabled):
            for tid, _score in ranked:
                shard = self.topic_shards[tid]
                if not shard.loaded:
                    continue
                shard.last_loaded_turn = turn
                available.add(int(tid))
            return available

        self.preload_attempts += 1
        load_budget = max(1, int(self.p.topic_graph_load_budget))
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
        current_topic: Optional[int],
        current_turn: int,
        semantic_scores: Dict[int, float],
        query_facet_ids: Optional[np.ndarray] = None,
        query_facet_weights: Optional[np.ndarray] = None,
        anchor_evidence: Optional[Dict[int, List[int]]] = None,
        anchor_debug: Optional[Dict[str, object]] = None,
    ) -> Tuple[List[int], Dict[str, object], int]:
        if not available_topics:
            return [], {"evidence_topics": [], "evidence_memories": []}, 0

        per_topic_evidence = max(1, int(self.p.topic_graph_per_topic_evidence))
        ranked_topics: List[Tuple[int, float]] = []
        evidence_topics: List[int] = []
        evidence_memories: List[int] = []
        evidence_by_topic: Dict[int, List[int]] = {}
        local_signals: Dict[int, Dict[str, List[int]]] = {}
        topic_rows: List[Dict[str, object]] = []
        ops = 0
        query_map = self._build_query_facet_map(query_facet_ids, query_facet_weights)
        momentum_probe, momentum_ops = self._one_turn_momentum_probe(
            query_vec=query_vec,
            current_topic=current_topic,
            current_turn=current_turn,
            semantic_scores=semantic_scores,
        )
        ops += int(momentum_ops)
        anchor_evidence = anchor_evidence or {}
        anchor_debug = anchor_debug or {}
        momentum_scores = {int(mem_idx): float(score) for mem_idx, score in momentum_probe}
        momentum_memories = [int(mem_idx) for mem_idx, _score in momentum_probe]

        for tid in sorted(available_topics):
            members = self.topic_shards[tid].members
            if not members:
                continue
            query_result = self.topic_local_index.collect_candidates(
                self,
                tid,
                query_vec,
                per_topic_evidence,
                query_facet_ids=query_facet_ids,
                query_facet_weights=query_facet_weights,
            )
            topic_ops = int(query_result.ops)
            idx = query_result.indices
            local_score_map: Dict[int, float] = {}
            local_group_map: Dict[int, int] = {}
            if len(query_result.memory_scores) == int(idx.size):
                for pos, mem_idx in enumerate(idx.tolist()):
                    local_score_map[int(mem_idx)] = float(query_result.memory_scores[pos])
            if len(query_result.memory_bundle_ids) == int(idx.size):
                for pos, mem_idx in enumerate(idx.tolist()):
                    local_group_map[int(mem_idx)] = int(query_result.memory_bundle_ids[pos])
            if tid in anchor_evidence:
                extra_ids = [int(mem_id) for mem_id in anchor_evidence.get(int(tid), [])]
                if extra_ids:
                    idx_set = set(int(x) for x in idx.tolist())
                    merged = [mem_id for mem_id in extra_ids if mem_id not in idx_set]
                    if merged:
                        idx = np.concatenate(
                            [np.asarray(merged, dtype=np.int32), idx.astype(np.int32, copy=False)]
                        )
            if current_topic is not None and tid == int(current_topic) and momentum_memories:
                idx_set = set(int(x) for x in idx.tolist())
                extra = [mem_id for mem_id in momentum_memories if mem_id not in idx_set]
                if extra:
                    idx = np.concatenate(
                        [np.asarray(extra, dtype=np.int32), idx.astype(np.int32, copy=False)]
                    )
            if idx.size <= 0:
                idx = np.asarray(members, dtype=np.int32)
            ops += topic_ops
            sims = self.vec_pool[idx] @ query_vec
            ops += int(idx.size)
            if sims.size <= 0:
                continue
            k = min(per_topic_evidence, int(sims.size))
            if local_score_map and query_map:
                semantic_map = {int(idx[pos]): float(sims[pos]) for pos in range(int(idx.size))}
                ranked_memories = self._select_memories_for_context(
                    memory_ids=idx.tolist(),
                    query_vec=query_vec,
                    query_map=query_map,
                    budget=k,
                    semantic_scores=semantic_map,
                    local_scores=local_score_map,
                    group_ids=local_group_map,
                )
                best_memories = [int(mem_id) for mem_id, _score in ranked_memories]
                best_sims = np.asarray(
                    [float(semantic_map.get(int(mem_id), 0.0)) for mem_id in best_memories],
                    dtype=np.float32,
                )
                ops += int(idx.size)
            else:
                if k < sims.size:
                    keep = np.argpartition(sims, -k)[-k:]
                    best = keep[np.argsort(sims[keep])[::-1]]
                    ops += int(sims.size)
                else:
                    best = np.argsort(sims)[::-1]
                    ops += int(sims.size)
                best_sims = sims[best]
                best_memories = [int(idx[i]) for i in best.tolist()]
            evidence_score = max(float(query_result.score), float(np.max(best_sims))) + 0.15 * float(np.mean(best_sims))
            total_score = float(candidate_scores.get(tid, 0.0)) + evidence_score
            ranked_topics.append((int(tid), total_score))
            evidence_topics.append(int(tid))
            evidence_memories.extend(best_memories)
            evidence_by_topic[int(tid)] = best_memories
            local_signals[int(tid)] = {
                "bundle_ids": [int(bid) for bid in query_result.bundle_ids],
                "category_ids": [int(cid) for cid in query_result.category_ids],
            }
            topic_rows.append(
                {
                    "tid": int(tid),
                    "total_score": float(total_score),
                    "route_score": float(candidate_scores.get(tid, 0.0)),
                    "peak_sim": float(np.max(best_sims)),
                    "mean_sim": float(np.mean(best_sims)),
                    "topic_cover": self._topic_facet_cover(best_memories),
                    "family_id": int(self._topic_family_id(int(tid))),
                }
            )
            if self.topic_shards[tid].loaded:
                self.resident_hit_events += 1

        ranked_topics = self._select_topics_for_context(topic_rows, query_map)
        selected_topics: List[int] = [int(tid) for tid, _score in ranked_topics]
        return selected_topics, {
            "evidence_topics": evidence_topics,
            "evidence_memories": evidence_memories,
            "evidence_by_topic": evidence_by_topic,
            "ranked_topics": ranked_topics,
            "local_signals": local_signals,
            "topic_families": {int(tid): int(self._topic_family_id(tid)) for tid, _score in ranked_topics},
            "momentum_probe_memories": momentum_memories,
            "momentum_probe_scores": {int(mem_idx): float(score) for mem_idx, score in momentum_scores.items()},
            "anchor_seed_ids": [int(aid) for aid in anchor_debug.get("anchor_seed_ids", [])],
            "anchor_visit_ids": [int(aid) for aid in anchor_debug.get("anchor_visit_ids", [])],
            "anchor_memory_ids": [int(mem_id) for mem_id in anchor_debug.get("anchor_memory_ids", [])],
        }, ops

    def retrieve(
        self,
        query_vec: np.ndarray,
        current_topic: Optional[int],
        current_turn: int,
        query_facet_ids: Optional[np.ndarray] = None,
        query_facet_weights: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], int, Dict[str, object]]:
        semantic_scores, ops0 = self._semantic_topic_scores(query_vec, current_topic)
        if not semantic_scores:
            return [], ops0, {"candidate_topics": [], "bridge_topics": []}

        candidate_scores, via_bridge, ops1 = self._expand_topic_candidates(query_vec, semantic_scores, current_topic)
        available_topics = self._preload_topics(candidate_scores, current_turn)
        selected_topics, debug, ops2 = self._retrieve_topic_evidence(
            query_vec,
            candidate_scores,
            available_topics,
            current_topic=current_topic,
            current_turn=current_turn,
            semantic_scores=semantic_scores,
            query_facet_ids=query_facet_ids,
            query_facet_weights=query_facet_weights,
            anchor_evidence=None,
            anchor_debug=None,
        )
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

    def _memory_facet_map(self, mem_idx: int) -> Dict[int, float]:
        if mem_idx < 0 or mem_idx >= self.mem_count:
            return {}
        out: Dict[int, float] = {}
        facet_ids = self.mem_facet_ids[mem_idx]
        facet_weights = self.mem_facet_weights[mem_idx]
        for pos in range(self.max_facet_slots):
            facet_id = int(facet_ids[pos])
            if facet_id < 0:
                continue
            weight = float(facet_weights[pos])
            if weight <= 0.0:
                continue
            out[facet_id] = weight
        return out

    def _build_query_facet_map(
        self,
        query_facet_ids: Optional[np.ndarray],
        query_facet_weights: Optional[np.ndarray],
    ) -> Dict[int, float]:
        if query_facet_ids is None or query_facet_weights is None:
            return {}
        out: Dict[int, float] = {}
        for pos in range(min(len(query_facet_ids), len(query_facet_weights))):
            facet_id = int(query_facet_ids[pos])
            if facet_id < 0:
                continue
            weight = float(query_facet_weights[pos])
            if weight <= 0.0:
                continue
            out[facet_id] = out.get(facet_id, 0.0) + weight
        return out

    def _topic_facet_cover(
        self,
        mem_ids: Sequence[int],
    ) -> Dict[int, float]:
        cover: Dict[int, float] = {}
        for mem_idx in mem_ids:
            mem_map = self._memory_facet_map(int(mem_idx))
            if not mem_map:
                continue
            for facet_id, weight in mem_map.items():
                if float(weight) > float(cover.get(int(facet_id), 0.0)):
                    cover[int(facet_id)] = float(weight)
        return cover

    def _select_topics_for_context(
        self,
        topic_rows: Sequence[Dict[str, object]],
        query_map: Dict[int, float],
    ) -> List[Tuple[int, float]]:
        if not topic_rows:
            return []

        max_topics = max(1, int(self.p.topic_graph_max_return_topics))
        query_total_weight = max(1e-6, float(sum(query_map.values())))
        selected: List[Tuple[int, float]] = []
        selected_set: Set[int] = set()
        best_facet_cover = {int(facet_id): 0.0 for facet_id in query_map}
        family_counts: Dict[int, int] = {}
        family_limit = max(1, int(self.p.topic_graph_family_member_limit))

        while len(selected) < max_topics:
            best_tid = -1
            best_score = -1e9
            best_cover: Optional[Dict[int, float]] = None
            best_family = -1

            for row in topic_rows:
                tid = int(row.get("tid", -1))
                if tid < 0 or tid in selected_set:
                    continue

                family_id = int(row.get("family_id", -1))
                family_repeat = family_counts.get(family_id, 0) if family_id >= 0 else 0
                if family_id >= 0 and family_repeat >= family_limit and len(selected) > 0:
                    self.self_excite_skip_events += 1
                    continue

                topic_cover = row.get("topic_cover", {})
                if not isinstance(topic_cover, dict):
                    topic_cover = {}

                coverage_gain = 0.0
                if query_map and topic_cover:
                    for facet_id, query_weight in query_map.items():
                        current = float(best_facet_cover.get(int(facet_id), 0.0))
                        updated = float(topic_cover.get(int(facet_id), 0.0))
                        if updated > current:
                            coverage_gain += float(query_weight) * (updated - current)
                    coverage_gain /= query_total_weight

                peak_sim = max(0.0, float(row.get("peak_sim", 0.0)))
                mean_sim = max(0.0, float(row.get("mean_sim", 0.0)))
                route_score = max(0.0, min(1.0, float(row.get("route_score", 0.0))))
                semantic_support = min(
                    1.0,
                    0.55 * peak_sim + 0.20 * mean_sim + 0.25 * route_score,
                )

                novelty_bonus = 0.10 if family_id >= 0 and family_repeat <= 0 else 0.0
                repeat_penalty = 0.06 * (family_repeat / float(family_repeat + 1)) if family_repeat > 0 else 0.0
                context_score = (
                    float(coverage_gain)
                    + 0.55 * float(semantic_support)
                    + float(novelty_bonus)
                    - float(repeat_penalty)
                )
                if context_score > best_score:
                    best_tid = tid
                    best_score = float(context_score)
                    best_cover = {int(fid): float(val) for fid, val in topic_cover.items()}
                    best_family = family_id

            if best_tid < 0:
                break

            selected.append((int(best_tid), float(best_score)))
            selected_set.add(int(best_tid))
            if best_cover:
                for facet_id, weight in best_cover.items():
                    best_facet_cover[int(facet_id)] = max(
                        float(best_facet_cover.get(int(facet_id), 0.0)),
                        float(weight),
                    )
            if best_family >= 0:
                family_counts[int(best_family)] = family_counts.get(int(best_family), 0) + 1

        if selected:
            return selected

        fallback = sorted(
            (
                (int(row.get("tid", -1)), float(row.get("total_score", 0.0)))
                for row in topic_rows
                if int(row.get("tid", -1)) >= 0
            ),
            key=lambda it: it[1],
            reverse=True,
        )
        return fallback[:max_topics]

    def _select_memories_for_context(
        self,
        memory_ids: Sequence[int],
        query_vec: np.ndarray,
        query_map: Dict[int, float],
        budget: int,
        semantic_scores: Optional[Dict[int, float]] = None,
        local_scores: Optional[Dict[int, float]] = None,
        group_ids: Optional[Dict[int, int]] = None,
    ) -> List[Tuple[int, float]]:
        if not memory_ids:
            return []

        semantic_scores = semantic_scores or {}
        local_scores = local_scores or {}
        group_ids = group_ids or {}
        unique_ids: List[int] = []
        seen_ids: Set[int] = set()
        for mem_idx in memory_ids:
            mem_id = int(mem_idx)
            if mem_id < 0 or mem_id >= self.mem_count or mem_id in seen_ids:
                continue
            seen_ids.add(mem_id)
            unique_ids.append(mem_id)
        if not unique_ids:
            return []

        local_max = max((max(0.0, float(score)) for score in local_scores.values()), default=0.0)
        query_total_weight = max(1e-6, float(sum(query_map.values())))
        sim_threshold = min(
            0.999,
            max(-1.0, float(self.p.context_similarity_saturation_threshold)),
        )

        cache: Dict[int, Dict[str, object]] = {}
        for mem_id in unique_ids:
            semantic = float(semantic_scores.get(mem_id, float(self.vec_pool[mem_id] @ query_vec)))
            local = float(local_scores.get(mem_id, 0.0))
            local_norm = (local / local_max) if local_max > 1e-6 else 0.0
            cache[mem_id] = {
                "semantic": max(0.0, semantic),
                "local": max(0.0, local_norm),
                "map": self._memory_facet_map(mem_id),
                "group_id": int(group_ids.get(mem_id, -1)),
                "vec": self.vec_pool[mem_id],
            }

        remaining: Set[int] = set(unique_ids)
        selected: List[Tuple[int, float]] = []
        selected_ids: List[int] = []
        best_facet_cover = {int(facet_id): 0.0 for facet_id in query_map}
        group_counts: Dict[int, int] = {}

        while remaining and len(selected) < max(1, int(budget)):
            best_mem = -1
            best_score = -1e9
            for mem_id in sorted(remaining):
                info = cache[mem_id]
                mem_map = info["map"]

                coverage_gain = 0.0
                if query_map and mem_map:
                    for facet_id, query_weight in query_map.items():
                        current = float(best_facet_cover.get(int(facet_id), 0.0))
                        updated = float(mem_map.get(int(facet_id), 0.0))
                        if updated > current:
                            coverage_gain += float(query_weight) * (updated - current)
                    coverage_gain /= query_total_weight

                max_pair_excess = 0.0
                if selected_ids:
                    for selected_id in selected_ids:
                        sim = float(info["vec"] @ self.vec_pool[int(selected_id)])
                        if sim > sim_threshold:
                            max_pair_excess = max(
                                max_pair_excess,
                                (sim - sim_threshold) / max(1e-6, 1.0 - sim_threshold),
                            )

                group_id = int(info["group_id"])
                group_repeat = group_counts.get(group_id, 0) if group_id >= 0 else 0
                group_bonus = 0.08 if group_id >= 0 and group_repeat <= 0 else 0.0
                group_penalty = 0.05 * (group_repeat / float(group_repeat + 1)) if group_repeat > 0 else 0.0

                semantic = float(info["semantic"])
                local = float(info["local"])
                if query_map:
                    context_score = (
                        float(coverage_gain)
                        + 0.50 * semantic
                        + 0.22 * local
                        + float(group_bonus)
                        - 0.14 * float(max_pair_excess)
                        - float(group_penalty)
                    )
                else:
                    context_score = (
                        0.72 * semantic
                        + 0.28 * local
                        + float(group_bonus)
                        - 0.14 * float(max_pair_excess)
                        - float(group_penalty)
                    )

                if context_score > best_score:
                    best_mem = int(mem_id)
                    best_score = float(context_score)

            if best_mem < 0:
                break

            selected.append((int(best_mem), float(best_score)))
            selected_ids.append(int(best_mem))
            remaining.remove(int(best_mem))
            info = cache[int(best_mem)]
            mem_map = info["map"]
            if isinstance(mem_map, dict):
                for facet_id in query_map:
                    best_facet_cover[int(facet_id)] = max(
                        float(best_facet_cover.get(int(facet_id), 0.0)),
                        float(mem_map.get(int(facet_id), 0.0)),
                    )
            group_id = int(info["group_id"])
            if group_id >= 0:
                group_counts[group_id] = group_counts.get(group_id, 0) + 1

        return selected

    def _dominant_memory_topic(self, mem_idx: int) -> Optional[int]:
        if mem_idx < 0 or mem_idx >= len(self.mem_origin_topic_counts):
            return None
        counts = self.mem_origin_topic_counts[mem_idx]
        if not counts:
            return None
        best_topic = None
        best_n = -1
        for tid, n in counts.items():
            if int(n) > best_n:
                best_topic = int(tid)
                best_n = int(n)
        return best_topic

    def _merge_memory_facets(
        self,
        mem_idx: int,
        facet_ids: Optional[np.ndarray],
        facet_weights: Optional[np.ndarray],
    ) -> None:
        if facet_ids is None or facet_weights is None or mem_idx < 0 or mem_idx >= self.mem_count:
            return
        merged = self._memory_facet_map(mem_idx)
        for pos in range(min(len(facet_ids), len(facet_weights))):
            facet_id = int(facet_ids[pos])
            if facet_id < 0:
                continue
            weight = float(facet_weights[pos])
            if weight <= 0.0:
                continue
            merged[facet_id] = merged.get(facet_id, 0.0) + weight
        if not merged:
            return
        total = max(1e-6, float(sum(merged.values())))
        ranked = sorted(
            ((int(fid), float(weight) / total) for fid, weight in merged.items() if weight > 0.0),
            key=lambda it: it[1],
            reverse=True,
        )[: self.max_facet_slots]
        self.mem_facet_ids[mem_idx].fill(-1)
        self.mem_facet_weights[mem_idx].fill(0.0)
        for pos, (facet_id, weight) in enumerate(ranked):
            self.mem_facet_ids[mem_idx, pos] = int(facet_id)
            self.mem_facet_weights[mem_idx, pos] = float(weight)

    def _bind_memory_topic(
        self,
        mem_idx: int,
        topic_id: int,
        vec: np.ndarray,
        turn: int,
        amount: int = 1,
    ) -> int:
        if mem_idx < 0 or topic_id < 0 or topic_id >= self.topic_count:
            return 0
        while len(self.mem_origin_topic_counts) <= mem_idx:
            self.mem_origin_topic_counts.append({})
        counts = self.mem_origin_topic_counts[mem_idx]
        counts[int(topic_id)] = counts.get(int(topic_id), 0) + max(1, int(amount))
        dominant = self._dominant_memory_topic(mem_idx)
        if dominant is not None:
            self.mem_topic[mem_idx] = int(dominant)

        if mem_idx in self.topic_member_sets[topic_id]:
            return 0

        self.topic_member_sets[topic_id].add(int(mem_idx))
        shard = self.topic_shards[topic_id]
        shard.members.append(int(mem_idx))
        if shard.centroid is None:
            shard.centroid = vec.astype(np.float32, copy=True)
        else:
            n = len(shard.members)
            shard.centroid = unit((((n - 1) * shard.centroid) + vec) / float(n)).astype(np.float32)
        return int(self.topic_local_index.add_memory(self, topic_id, vec, mem_idx, turn))

    def _topic_merge_candidate_topics(
        self,
        vec: np.ndarray,
        topic_id: int,
    ) -> Tuple[List[int], int]:
        known = self._known_topic_ids()
        if known.size <= 0:
            return [], 0
        scores, ops = self._semantic_topic_scores(vec, current_topic=topic_id)
        ranked = sorted(scores.items(), key=lambda it: it[1], reverse=True)
        limit = max(
            1,
            min(
                self.topic_count,
                max(int(self.p.topic_graph_seed_topics) + 2, int(self.p.topic_graph_load_budget)),
            ),
        )
        out: List[int] = []
        seen: Set[int] = set()
        if 0 <= topic_id < self.topic_count and self.topic_shards[topic_id].members:
            out.append(int(topic_id))
            seen.add(int(topic_id))
        for tid, _score in ranked:
            if tid in seen or not self.topic_shards[int(tid)].members:
                continue
            out.append(int(tid))
            seen.add(int(tid))
            if len(out) >= limit:
                break
        return out, ops

    def _find_merge_target(
        self,
        vec: np.ndarray,
        topic_id: int,
    ) -> Tuple[Optional[int], float, int]:
        topic_ids, ops = self._topic_merge_candidate_topics(vec, topic_id)
        if not topic_ids:
            return None, -1.0, ops

        candidate_memories: List[int] = []
        seen_memories: Set[int] = set()
        for tid in topic_ids:
            for mem_idx in self.topic_shards[int(tid)].members:
                if mem_idx in seen_memories or mem_idx < 0 or mem_idx >= self.mem_count:
                    continue
                seen_memories.add(int(mem_idx))
                candidate_memories.append(int(mem_idx))
        if not candidate_memories:
            return None, -1.0, ops

        idx = np.asarray(candidate_memories, dtype=np.int32)
        sims = self.vec_pool[idx] @ vec
        ops += int(idx.size)
        if sims.size <= 0:
            return None, -1.0, ops
        best_pos = int(np.argmax(sims))
        return int(idx[best_pos]), float(sims[best_pos]), ops

    def _context_candidate_memories(
        self,
        selected_topics: Sequence[int],
        debug: Dict[str, object],
    ) -> List[int]:
        evidence_by_topic = debug.get("evidence_by_topic", {})
        if not isinstance(evidence_by_topic, dict):
            evidence_by_topic = {}

        selected_set = set(int(tid) for tid in selected_topics)
        topic_order: List[int] = []
        seen_topics: Set[int] = set()
        ranked_topics = debug.get("ranked_topics", [])
        if isinstance(ranked_topics, list):
            for item in ranked_topics:
                if not isinstance(item, (list, tuple)) or not item:
                    continue
                tid = int(item[0])
                if tid in selected_set and tid not in seen_topics:
                    topic_order.append(tid)
                    seen_topics.add(tid)
        for tid in selected_topics:
            if int(tid) not in seen_topics:
                topic_order.append(int(tid))
                seen_topics.add(int(tid))

        candidates: List[int] = []
        seen_memories: Set[int] = set()
        for tid in topic_order:
            mem_ids = evidence_by_topic.get(int(tid), [])
            if not isinstance(mem_ids, list):
                continue
            for mem_idx in mem_ids:
                mem_id = int(mem_idx)
                if mem_id < 0 or mem_id >= self.mem_count or mem_id in seen_memories:
                    continue
                seen_memories.add(mem_id)
                candidates.append(mem_id)

        if candidates:
            return candidates

        fallback = debug.get("evidence_memories", [])
        if not isinstance(fallback, list):
            return []
        for mem_idx in fallback:
            mem_id = int(mem_idx)
            if mem_id < 0 or mem_id >= self.mem_count or mem_id in seen_memories:
                continue
            seen_memories.add(mem_id)
            candidates.append(mem_id)
        return candidates

    def _context_saturation_score(
        self,
        context_memories: Sequence[int],
    ) -> float:
        if not context_memories:
            return 0.0

        sim_threshold = min(
            0.999,
            max(-1.0, float(self.p.context_similarity_saturation_threshold)),
        )
        pair_excess = 0.0
        pair_count = 0
        if len(context_memories) >= 2:
            for i in range(len(context_memories)):
                vec_a = self.vec_pool[int(context_memories[i])]
                for j in range(i + 1, len(context_memories)):
                    sim = float(vec_a @ self.vec_pool[int(context_memories[j])])
                    if sim > sim_threshold:
                        pair_excess += (sim - sim_threshold) / max(1e-6, 1.0 - sim_threshold)
                    pair_count += 1
        pair_excess = pair_excess / float(pair_count) if pair_count > 0 else 0.0

        dominant_topics: List[int] = []
        family_ids: List[int] = []
        for mem_idx in context_memories:
            dominant = self._dominant_memory_topic(int(mem_idx))
            if dominant is None:
                continue
            dominant_topics.append(int(dominant))
            family_id = self._topic_family_id(int(dominant))
            if family_id >= 0:
                family_ids.append(int(family_id))

        topic_concentration = 0.0
        if dominant_topics:
            topic_concentration = 1.0 - (len(set(dominant_topics)) / float(len(dominant_topics)))

        family_concentration = 0.0
        if family_ids:
            family_concentration = 1.0 - (len(set(family_ids)) / float(len(family_ids)))

        return float(
            min(
                1.0,
                max(
                    0.0,
                    0.55 * pair_excess + 0.30 * topic_concentration + 0.15 * family_concentration,
                ),
            )
        )

    def _assemble_context_memories(
        self,
        candidate_memories: Sequence[int],
        query_vec: np.ndarray,
        query_map: Dict[int, float],
        context_budget: int,
    ) -> Tuple[List[int], float, float]:
        budget = max(1, int(context_budget))
        if not candidate_memories:
            return [], 0.0, 0.0

        query_total_weight = max(1e-6, float(sum(query_map.values())))
        association_weight = max(0.0, float(self.p.context_association_weight))
        saturation_weight = max(0.0, float(self.p.context_saturation_weight))
        irrelevance_weight = max(0.0, float(self.p.context_irrelevance_weight))
        sim_threshold = min(
            0.999,
            max(-1.0, float(self.p.context_similarity_saturation_threshold)),
        )
        min_marginal = float(self.p.context_min_marginal_gain)

        cache: Dict[int, Dict[str, object]] = {}
        for mem_idx in candidate_memories:
            mem_map = self._memory_facet_map(int(mem_idx))
            facet_relevance = 0.0
            if mem_map and query_map:
                for facet_id, query_weight in query_map.items():
                    facet_relevance += float(query_weight) * float(mem_map.get(int(facet_id), 0.0))
                facet_relevance /= query_total_weight
            semantic_relevance = max(0.0, float(self.vec_pool[int(mem_idx)] @ query_vec))
            base_relevance = min(1.0, 0.62 * facet_relevance + 0.38 * semantic_relevance)
            irrelevance = max(0.0, 1.0 - min(1.0, facet_relevance + 0.30 * semantic_relevance))
            dominant_topic = self._dominant_memory_topic(int(mem_idx))
            family_id = self._topic_family_id(int(dominant_topic)) if dominant_topic is not None else -1
            cache[int(mem_idx)] = {
                "map": mem_map,
                "vec": self.vec_pool[int(mem_idx)],
                "base_relevance": base_relevance,
                "irrelevance": irrelevance,
                "dominant_topic": dominant_topic,
                "family_id": family_id,
            }

        remaining: Set[int] = set(int(mem_idx) for mem_idx in candidate_memories if int(mem_idx) in cache)
        selected: List[int] = []
        best_facet_cover = {int(facet_id): 0.0 for facet_id in query_map}
        topic_counts: Dict[int, int] = {}
        family_counts: Dict[int, int] = {}
        association_total = 0.0

        while remaining and len(selected) < budget:
            best_mem = -1
            best_score = -1e9
            best_association = 0.0
            for mem_idx in sorted(remaining):
                info = cache[int(mem_idx)]
                mem_map = info["map"]

                coverage_gain = 0.0
                if query_map and mem_map:
                    for facet_id, query_weight in query_map.items():
                        current = float(best_facet_cover.get(int(facet_id), 0.0))
                        updated = float(mem_map.get(int(facet_id), 0.0))
                        if updated > current:
                            coverage_gain += float(query_weight) * (updated - current)
                    coverage_gain /= query_total_weight

                pair_excess: List[float] = []
                for selected_idx in selected:
                    selected_info = cache[int(selected_idx)]
                    sim = float(info["vec"] @ selected_info["vec"])
                    if sim > sim_threshold:
                        pair_excess.append((sim - sim_threshold) / max(1e-6, 1.0 - sim_threshold))
                max_pair_excess = max(pair_excess) if pair_excess else 0.0
                mean_pair_excess = (sum(pair_excess) / len(pair_excess)) if pair_excess else 0.0

                topic_saturation = 0.0
                dominant_topic = info["dominant_topic"]
                if dominant_topic is not None:
                    count = topic_counts.get(int(dominant_topic), 0)
                    if count > 0:
                        topic_saturation = count / float(count + 1)

                family_saturation = 0.0
                family_id = int(info["family_id"])
                if family_id >= 0:
                    count = family_counts.get(family_id, 0)
                    if count > 0:
                        family_saturation = count / float(count + 1)

                saturation = min(
                    1.0,
                    0.60 * max_pair_excess
                    + 0.20 * mean_pair_excess
                    + 0.20 * max(topic_saturation, family_saturation),
                )

                association = 0.0
                if dominant_topic is not None and topic_counts.get(int(dominant_topic), 0) <= 0:
                    association += 0.55
                if family_id >= 0 and family_counts.get(family_id, 0) <= 0:
                    association += 0.45
                association = min(1.0, association)

                marginal = (
                    float(coverage_gain)
                    + 0.18 * float(info["base_relevance"])
                    + association_weight * association
                    - saturation_weight * float(saturation)
                    - 0.50 * irrelevance_weight * float(info["irrelevance"])
                )
                if marginal > best_score:
                    best_score = marginal
                    best_mem = int(mem_idx)
                    best_association = association

            if best_mem < 0:
                break
            if selected and best_score < min_marginal:
                break

            selected.append(int(best_mem))
            remaining.remove(int(best_mem))
            info = cache[int(best_mem)]
            mem_map = info["map"]
            if isinstance(mem_map, dict):
                for facet_id in query_map:
                    best_facet_cover[int(facet_id)] = max(
                        float(best_facet_cover.get(int(facet_id), 0.0)),
                        float(mem_map.get(int(facet_id), 0.0)),
                    )
            dominant_topic = info["dominant_topic"]
            if dominant_topic is not None:
                topic_counts[int(dominant_topic)] = topic_counts.get(int(dominant_topic), 0) + 1
            family_id = int(info["family_id"])
            if family_id >= 0:
                family_counts[family_id] = family_counts.get(family_id, 0) + 1
            association_total += float(best_association)

        if not selected and cache:
            fallback = max(cache.items(), key=lambda it: float(it[1]["base_relevance"]))[0]
            selected = [int(fallback)]
            association_total = 1.0

        fill_ratio = float(len(selected)) / float(max(1, budget))
        association_score = float(association_total / max(1, len(selected)))
        return selected, association_score, fill_ratio

    def _eval_context_quality(
        self,
        selected_topics: Sequence[int],
        debug: Dict[str, object],
        query_vec: np.ndarray,
        query_facet_ids: np.ndarray,
        query_facet_weights: np.ndarray,
    ) -> float:
        query_map = self._build_query_facet_map(query_facet_ids, query_facet_weights)

        context_budget = max(
            1,
            int(self.p.topic_graph_max_return_topics) * int(self.p.topic_graph_per_topic_evidence),
        )
        candidate_memories = self._context_candidate_memories(selected_topics, debug)
        context_memories, association, fill_ratio = self._assemble_context_memories(
            candidate_memories=candidate_memories,
            query_vec=query_vec,
            query_map=query_map,
            context_budget=context_budget,
        )
        debug["assembled_context_memories"] = [int(mem_idx) for mem_idx in context_memories]
        debug["assembled_context_fill_ratio"] = float(fill_ratio)

        coverage = 0.0
        total_query_weight = max(1e-6, float(sum(query_map.values())))
        cached_maps = [self._memory_facet_map(mem_idx) for mem_idx in context_memories]
        if query_map and context_memories:
            for facet_id, query_weight in query_map.items():
                best = 0.0
                for mem_map in cached_maps:
                    if not mem_map:
                        continue
                    best = max(best, float(mem_map.get(int(facet_id), 0.0)))
                coverage += query_weight * best
            coverage /= total_query_weight

        redundancy = 0.0
        if len(context_memories) >= 2:
            pair_sum = 0.0
            pair_count = 0
            for i in range(len(cached_maps)):
                a = cached_maps[i]
                if not a:
                    continue
                for j in range(i + 1, len(cached_maps)):
                    b = cached_maps[j]
                    if not b:
                        continue
                    overlap = 0.0
                    for facet_id, weight_a in a.items():
                        overlap += min(float(weight_a), float(b.get(facet_id, 0.0)))
                    pair_sum += overlap
                    pair_count += 1
            if pair_count > 0:
                redundancy = pair_sum / float(pair_count)

        irrelevance = 0.0
        if context_memories:
            irr_sum = 0.0
            query_ids = set(int(fid) for fid in query_map)
            for mem_map in cached_maps:
                if not mem_map:
                    continue
                relevance = sum(float(weight) for facet_id, weight in mem_map.items() if facet_id in query_ids)
                irr_sum += max(0.0, 1.0 - min(1.0, relevance))
            irrelevance = irr_sum / float(len(context_memories))

        saturation = self._context_saturation_score(context_memories)
        quality = max(
            0.0,
            min(
                1.0,
                float(coverage)
                + float(self.p.context_association_weight) * float(association)
                - float(self.p.context_redundancy_weight) * float(redundancy)
                - float(self.p.context_saturation_weight) * float(saturation)
                - float(self.p.context_irrelevance_weight) * float(irrelevance),
            ),
        )

        self.context_eval_count += 1
        self.context_quality_sum += quality
        self.context_coverage_sum += float(coverage)
        self.context_association_sum += float(association)
        self.context_saturation_sum += float(saturation)
        self.context_redundancy_sum += float(redundancy)
        self.context_irrelevance_sum += float(irrelevance)
        self.context_fill_ratio_sum += float(fill_ratio)
        return quality

    def _apply_query_feedback(
        self,
        current_topic: int,
        target_topic: int,
        selected_topics: List[int],
        bridge_topics: Sequence[int],
        local_signals: Optional[Dict[int, Dict[str, List[int]]]],
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
        if self._topic_local_index_mode() != "deep_artmap":
            return
        if not local_signals:
            return

        recall_lr = float(self.p.topic_graph_deep_artmap_bundle_recall_lr)
        adopt_lr = float(self.p.topic_graph_deep_artmap_bundle_adopt_lr)
        edge_lr = float(self.p.topic_graph_deep_artmap_bundle_edge_feedback)

        if target_topic in selected_topics:
            target_signal = local_signals.get(int(target_topic), {})
            bundle_ids = [int(bid) for bid in target_signal.get("bundle_ids", [])]
            for bid in bundle_ids:
                self._deep_artmap_reinforce_bundle(int(target_topic), bid, "recall", recall_lr)
            for i in range(len(bundle_ids)):
                for j in range(i + 1, len(bundle_ids)):
                    state = self.topic_shards[int(target_topic)].deep_artmap
                    if state is None:
                        break
                    self._deep_artmap_link_bundles(state, int(bundle_ids[i]), int(bundle_ids[j]))
                    a = state.bundles[int(bundle_ids[i])]
                    b = state.bundles[int(bundle_ids[j])]
                    a.neighbors[int(bundle_ids[j])] = float(a.neighbors.get(int(bundle_ids[j]), 0.0)) + edge_lr
                    b.neighbors[int(bundle_ids[i])] = float(b.neighbors.get(int(bundle_ids[i]), 0.0)) + edge_lr

        if selected_topics and selected_topics[0] == target_topic:
            target_signal = local_signals.get(int(target_topic), {})
            bundle_ids = [int(bid) for bid in target_signal.get("bundle_ids", [])]
            if bundle_ids:
                self._deep_artmap_reinforce_bundle(int(target_topic), int(bundle_ids[0]), "adopt", adopt_lr)
                for bid in bundle_ids[1:]:
                    self._deep_artmap_reinforce_bundle(int(target_topic), int(bid), "adopt", adopt_lr * 0.35)

    def _summary(self) -> Dict[str, float]:
        loaded_topics = sum(1 for shard in self.topic_shards if shard.loaded)
        active_topics = sum(1 for shard in self.topic_shards if shard.members)
        bridge_edges = sum(len(edges) for edges in self.topic_bridges)
        eval_n = max(1, self.eval_count)
        context_n = max(1, self.context_eval_count)
        merge_rate = self.merge_count / max(1, self.p.turns)
        avg_query_ops = self.sim_ops_query_total / max(1, self.query_turns)
        avg_add_ops = self.sim_ops_add_total / max(1, self.p.turns)
        avg_turn_ops = float(np.mean(self.turn_sim_ops)) if self.turn_sim_ops else 0.0
        p95_turn_ops = float(np.quantile(self.turn_sim_ops, 0.95)) if self.turn_sim_ops else 0.0
        target_hit = self.target_hit_sum / eval_n
        target_precision = self.target_precision_sum / eval_n
        target_mrr = self.target_mrr_sum / eval_n
        local_summary = self.topic_local_index.summary(self, active_topics)
        origin_topic_total = sum(len(counts) for counts in self.mem_origin_topic_counts[: self.mem_count])
        multi_topic_count = sum(1 for counts in self.mem_origin_topic_counts[: self.mem_count] if len(counts) > 1)
        family_count = len(self.topic_family_members)
        family_max_size = max((len(members) for members in self.topic_family_members), default=0)
        self_excite_avg_penalty = self.self_excite_penalty_total / max(1, self.self_excite_event_count)
        self_excite_avg_escape = self.self_excite_escape_bonus_total / max(1, self.self_excite_escape_event_count)
        momentum_avg_candidates = self.momentum_probe_total_candidates / max(1, self.momentum_probe_hits)
        anchor_edge_count = sum(len(neighbors) for neighbors in self.anchor_neighbors) / 2.0
        anchor_avg_seeds = self.anchor_probe_seed_total / max(1, self.anchor_probe_count)
        anchor_avg_candidates = self.anchor_probe_candidate_total / max(1, self.anchor_probe_count)
        avg_memories_per_turn = float(self.mem_count / max(1, self.p.turns))

        return {
            "turns": float(self.p.turns),
            "query_turns": float(self.query_turns),
            "memory_count": float(self.mem_count),
            "topic_graph_avg_memories_per_turn": avg_memories_per_turn,
            "hot_memory_count": float(loaded_topics),
            "hot_memory_ratio": (float(loaded_topics) / max(1.0, float(active_topics))),
            "cluster_count": float(active_topics),
            "hot_cluster_count": float(loaded_topics),
            "supercluster_count": 0.0,
            "supercluster_rebuild_count": 0.0,
            "avg_clusters_per_super": 0.0,
            "merge_count": float(self.merge_count),
            "new_count": float(self.new_count),
            "merge_rate": merge_rate,
            "normalize_events": 0.0,
            "total_heat": 0.0,
            "heat_gini": 0.0,
            "context_quality_score": (self.context_quality_sum / context_n),
            "context_coverage_score": (self.context_coverage_sum / context_n),
            "context_association_score": (self.context_association_sum / context_n),
            "context_saturation_score": (self.context_saturation_sum / context_n),
            "context_redundancy_score": (self.context_redundancy_sum / context_n),
            "context_irrelevance_score": (self.context_irrelevance_sum / context_n),
            "context_fill_ratio": (self.context_fill_ratio_sum / context_n),
            "topic_graph_avg_origin_topics_per_memory": float(origin_topic_total / max(1, self.mem_count)),
            "topic_graph_multi_topic_memory_ratio": float(multi_topic_count / max(1, self.mem_count)),
            "topic_graph_family_count": float(family_count),
            "topic_graph_family_max_size": float(family_max_size),
            "topic_graph_family_rebuild_count": float(self.topic_family_rebuild_count),
            "topic_graph_self_excite_event_count": float(self.self_excite_event_count),
            "topic_graph_self_excite_skip_events": float(self.self_excite_skip_events),
            "topic_graph_self_excite_avg_penalty": float(self_excite_avg_penalty),
            "topic_graph_self_excite_avg_escape_bonus": float(self_excite_avg_escape),
            "topic_graph_anchor_graph_enabled": 0.0,
            "topic_graph_anchor_count": float(self.anchor_count),
            "topic_graph_anchor_edge_count": float(anchor_edge_count),
            "topic_graph_anchor_probe_count": float(self.anchor_probe_count),
            "topic_graph_anchor_probe_avg_seeds": float(anchor_avg_seeds),
            "topic_graph_anchor_probe_avg_candidates": float(anchor_avg_candidates),
            "topic_graph_momentum_probe_enabled": 1.0 if self.p.topic_graph_momentum_probe_enabled else 0.0,
            "topic_graph_momentum_probe_events": float(self.momentum_probe_events),
            "topic_graph_momentum_probe_hits": float(self.momentum_probe_hits),
            "topic_graph_momentum_probe_avg_candidates": float(momentum_avg_candidates),
            "target_precision_at_k": target_precision,
            "target_recall_recent": target_hit,
            "target_recall_all": target_hit,
            "target_hit_rate": target_hit,
            "target_recent_hit_rate": target_hit,
            "target_mrr": target_mrr,
            "empty_query_rate": (self.empty_query_count / max(1, self.query_turns)),
            "empty_target_query_rate": (self.empty_target_query_count / max(1, self.query_turns)),
            "avg_returned_turns": (self.returned_topics_sum / eval_n),
            "avg_sim_ops_add_per_turn": avg_add_ops,
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
            "topic_graph_topic_hnsw_enabled": 1.0 if self._topic_hnsw_ready() else 0.0,
            "topic_graph_topic_hnsw_build_events": float(self.topic_hnsw_build_events),
            "topic_graph_topic_hnsw_query_events": float(self.topic_hnsw_query_events),
            "topic_graph_topic_hnsw_failure_events": float(self.topic_hnsw_failure_events),
            "preload_attempts": float(self.preload_attempts),
            "preload_successes": float(self.preload_successes),
            "preload_success_rate": (self.preload_successes / max(1, self.preload_attempts)),
            "preload_memories_heated": 0.0,
            "preload_clusters_loaded": float(self.preload_clusters_loaded),
            "preload_topic_predictions": 0.0,
            "preload_correct_predictions": 0.0,
            "preload_prediction_accuracy": 0.0,
            "topic_temporal_state_enabled": 1.0 if self.p.topic_temporal_state_enabled else 0.0,
            "ghsom_enabled": 1.0 if self.p.ghsom_enabled else 0.0,
            "ghsom_active": 1.0 if self._topic_ghsom_active() else 0.0,
            **local_summary,
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
        if not self.snapshots:
            score = float(summary.get("context_quality_score", 0.0))
            summary["context_quality_start"] = score
            summary["context_quality_end"] = score
            summary["context_quality_delta"] = 0.0
            summary["context_quality_max"] = score
            summary["context_quality_min"] = score
            return summary
        vals = [float(s.get("context_quality_score", 0.0)) for s in self.snapshots]
        start = vals[0]
        end = float(summary.get("context_quality_score", vals[-1]))
        summary["context_quality_start"] = start
        summary["context_quality_end"] = end
        summary["context_quality_delta"] = end - start
        summary["context_quality_max"] = max(vals)
        summary["context_quality_min"] = min(vals)
        return summary

    def run(self) -> Dict[str, object]:
        current_topic = self.topic.initial_topic()
        prev_topic: Optional[int] = None

        for turn in range(1, self.p.turns + 1):
            self._decay_bridges()
            self._decay_deep_artmap_feedback()
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

            turn_vec, atomic_records = self.topic.turn_sample_with_atomics(current_topic)
            query_ops = 0

            if turn > self.p.warmup_turns and self.rng.random() < self.p.query_prob:
                target_topic = self._sample_query_topic(current_topic, turn)
                noise_mix = self.p.query_noise_mix
                query_vec, query_facet_ids, query_facet_weights = self.topic.query_sample(target_topic, noise_mix)
                selected_topics, query_ops, debug = self.retrieve(
                    query_vec,
                    current_topic=current_topic,
                    current_turn=turn,
                    query_facet_ids=query_facet_ids,
                    query_facet_weights=query_facet_weights,
                )
                self.sim_ops_query_total += query_ops
                self.query_turns += 1
                if not selected_topics:
                    self.empty_query_count += 1

                hit, _mrr = self._eval_query(selected_topics, target_topic)
                self._eval_context_quality(
                    selected_topics=selected_topics,
                    debug=debug,
                    query_vec=query_vec,
                    query_facet_ids=query_facet_ids,
                    query_facet_weights=query_facet_weights,
                )
                if hit <= 0:
                    self.empty_target_query_count += 1
                self._apply_query_feedback(
                    current_topic=current_topic,
                    target_topic=target_topic,
                    selected_topics=selected_topics,
                    bridge_topics=debug.get("bridge_topics", []),
                    local_signals=debug.get("local_signals"),
                    turn=turn,
                )

            self._register_turn_topic(turn, current_topic)
            add_ops = 0
            momentum_memories: List[int] = []
            for atomic_vec, atomic_facet_ids, atomic_facet_weights in atomic_records:
                one_ops, one_memory_ids = self._add_memory(
                    atomic_vec,
                    turn,
                    current_topic,
                    facet_ids=atomic_facet_ids,
                    facet_weights=atomic_facet_weights,
                )
                add_ops += int(one_ops)
                for mem_id in one_memory_ids:
                    if int(mem_id) not in momentum_memories:
                        momentum_memories.append(int(mem_id))
            # One-turn momentum lane: the last turn's atomic memories may join exactly
            # one future probe when the topic stays the same. Do not generalize this
            # into many synthetic queries or a multi-turn cache, or it turns into
            # uncontrolled "dog-paddle" exploration that breaks local momentum.
            self._update_one_turn_momentum(turn, current_topic, momentum_memories)
            self.sim_ops_add_total += int(add_ops)
            self._maybe_rebuild_topic_families(turn)
            self.turn_sim_ops.append(int(query_ops + add_ops))

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
    context_quality = col("context_quality_score")
    context_coverage = col("context_coverage_score")
    context_association = col("context_association_score")
    context_saturation = col("context_saturation_score")
    context_redundancy = col("context_redundancy_score")
    context_irrelevance = col("context_irrelevance_score")
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
    has_context_quality = any("context_quality_score" in r for r in rows) or ("context_quality_score" in summary)

    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    ax = axes[0, 0]
    if has_context_quality:
        ax.plot(turns, context_quality, marker="o", linewidth=2.0, label="context_quality")
        ax.plot(turns, context_coverage, marker="o", linewidth=2.0, label="coverage")
        ax.plot(turns, context_association, marker="o", linestyle="-.", linewidth=1.8, label="association")
        ax.plot(turns, context_saturation, marker="o", linestyle=":", linewidth=1.8, label="saturation")
        ax.plot(turns, context_redundancy, marker="o", linewidth=2.0, label="redundancy")
        ax.plot(turns, context_irrelevance, marker="o", linestyle="--", linewidth=1.8, label="irrelevance")
        ax.axhline(
            float(summary.get("context_quality_start", context_quality[0] if context_quality else 0.0)),
            color="#888888",
            linestyle="--",
            linewidth=1.4,
        )
        ax.set_title("Context Assembly Quality")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="best")
    else:
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

    title_suffix = (
        f"context_quality_delta={summary.get('context_quality_delta', 0.0):+.3f}"
        if has_context_quality
        else f"gini_delta={summary.get('heat_gini_delta', 0.0):+.3f}"
    )
    fig.suptitle(
        "Agent Memory Simulation Dashboard\n"
        f"turns={int(summary.get('turns', 0))}, merge_rate={summary.get('merge_rate', 0.0):.3f}, "
        f"{title_suffix}",
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
    parser.add_argument("--merge-limit", type=float, default=0.95,
                        help="Merge a new memory into an existing one when cosine similarity exceeds this threshold")
    parser.add_argument("--topic-temporal-state", type=lambda x: x.lower() == 'true', default=False,
                        help="Enable temporal topic state with bundle weights and velocity")
    parser.add_argument("--topic-temporal-velocity-mix", type=float, default=0.32,
                        help="How strongly bundle velocity steers temporal topic state")
    parser.add_argument("--topic-temporal-momentum", type=float, default=0.78,
                        help="Momentum of temporal bundle velocity")
    parser.add_argument("--topic-temporal-reversion", type=float, default=0.06,
                        help="Mean reversion of temporal bundle state toward its base distribution")
    parser.add_argument("--topic-temporal-noise", type=float, default=0.07,
                        help="Noise injected into temporal bundle velocity")
    parser.add_argument("--topic-temporal-focus-stickiness", type=float, default=0.72,
                        help="Probability of keeping the same latent bundle focus on the next turn")
    parser.add_argument("--topic-temporal-turn-mix", type=float, default=0.78,
                        help="How much turn generation follows the temporal bundle state instead of the base topic mix")
    parser.add_argument("--topic-temporal-query-mix", type=float, default=0.68,
                        help="How much query generation follows the temporal bundle state instead of the base topic mix")
    parser.add_argument("--topic-temporal-residual-mix", type=float, default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-temporal-query-residual-mix", type=float, default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-temporal-residual-drift", type=float, default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-atomic-min", type=int, default=1,
                        help="Minimum number of atomic memories extracted from each simulated turn")
    parser.add_argument("--topic-atomic-max", type=int, default=3,
                        help="Maximum number of atomic memories extracted from each simulated turn")
    parser.add_argument("--topic-atomic-facet-min", type=int, default=1,
                        help="Minimum number of latent facets carried by one simulated atomic memory")
    parser.add_argument("--topic-atomic-facet-max", type=int, default=2,
                        help="Maximum number of latent facets carried by one simulated atomic memory")
    parser.add_argument("--topic-atomic-secondary-mix", type=float, default=0.16,
                        help="Weak secondary-facet weight used to keep simulated atomic memories component-like instead of centroid-like")
    parser.add_argument("--topic-atomic-max-self-sim", type=float, default=0.93,
                        help="Target maximum cosine similarity between atomic memories generated from the same turn")
    parser.add_argument("--topic-atomic-anchor-deflate", type=float, default=0.08,
                        help="Subtract a small amount of turn-anchor direction so one atomic memory behaves more like a turn component")
    parser.add_argument("--topic-anchor-graph-enabled", type=lambda x: x.lower() == 'true', default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-anchor-seed-k", type=int, default=2,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-anchor-neighbor-topk", type=int, default=2,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-anchor-max-hops", type=int, default=1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-anchor-edge-decay", type=float, default=0.65,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-momentum-probe-enabled", type=lambda x: x.lower() == 'true', default=False,
                        help="Let the previous turn's atomic memories join exactly one next-turn probe when the topic stays unchanged")
    parser.add_argument("--topic-momentum-probe-cap", type=int, default=1,
                        help="Maximum number of previous-turn atomic memories allowed into the one-turn momentum probe")
    parser.add_argument("--topic-momentum-probe-min-sim", type=float, default=0.58,
                        help="Minimum cosine similarity required for a previous-turn atomic memory to join the next-turn momentum probe")
    parser.add_argument("--topic-momentum-probe-topic-margin", type=float, default=0.03,
                        help="Only use one-turn momentum when the current topic stays within this semantic margin of the best query topic")
    parser.add_argument("--context-redundancy-weight", type=float, default=0.35,
                        help="Penalty weight for facet-overlap redundancy in context assembly")
    parser.add_argument("--context-irrelevance-weight", type=float, default=0.20,
                        help="Penalty weight for off-query memories in context assembly")
    parser.add_argument("--context-association-weight", type=float, default=0.16,
                        help="Bonus weight for opening new topic/family associative space during context assembly")
    parser.add_argument("--context-saturation-weight", type=float, default=0.22,
                        help="Penalty weight for near-duplicate saturation during context assembly")
    parser.add_argument("--context-similarity-threshold", type=float, default=0.95,
                        help="Cosine threshold above which similar memories start saturating the context budget")
    parser.add_argument("--context-min-marginal-gain", type=float, default=0.02,
                        help="Stop filling context budget once the best remaining memory falls below this marginal gain")
    
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
    parser.add_argument("--topic-family-topk", type=int, default=3,
                        help="Build topic families from mutual top-k topic neighbors")
    parser.add_argument("--topic-family-similarity", type=float, default=0.82,
                        help="Minimum topic similarity for the family graph")
    parser.add_argument("--topic-family-member-limit", type=int, default=2,
                        help="Maximum topics from the same family allowed in one query expansion/result set")
    parser.add_argument("--topic-family-source", type=str, default="deep_artmap",
                        choices=["centroid", "deep_artmap", "deep_artmap_graph"],
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-rebuild-interval", type=int, default=128,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-exemplars-per-topic", type=int, default=3,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-query-exemplars", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-query-gate", type=float, default=0.56,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-hnsw-enabled", type=lambda x: x.lower() == 'true', default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-hnsw-min-members", type=int, default=24,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-hnsw-k", type=int, default=12,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-hnsw-m", type=int, default=16,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-hnsw-ef-construction", type=int, default=80,
                        help=argparse.SUPPRESS)
    parser.add_argument("--topic-family-hnsw-ef-search", type=int, default=24,
                        help=argparse.SUPPRESS)
    parser.add_argument("--self-excite-penalty", type=float, default=0.18,
                        help="Penalty for same-family or anchor-family expansion during topic_graph search")
    parser.add_argument("--family-revisit-penalty", type=float, default=0.10,
                        help="Extra penalty when the search revisits an already visited topic family")
    parser.add_argument("--family-escape-bonus", type=float, default=0.06,
                        help="Bonus for expanding into an unseen family outside the current anchor family")
    parser.add_argument("--topic-local-index", type=str, default="topoart",
                        choices=["topoart", "deep_artmap", "ghsom", "exact"],
                        help="Topic-local index used inside topic_graph retrieval")
    parser.add_argument("--topoart-min-members", type=int, default=8,
                        help="Only use TopoART routing after a topic has at least this many memories")
    parser.add_argument("--deep-artmap-min-members", type=int, default=4,
                        help="Only use deep_artmap routing after a topic has at least this many memories")
    parser.add_argument("--topoart-vigilance", type=float, default=0.88,
                        help="Primary TopoART resonance threshold")
    parser.add_argument("--topoart-secondary-vigilance", type=float, default=0.80,
                        help="Secondary TopoART resonance threshold used for topology links")
    parser.add_argument("--topoart-beta", type=float, default=0.28,
                        help="TopoART winner prototype learning rate")
    parser.add_argument("--topoart-beta-secondary", type=float, default=0.10,
                        help="TopoART secondary prototype learning rate")
    parser.add_argument("--topoart-query-categories", type=int, default=4,
                        help="Max TopoART categories to expand during a query")
    parser.add_argument("--topoart-neighbor-topk", type=int, default=3,
                        help="Max linked TopoART neighbor categories considered per winner")
    parser.add_argument("--topoart-exemplars", type=int, default=12,
                        help="Representative memories stored per TopoART category")
    parser.add_argument("--topoart-prune-interval", type=int, default=256,
                        help="TopoART maintenance interval; 0 disables category pruning")
    parser.add_argument("--topoart-prune-min-support", type=int, default=2,
                        help="Prune TopoART categories below this support if stale")
    parser.add_argument("--topoart-match-slack", type=float, default=0.05,
                        help="Adaptive vigilance slack below a category's recent match quality")
    parser.add_argument("--topoart-category-capacity", type=int, default=6,
                        help="Start tightening TopoART vigilance once a category grows beyond this size")
    parser.add_argument("--topoart-capacity-boost", type=float, default=0.030,
                        help="Extra vigilance added to oversized TopoART categories")
    parser.add_argument("--topoart-link-margin", type=float, default=0.045,
                        help="Link a near runner-up category when its similarity is within this margin of the winner")
    parser.add_argument("--topoart-query-margin", type=float, default=0.080,
                        help="Expand extra query categories within this similarity margin of the best category")
    parser.add_argument("--topoart-temporal-link-window", type=int, default=6,
                        help="Link recently consecutive TopoART winners within the same topic")
    parser.add_argument("--deep-artmap-bundle-vigilance", type=float, default=0.78,
                        help="Bundle-level resonance threshold for deep_artmap")
    parser.add_argument("--deep-artmap-bundle-beta", type=float, default=0.18,
                        help="Bundle-level prototype learning rate for deep_artmap")
    parser.add_argument("--deep-artmap-query-bundles", type=int, default=3,
                        help="Max context bundles to expand per topic query in deep_artmap")
    parser.add_argument("--deep-artmap-query-margin", type=float, default=0.10,
                        help="Expand additional bundles within this margin of the best bundle")
    parser.add_argument("--deep-artmap-neighbor-topk", type=int, default=2,
                        help="Max linked neighbor bundles considered per selected bundle")
    parser.add_argument("--deep-artmap-temporal-link-window", type=int, default=8,
                        help="Link recent consecutive deep_artmap bundles within the same topic")
    parser.add_argument("--deep-artmap-bundle-prior-weight", type=float, default=0.18,
                        help="How much learned bundle recall/adopt priors affect deep_artmap query ranking")
    parser.add_argument("--deep-artmap-bundle-recall-lr", type=float, default=0.10,
                        help="Feedback learning rate for bundles that help recall the target topic")
    parser.add_argument("--deep-artmap-bundle-adopt-lr", type=float, default=0.18,
                        help="Feedback learning rate for bundles used by the top adopted topic")
    parser.add_argument("--deep-artmap-bundle-edge-feedback", type=float, default=0.14,
                        help="Extra reinforcement added to co-used bundle links after a successful hit")
    parser.add_argument("--deep-artmap-bundle-decay", type=float, default=0.996,
                        help="Per-turn decay for deep_artmap bundle priors and bundle links")
    parser.add_argument("--topic-hnsw-enabled", type=lambda x: x.lower() == 'true', default=False,
                        help="Enable a global topic-centroid HNSW for seed topic retrieval")
    parser.add_argument("--topic-hnsw-k", type=int, default=48,
                        help="Topic HNSW search width before bridge expansion")
    parser.add_argument("--topic-hnsw-m", type=int, default=16,
                        help="Topic HNSW M parameter")
    parser.add_argument("--topic-hnsw-ef-construction", type=int, default=80,
                        help="Topic HNSW ef_construction")
    parser.add_argument("--topic-hnsw-ef-search", type=int, default=32,
                        help="Topic HNSW ef_search; used as the topic seed cost proxy")
    
    args = parser.parse_args()

    if args.topic_hnsw_enabled and HNSWIndex is None:
        raise RuntimeError("HNSW requested but mori_hnsw binding is unavailable")
    
    ghsom_enabled = args.ghsom_enabled
    if ghsom_enabled is None:
        ghsom_enabled = (
            args.retrieval_model == "topic_graph" and args.topic_local_index == "ghsom"
        )

    params = SimParams(
        retrieval_model=args.retrieval_model,
        turns=args.turns,
        seed=args.seed,
        report_every=args.report_every,
        smart_preload_enabled=args.preload_enabled,
        preload_budget_per_query=args.preload_budget,
        preload_max_io_per_turn=args.preload_max_io,
        memory_drop_sim=max(-1.0, min(1.0, float(args.memory_drop_sim))),
        merge_limit=max(-1.0, min(1.0, float(args.merge_limit))),
        topic_temporal_state_enabled=bool(args.topic_temporal_state),
        topic_temporal_velocity_mix=max(0.0, float(args.topic_temporal_velocity_mix)),
        topic_temporal_state_momentum=max(0.0, min(0.999, float(args.topic_temporal_momentum))),
        topic_temporal_state_reversion=max(0.0, float(args.topic_temporal_reversion)),
        topic_temporal_state_noise=max(0.0, float(args.topic_temporal_noise)),
        topic_temporal_focus_stickiness=max(0.0, min(0.999, float(args.topic_temporal_focus_stickiness))),
        topic_temporal_turn_state_mix=max(0.0, min(1.0, float(args.topic_temporal_turn_mix))),
        topic_temporal_query_state_mix=max(0.0, min(1.0, float(args.topic_temporal_query_mix))),
        topic_atomic_memories_min=max(1, int(args.topic_atomic_min)),
        topic_atomic_memories_max=max(max(1, int(args.topic_atomic_min)), int(args.topic_atomic_max)),
        topic_atomic_facet_min=max(1, int(args.topic_atomic_facet_min)),
        topic_atomic_facet_max=max(max(1, int(args.topic_atomic_facet_min)), int(args.topic_atomic_facet_max)),
        topic_atomic_secondary_mix=max(0.0, float(args.topic_atomic_secondary_mix)),
        topic_atomic_max_self_similarity=max(-1.0, min(0.999, float(args.topic_atomic_max_self_sim))),
        topic_atomic_anchor_deflate=max(0.0, min(0.20, float(args.topic_atomic_anchor_deflate))),
        topic_graph_anchor_graph_enabled=False,
        topic_graph_anchor_seed_k=max(1, int(args.topic_anchor_seed_k)),
        topic_graph_anchor_neighbor_topk=max(1, int(args.topic_anchor_neighbor_topk)),
        topic_graph_anchor_max_hops=max(0, int(args.topic_anchor_max_hops)),
        topic_graph_anchor_edge_decay=max(0.0, min(1.0, float(args.topic_anchor_edge_decay))),
        topic_graph_momentum_probe_enabled=bool(args.topic_momentum_probe_enabled),
        topic_graph_momentum_probe_cap=max(1, int(args.topic_momentum_probe_cap)),
        topic_graph_momentum_probe_min_sim=max(-1.0, min(1.0, float(args.topic_momentum_probe_min_sim))),
        topic_graph_momentum_probe_topic_margin=max(0.0, float(args.topic_momentum_probe_topic_margin)),
        context_redundancy_weight=max(0.0, float(args.context_redundancy_weight)),
        context_irrelevance_weight=max(0.0, float(args.context_irrelevance_weight)),
        context_association_weight=max(0.0, float(args.context_association_weight)),
        context_saturation_weight=max(0.0, float(args.context_saturation_weight)),
        context_similarity_saturation_threshold=max(-1.0, min(0.999, float(args.context_similarity_threshold))),
        context_min_marginal_gain=float(args.context_min_marginal_gain),
        soft_gate_enabled=args.soft_gate_enabled,
        ghsom_enabled=ghsom_enabled,
        ghsom_max_depth=max(1, int(args.ghsom_max_depth)),
        ghsom_min_samples_for_expansion=max(2, int(args.ghsom_min_samples)),
        ghsom_linear_scan_threshold=max(2, int(args.ghsom_threshold)),
        topic_graph_family_topk=max(1, int(args.topic_family_topk)),
        topic_graph_family_similarity=max(-1.0, min(1.0, float(args.topic_family_similarity))),
        topic_graph_family_member_limit=max(1, int(args.topic_family_member_limit)),
        topic_graph_self_excite_penalty=max(0.0, float(args.self_excite_penalty)),
        topic_graph_family_revisit_penalty=max(0.0, float(args.family_revisit_penalty)),
        topic_graph_family_escape_bonus=max(0.0, float(args.family_escape_bonus)),
        topic_graph_local_index=str(args.topic_local_index).strip().lower(),
        topic_graph_topoart_min_members=max(2, int(args.topoart_min_members)),
        topic_graph_deep_artmap_min_members=max(2, int(args.deep_artmap_min_members)),
        topic_graph_topoart_vigilance=max(-1.0, min(1.0, float(args.topoart_vigilance))),
        topic_graph_topoart_secondary_vigilance=max(-1.0, min(1.0, float(args.topoart_secondary_vigilance))),
        topic_graph_topoart_beta=max(0.0, min(1.0, float(args.topoart_beta))),
        topic_graph_topoart_beta_secondary=max(0.0, min(1.0, float(args.topoart_beta_secondary))),
        topic_graph_topoart_query_categories=max(1, int(args.topoart_query_categories)),
        topic_graph_topoart_neighbor_topk=max(0, int(args.topoart_neighbor_topk)),
        topic_graph_topoart_exemplars=max(1, int(args.topoart_exemplars)),
        topic_graph_topoart_prune_interval=max(0, int(args.topoart_prune_interval)),
        topic_graph_topoart_prune_min_support=max(1, int(args.topoart_prune_min_support)),
        topic_graph_topoart_match_slack=max(0.0, float(args.topoart_match_slack)),
        topic_graph_topoart_category_capacity=max(4, int(args.topoart_category_capacity)),
        topic_graph_topoart_capacity_boost=max(0.0, float(args.topoart_capacity_boost)),
        topic_graph_topoart_link_margin=max(0.0, float(args.topoart_link_margin)),
        topic_graph_topoart_query_margin=max(0.0, float(args.topoart_query_margin)),
        topic_graph_topoart_temporal_link_window=max(0, int(args.topoart_temporal_link_window)),
        topic_graph_deep_artmap_bundle_vigilance=max(-1.0, min(1.0, float(args.deep_artmap_bundle_vigilance))),
        topic_graph_deep_artmap_bundle_beta=max(0.0, min(1.0, float(args.deep_artmap_bundle_beta))),
        topic_graph_deep_artmap_query_bundles=max(1, int(args.deep_artmap_query_bundles)),
        topic_graph_deep_artmap_query_margin=max(0.0, float(args.deep_artmap_query_margin)),
        topic_graph_deep_artmap_neighbor_topk=max(0, int(args.deep_artmap_neighbor_topk)),
        topic_graph_deep_artmap_temporal_link_window=max(0, int(args.deep_artmap_temporal_link_window)),
        topic_graph_deep_artmap_bundle_prior_weight=max(0.0, float(args.deep_artmap_bundle_prior_weight)),
        topic_graph_deep_artmap_bundle_recall_lr=max(0.0, float(args.deep_artmap_bundle_recall_lr)),
        topic_graph_deep_artmap_bundle_adopt_lr=max(0.0, float(args.deep_artmap_bundle_adopt_lr)),
        topic_graph_deep_artmap_bundle_edge_feedback=max(0.0, float(args.deep_artmap_bundle_edge_feedback)),
        topic_graph_deep_artmap_bundle_decay=max(0.0, min(0.9999, float(args.deep_artmap_bundle_decay))),
        topic_graph_topic_hnsw_enabled=bool(args.topic_hnsw_enabled),
        topic_graph_topic_hnsw_k=max(4, int(args.topic_hnsw_k)),
        topic_graph_topic_hnsw_m=max(4, int(args.topic_hnsw_m)),
        topic_graph_topic_hnsw_ef_construction=max(16, int(args.topic_hnsw_ef_construction)),
        topic_graph_topic_hnsw_ef_search=max(8, int(args.topic_hnsw_ef_search)),
    )
    
    print(f"Retrieval model: {args.retrieval_model}")
    print(f"Running simulation with {args.turns} turns...")
    print(f"Smart preloading: {'enabled' if args.preload_enabled else 'disabled'}")
    print(f"Soft gate: {'enabled' if args.soft_gate_enabled else 'disabled'}")
    print(
        "Temporal topic state: "
        f"{'enabled' if args.topic_temporal_state else 'disabled'}"
    )
    print(
        "Atomic memories per turn: "
        f"{int(args.topic_atomic_min)}-{int(args.topic_atomic_max)} "
        f"(facets {int(args.topic_atomic_facet_min)}-{int(args.topic_atomic_facet_max)}, "
        f"max_self_sim={float(args.topic_atomic_max_self_sim):.2f})"
    )
    if args.retrieval_model == "memory":
        print(f"GHSOM index: {'enabled' if ghsom_enabled else 'disabled'}")
    else:
        print(f"Topic-local index: {args.topic_local_index}")
        if args.topic_local_index == "ghsom":
            print(
                "GHSOM index: "
                f"configured={ghsom_enabled}, active={'yes' if ghsom_enabled else 'no'} "
                "(topic_graph local pruning)"
            )
        elif args.topic_local_index == "deep_artmap":
            print(
                "Deep ARTMAP: "
                f"min_members={int(args.deep_artmap_min_members)}, "
                f"bundle_vigilance={float(args.deep_artmap_bundle_vigilance):.2f}, "
                f"query_bundles={int(args.deep_artmap_query_bundles)}, "
                "layers=2"
            )
        elif args.topic_local_index == "topoart":
            print(
                "TopoART: "
                f"vigilance={float(args.topoart_vigilance):.2f}, "
                f"secondary={float(args.topoart_secondary_vigilance):.2f}, "
                f"query_categories={int(args.topoart_query_categories)}, "
                f"capacity={int(args.topoart_category_capacity)}"
            )
    if args.retrieval_model == "topic_graph":
        family_mode = "deep_artmap" if args.topic_local_index == "deep_artmap" else "centroid"
        print(
            "Topic HNSW: "
            f"{'enabled' if args.topic_hnsw_enabled else 'disabled'} "
            f"(k={int(args.topic_hnsw_k)})"
        )
        print(
            "One-turn momentum probe: "
            f"{'enabled' if args.topic_momentum_probe_enabled else 'disabled'} "
            f"(cap={int(args.topic_momentum_probe_cap)}, "
            f"min_sim={float(args.topic_momentum_probe_min_sim):.2f}, "
            f"topic_margin={float(args.topic_momentum_probe_topic_margin):.2f})"
        )
        print(
            "Topic families: "
            f"topk={int(args.topic_family_topk)}, "
            f"sim={float(args.topic_family_similarity):.2f}, "
            f"member_limit={int(args.topic_family_member_limit)}, "
            f"mode={family_mode}"
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
        print(
            "Context quality/coverage/association/saturation: "
            f"{summary.get('context_quality_score', 0.0):.4f}/"
            f"{summary.get('context_coverage_score', 0.0):.4f}/"
            f"{summary.get('context_association_score', 0.0):.4f}/"
            f"{summary.get('context_saturation_score', 0.0):.4f}"
        )
        print(
            "Merge rate/origin-topics: "
            f"{summary.get('merge_rate', 0.0):.4f}/"
            f"{summary.get('topic_graph_avg_origin_topics_per_memory', 0.0):.2f}"
        )
        print(
            "Context fill/redundancy/irrelevance: "
            f"{summary.get('context_fill_ratio', 0.0):.4f}/"
            f"{summary.get('context_redundancy_score', 0.0):.4f}/"
            f"{summary.get('context_irrelevance_score', 0.0):.4f}"
        )
        print(
            "Family risk penalty/skip: "
            f"{summary.get('topic_graph_self_excite_avg_penalty', 0.0):.4f}/"
            f"{summary.get('topic_graph_self_excite_skip_events', 0.0):.0f}"
        )
        print(
            "Average memories per turn: "
            f"{summary.get('topic_graph_avg_memories_per_turn', 0.0):.2f}"
        )
        print(
            "Momentum probe events/hits/avg candidates: "
            f"{summary.get('topic_graph_momentum_probe_events', 0.0):.0f}/"
            f"{summary.get('topic_graph_momentum_probe_hits', 0.0):.0f}/"
            f"{summary.get('topic_graph_momentum_probe_avg_candidates', 0.0):.2f}"
        )
        print(
            "Topic families/rebuilds: "
            f"{summary.get('topic_graph_family_count', 0.0):.0f}/"
            f"{summary.get('topic_graph_family_rebuild_count', 0.0):.0f}"
        )
        print(f"Topic graph bridge edges: {summary.get('topic_graph_bridge_edges', 0):.0f}")
        print(f"Loaded topics: {summary.get('topic_graph_loaded_topics', 0):.0f}")
        if args.topic_local_index == "topoart":
            print(f"TopoART categories: {summary.get('topoart_category_count', 0):.0f}")
            print(f"TopoART avg categories/topic: {summary.get('topoart_avg_categories_per_active_topic', 0.0):.2f}")
            print(f"TopoART avg categories/query: {summary.get('topoart_probe_avg_categories', 0.0):.2f}")
        if args.topic_local_index == "deep_artmap":
            print(f"Layer-0 categories: {summary.get('topoart_category_count', 0):.0f}")
            print(f"Layer-0 avg categories/topic: {summary.get('topoart_avg_categories_per_active_topic', 0.0):.2f}")
            print(f"Deep ARTMAP bundles: {summary.get('deep_artmap_bundle_count', 0):.0f}")
            print(f"Deep ARTMAP avg categories/bundle: {summary.get('deep_artmap_avg_categories_per_bundle', 0.0):.2f}")
            print(f"Deep ARTMAP avg bundles/query: {summary.get('deep_artmap_probe_avg_bundles', 0.0):.2f}")
            print(f"Deep ARTMAP avg categories/query: {summary.get('deep_artmap_probe_avg_categories', 0.0):.2f}")
            print(f"Deep ARTMAP prior(recall/adopt): {summary.get('deep_artmap_bundle_avg_recall_prior', 0.0):.3f}/{summary.get('deep_artmap_bundle_avg_adopt_prior', 0.0):.3f}")
        if args.topic_hnsw_enabled:
            print(f"Topic HNSW query events: {summary.get('topic_graph_topic_hnsw_query_events', 0):.0f}")
    if args.topic_local_index == "ghsom" and ghsom_enabled:
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
