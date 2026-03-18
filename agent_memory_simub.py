#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from bridge import MoriMemoryBridge


TURN_RE = re.compile(r"第(\d+)轮")
FACT_RE = re.compile(r"fact:(F\d{5,})")

DEFAULT_TOPICS = (
    "anime",
    "food",
    "travel",
    "tech",
    "pet",
    "fitness",
)


@dataclass(frozen=True)
class TopicSpec:
    topic_id: str
    shared_label: str
    personal_label: str


TOPIC_SPECS = {
    "anime": TopicSpec("anime", "seasonal_anime", "watch_pref"),
    "food": TopicSpec("food", "late_night_food", "spice_pref"),
    "travel": TopicSpec("travel", "weekend_trip", "route_pref"),
    "tech": TopicSpec("tech", "device_setup", "tool_pref"),
    "pet": TopicSpec("pet", "pet_care", "snack_pref"),
    "fitness": TopicSpec("fitness", "training_plan", "recovery_pref"),
}


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 1e-12:
        return [0.0 for _ in vec]
    return [x / norm for x in vec]


def blend(parts: Iterable[tuple[list[float], float]]) -> list[float]:
    acc: list[float] | None = None
    for vec, weight in parts:
        if acc is None:
            acc = [0.0 for _ in vec]
        for idx, value in enumerate(vec):
            acc[idx] += value * weight
    return normalize(acc or [])


def random_unit_vector(rng: random.Random, dim: int) -> list[float]:
    return normalize([rng.gauss(0.0, 1.0) for _ in range(dim)])


def stats_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    pos = clamp(percentile, 0.0, 100.0) / 100.0 * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    alpha = pos - lo
    return float(ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha)


def summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": len(values),
        "p50_ms": round(stats_percentile(values, 50.0), 3),
        "p95_ms": round(stats_percentile(values, 95.0), 3),
        "max_ms": round(max(values), 3),
    }


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def choose_weighted(rng: random.Random, weighted_items: list[tuple[str, float]]) -> str:
    total = 0.0
    for _, weight in weighted_items:
        total += max(0.0, float(weight))
    if total <= 1e-9:
        return weighted_items[0][0]
    pick = rng.random() * total
    running = 0.0
    for item, weight in weighted_items:
        running += max(0.0, float(weight))
        if pick <= running:
            return item
    return weighted_items[-1][0]


def parse_turns_from_blocks(blocks: list[str]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for block in blocks:
        for match in TURN_RE.findall(block):
            turn = int(match)
            if turn > 0 and turn not in seen:
                seen.add(turn)
                out.append(turn)
    return out


def parse_fact_ids_from_blocks(blocks: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        for fact_id in FACT_RE.findall(block):
            if fact_id not in seen:
                seen.add(fact_id)
                out.append(fact_id)
    return out


def flatten_block_contents(lua_blocks: Any) -> list[str]:
    out: list[str] = []
    idx = 1
    while True:
        block = lua_blocks[idx]
        if block is None:
            break
        content = block["content"]
        if content is not None:
            out.append(str(content))
        idx += 1
    return out


def make_fact_text(
    fact_id: str,
    room_id: str,
    user_id: str,
    topic_id: str,
    mode: str,
    ordinal: int,
) -> str:
    spec = TOPIC_SPECS[topic_id]
    if mode == "shared_statement":
        return (
            f"fact:{fact_id} room={room_id} topic={topic_id} mode=shared "
            f"shared_label={spec.shared_label} turn_note={ordinal} "
            f"focus=room_discussion stable_hint=keep_room_scope"
        )
    return (
        f"fact:{fact_id} room={room_id} user={user_id} topic={topic_id} mode=personal "
        f"personal_label={spec.personal_label} turn_note={ordinal} "
        f"profile_hint=keep_actor_identity stable_hint=avoid_cross_user_mix"
    )


def make_query_text(room_id: str, user_id: str, topic_id: str, mode: str, fragmented: bool) -> str:
    spec = TOPIC_SPECS[topic_id]
    if mode == "shared_query":
        if fragmented:
            return f"继续刚才 room={room_id} topic={topic_id} {spec.shared_label}"
        return (
            f"继续 room={room_id} topic={topic_id} 的共享话题，"
            f"请沿着 {spec.shared_label} 那条线接着说。"
        )
    if fragmented:
        return f"还是我刚才 user={user_id} topic={topic_id} {spec.personal_label}"
    return (
        f"继续我刚才 user={user_id} topic={topic_id} 的个人偏好，"
        f"请沿着 {spec.personal_label} 那条线接着说。"
    )


def make_assistant_text(turn: int, room_id: str, user_id: str, topic_id: str, mode: str) -> str:
    if "personal" in mode:
        return (
            f"ack turn={turn} room={room_id} user={user_id} topic={topic_id} "
            f"personal_context_kept"
        )
    return (
        f"ack turn={turn} room={room_id} user={user_id} topic={topic_id} "
        f"shared_context_kept"
    )


def to_lua_table(lua: Any, value: Any) -> Any:
    if isinstance(value, dict):
        table = lua.table()
        for key, item in value.items():
            table[key] = to_lua_table(lua, item)
        return table
    if isinstance(value, (list, tuple)):
        table = lua.table()
        for idx, item in enumerate(value, start=1):
            table[idx] = to_lua_table(lua, item)
        return table
    return value


class LiveScenario:
    def __init__(
        self,
        *,
        seed: int,
        dim: int,
        turns: int,
        rooms: int,
        users_per_room: int,
        topics: list[str],
    ) -> None:
        self.seed = seed
        self.dim = dim
        self.turns = turns
        self.rooms = rooms
        self.users_per_room = users_per_room
        self.topic_ids = list(topics)
        self.rng = random.Random(seed)

        self.room_ids = [f"room{idx + 1}" for idx in range(rooms)]
        self.room_bias = {
            room_id: random_unit_vector(self.rng, dim) for room_id in self.room_ids
        }
        self.topic_base = {
            topic_id: random_unit_vector(self.rng, dim) for topic_id in self.topic_ids
        }
        self.users_by_room: dict[str, list[dict[str, str]]] = {}
        self.user_bias: dict[str, list[float]] = {}
        self.shared_history: dict[str, list[int]] = {}
        self.personal_history: dict[str, list[int]] = {}
        self.fact_id_for_turn: dict[int, str] = {}
        self.turn_for_fact_id: dict[str, int] = {}
        self.turn_meta: dict[int, dict[str, Any]] = {}
        self.last_room_id = self.room_ids[0]

        for room_id in self.room_ids:
            room_users: list[dict[str, str]] = []
            for idx in range(users_per_room):
                user_id = f"{room_id}_u{idx + 1}"
                shared_topic = self.topic_ids[idx % len(self.topic_ids)]
                personal_topic = self.topic_ids[(idx + 2) % len(self.topic_ids)]
                room_users.append(
                    {
                        "user_id": user_id,
                        "nickname": f"{room_id}_nick_{idx + 1}",
                        "shared_topic": shared_topic,
                        "personal_topic": personal_topic,
                    }
                )
                self.user_bias[user_id] = random_unit_vector(self.rng, dim)
            self.users_by_room[room_id] = room_users

    def _shared_key(self, room_id: str, topic_id: str) -> str:
        return f"shared:{room_id}:{topic_id}"

    def _personal_key(self, room_id: str, user_id: str, topic_id: str) -> str:
        return f"personal:{room_id}:{user_id}:{topic_id}"

    def _pick_room(self) -> str:
        if self.rng.random() < 0.68:
            return self.last_room_id
        return self.rng.choice(self.room_ids)

    def _pick_user(self, room_id: str) -> dict[str, str]:
        users = self.users_by_room[room_id]
        return self.rng.choice(users)

    def _make_shared_vec(self, room_id: str, user_id: str, topic_id: str, noise_scale: float) -> list[float]:
        return blend(
            (
                (self.topic_base[topic_id], 0.95),
                (self.room_bias[room_id], 0.10),
                (self.user_bias[user_id], 0.03),
                (random_unit_vector(self.rng, self.dim), noise_scale),
            )
        )

    def _make_personal_vec(self, room_id: str, user_id: str, topic_id: str, noise_scale: float) -> list[float]:
        return blend(
            (
                (self.topic_base[topic_id], 0.84),
                (self.room_bias[room_id], 0.08),
                (self.user_bias[user_id], 0.24),
                (random_unit_vector(self.rng, self.dim), noise_scale),
            )
        )

    def _history_candidates(self, room_id: str) -> tuple[list[str], list[str]]:
        shared_keys = [
            key
            for key, turns in self.shared_history.items()
            if key.startswith(f"shared:{room_id}:") and turns
        ]
        personal_keys = [
            key
            for key, turns in self.personal_history.items()
            if key.startswith(f"personal:{room_id}:") and turns
        ]
        return shared_keys, personal_keys

    def _record_turn(self, turn: int, meta: dict[str, Any]) -> None:
        self.turn_meta[turn] = meta
        fact_id = str(meta.get("fact_id") or "")
        if fact_id:
            self.fact_id_for_turn[turn] = fact_id
            self.turn_for_fact_id[fact_id] = turn
        if meta["mode"] == "shared_statement":
            self.shared_history.setdefault(meta["key"], []).append(turn)
        elif meta["mode"] == "personal_statement":
            self.personal_history.setdefault(meta["key"], []).append(turn)

    def make_turn(self, turn: int) -> dict[str, Any]:
        room_id = self._pick_room()
        self.last_room_id = room_id
        user = self._pick_user(room_id)
        shared_candidates, personal_candidates = self._history_candidates(room_id)

        mode = choose_weighted(
            self.rng,
            [
                ("shared_statement", 0.32),
                ("personal_statement", 0.28),
                ("shared_query", 0.18 if shared_candidates else 0.0),
                ("personal_query", 0.14 if personal_candidates else 0.0),
                ("fragment_query", 0.08 if (shared_candidates or personal_candidates) else 0.0),
            ],
        )

        fragmented = mode == "fragment_query"
        fact_id = ""
        expected_turns: list[int] = []
        broad_key = ""
        strict_key = ""
        eval_scope = ""

        if mode == "shared_statement":
            topic_id = user["shared_topic"]
            key = self._shared_key(room_id, topic_id)
            user_id = user["user_id"]
            fact_id = f"F{turn:05d}"
            user_input = make_fact_text(fact_id, room_id, user_id, topic_id, mode, turn)
            query_vec = self._make_shared_vec(room_id, user_id, topic_id, 0.03)
            user_vec = query_vec
            broad_key = key
            strict_key = key
            eval_scope = "none"
        elif mode == "personal_statement":
            topic_id = user["personal_topic"]
            key = self._personal_key(room_id, user["user_id"], topic_id)
            user_id = user["user_id"]
            fact_id = f"F{turn:05d}"
            user_input = make_fact_text(fact_id, room_id, user_id, topic_id, mode, turn)
            query_vec = self._make_personal_vec(room_id, user_id, topic_id, 0.03)
            user_vec = query_vec
            broad_key = f"room_topic:{room_id}:{topic_id}"
            strict_key = key
            eval_scope = "none"
        else:
            use_personal = False
            if mode == "personal_query":
                use_personal = True
            elif mode == "fragment_query":
                use_personal = bool(personal_candidates) and (
                    not shared_candidates or self.rng.random() < 0.58
                )

            if use_personal:
                key = self.rng.choice(personal_candidates)
                _, room_id_from_key, user_id, topic_id = key.split(":", 3)
                room_id = room_id_from_key
                user = next(
                    item for item in self.users_by_room[room_id] if item["user_id"] == user_id
                )
                expected_turns = list(self.personal_history.get(key, [])[-3:])
                broad_key = f"room_topic:{room_id}:{topic_id}"
                strict_key = key
                eval_scope = "personal"
                query_mode = "personal_query"
                query_vec = self._make_personal_vec(room_id, user_id, topic_id, 0.02)
            else:
                key = self.rng.choice(shared_candidates)
                _, room_id_from_key, topic_id = key.split(":", 2)
                room_id = room_id_from_key
                user = self._pick_user(room_id)
                user_id = user["user_id"]
                expected_turns = list(self.shared_history.get(key, [])[-3:])
                broad_key = key
                strict_key = key
                eval_scope = "shared"
                query_mode = "shared_query"
                query_vec = self._make_shared_vec(room_id, user_id, topic_id, 0.02)

            mode = query_mode
            user_input = make_query_text(room_id, user["user_id"], topic_id, mode, fragmented)
            user_vec = query_vec

        assistant_text = make_assistant_text(turn, room_id, user["user_id"], topic_id, mode)
        meta = {
            "turn": turn,
            "room_id": room_id,
            "user_id": user["user_id"],
            "nickname": user["nickname"],
            "topic_id": topic_id,
            "mode": mode,
            "key": strict_key if strict_key else key,
            "broad_key": broad_key,
            "strict_key": strict_key if strict_key else key,
            "expected_turns": expected_turns,
            "eval_scope": eval_scope,
            "fact_id": fact_id,
            "user_input": user_input,
            "assistant_text": assistant_text,
            "user_vec": user_vec,
            "query_vec": query_vec,
            "fragmented": fragmented,
        }
        self._record_turn(turn, meta)
        return meta


class MemoryHarness:
    def __init__(self, lua_root: Path, workdir: Path, args: argparse.Namespace) -> None:
        self.lua_root = lua_root
        self.workdir = workdir
        self.args = args
        self.bridge = MoriMemoryBridge(lua_root=lua_root)
        self.lua = self.bridge.lua
        self._shutdown = False
        os.chdir(workdir)
        Path("memory/v4/topic_graph").mkdir(parents=True, exist_ok=True)
        self._configure()
        self._fp_memory_count = self.lua.eval(
            "function(anchor) "
            "local topic = require('module.memory.topic'); "
            "local fp = topic.get_topic_fingerprint(anchor) or {}; "
            "return tonumber(fp.memory_count) or 0 "
            "end"
        )
        self._memory_stats = self.lua.eval(
            "function() "
            "local tg = require('module.memory.topic_graph'); "
            "local total = 0; "
            "local cross_scope_origin = 0; "
            "local scope_counts = {}; "
            "for _, mem in pairs(tg.state.memories or {}) do "
            "  if mem then "
            "    total = total + 1; "
            "    local scope = tostring(mem.scope_key or ''); "
            "    scope_counts[scope] = (tonumber(scope_counts[scope]) or 0) + 1; "
            "    local origin_scope = nil; "
            "    local mixed = false; "
            "    for anchor in pairs(mem.origin_topics or {}) do "
            "      local scoped = tostring(anchor):match('^(.-)|[ASC]:%d+$') or ''; "
            "      if origin_scope == nil then origin_scope = scoped end "
            "      if scoped ~= origin_scope then mixed = true end "
            "    end "
            "    if mixed then cross_scope_origin = cross_scope_origin + 1 end "
            "  end "
            "end "
            "local parts = {}; "
            "for scope, count in pairs(scope_counts) do "
            "  parts[#parts + 1] = tostring(scope) .. '=' .. tostring(count) "
            "end "
            "table.sort(parts) "
            "return total, cross_scope_origin, table.concat(parts, ';') "
            "end"
        )

    def _configure(self) -> None:
        self.lua.execute(
            f"""
            local config = require("module.config")
            config.reset()
            config.settings.topic.allow_llm_summary = false
            config.settings.topic_graph.topic_hnsw.enabled = false
            config.settings.guard.enabled = true
            config.settings.guard.default_credit_by_source.bilibili = 1.0
            config.settings.guard.credit_decay = 1.0
            config.settings.guard.credit_bonus = 0.0
            config.settings.guard.credit_penalty = 0.0
            config.settings.guard.block_threshold = -1.0
            config.settings.guard.restore_threshold = 0.0
            config.settings.guard.allow_recall_threshold = 0.0
            config.settings.guard.allow_history_threshold = 0.0
            config.settings.guard.allow_topic_threshold = 0.0
            config.settings.guard.allow_memory_write_threshold = 0.0
            config.settings.guard.scope_strategy = "source_room"
            config.settings.guard.anchor_scope_prefix = true
            config.settings.disentangle.enabled = true
            config.settings.disentangle.enable_sources = {{ "bilibili" }}
            config.settings.disentangle.max_streams = {int(self.args.max_streams)}
            config.settings.disentangle.assign_threshold = {float(self.args.assign_threshold)}
            config.settings.disentangle.reset_threshold = {float(self.args.reset_threshold)}
            config.settings.disentangle.commit_idle_turns = {int(self.args.commit_idle_turns)}
            config.settings.disentangle.pending_context_turns = {int(self.args.pending_context_turns)}
            config.settings.disentangle.stale_turns = {int(self.args.stale_turns)}
            """
        )

    def compile_context(self, meta: dict[str, Any]) -> list[str]:
        lua_meta = to_lua_table(self.lua, meta)
        blocks = self.bridge.compile_context(lua_meta)
        return flatten_block_contents(blocks)

    def ingest_turn(self, meta: dict[str, Any]) -> dict[str, Any]:
        lua_meta = to_lua_table(self.lua, meta)
        result = self.bridge.ingest_turn(lua_meta)
        disentangle = result["disentangle"]
        return {
            "turn": int(result["turn"] or 0),
            "topic_anchor": str(result["topic_anchor"] or ""),
            "scope_key": str(result["scope_key"] or ""),
            "credit": float(result["credit"] or 0.0),
            "disentangle": {
                "reason": str(disentangle["reason"] or "") if disentangle else "",
                "is_new": bool(disentangle["is_new"]) if disentangle else False,
                "merged": bool(disentangle["merged"]) if disentangle else False,
                "dropped": bool(disentangle["dropped"]) if disentangle else False,
                "use_local_sequence": bool(disentangle["use_local_sequence"]) if disentangle else False,
                "sequence_key": str(disentangle["sequence_key"] or "") if disentangle else "",
                "mode": str(disentangle["mode"] or "") if disentangle else "",
            },
        }

    def fingerprint_memory_count(self, anchor: str) -> int:
        if not anchor:
            return 0
        return int(self._fp_memory_count(anchor) or 0)

    def memory_stats(self) -> dict[str, Any]:
        total, cross_scope_origin, scope_blob = self._memory_stats()
        scope_counts: dict[str, int] = {}
        blob = str(scope_blob or "")
        if blob:
            for part in blob.split(";"):
                if "=" not in part:
                    continue
                scope, count = part.split("=", 1)
                scope_counts[scope] = int(count)
        return {
            "memory_total": int(total or 0),
            "cross_scope_origin_count": int(cross_scope_origin or 0),
            "scope_counts": scope_counts,
        }

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self.bridge.shutdown()
        self._shutdown = True


def evaluate_query(
    turn_meta: dict[str, Any],
    retrieved_turns: list[int],
    scenario: LiveScenario,
) -> dict[str, Any]:
    relevant_turns = set(int(t) for t in turn_meta["expected_turns"])
    retrieved = [t for t in retrieved_turns if t in scenario.turn_meta]
    hit = bool(relevant_turns.intersection(retrieved))
    empty = len(retrieved) == 0
    relevant_hits = sum(1 for t in retrieved if t in relevant_turns)
    precision = float(relevant_hits) / float(len(retrieved)) if retrieved else 0.0

    room_id = turn_meta["room_id"]
    user_id = turn_meta["user_id"]
    topic_id = turn_meta["topic_id"]
    cross_room = False
    cross_user_personal = False
    for retrieved_turn in retrieved:
        meta = scenario.turn_meta[retrieved_turn]
        if meta["room_id"] != room_id:
            cross_room = True
        if (
            turn_meta["eval_scope"] == "personal"
            and meta["room_id"] == room_id
            and meta["topic_id"] == topic_id
            and meta["user_id"] != user_id
            and meta["mode"].startswith("personal")
        ):
            cross_user_personal = True

    return {
        "hit": hit,
        "empty": empty,
        "precision": precision,
        "cross_room": cross_room,
        "cross_user_personal": cross_user_personal,
    }


def run_single(seed: int, args: argparse.Namespace, workdir: Path) -> dict[str, Any]:
    scenario = LiveScenario(
        seed=seed,
        dim=args.dim,
        turns=args.turns,
        rooms=args.rooms,
        users_per_room=args.users_per_room,
        topics=args.topics,
    )
    harness = MemoryHarness(Path(__file__).resolve().parent, workdir, args)

    compile_ms: list[float] = []
    ingest_ms: list[float] = []
    anchors: list[str] = []
    local_sequence_anchors: list[str] = []
    query_records: list[dict[str, Any]] = []
    failure_examples: list[dict[str, Any]] = []
    disentangle_stats = {
        "dropped": 0,
        "reset_topic": 0,
        "merged": 0,
        "is_new": 0,
    }

    try:
        for turn in range(1, args.turns + 1):
            spec = scenario.make_turn(turn)
            request_meta = {
                "turn": turn,
                "source": "bilibili",
                "room_id": spec["room_id"],
                "user_id": spec["user_id"],
                "nickname": spec["nickname"],
                "user_input": spec["user_input"],
                "raw_user_input": spec["user_input"],
                "query_vec": spec["query_vec"],
                "user_vec": spec["user_vec"],
                "max_selected_turns": args.max_selected_turns,
                "pending_context_turns": args.pending_context_turns,
                "user_chars": 140,
                "assistant_chars": 180,
            }

            start = time.perf_counter()
            blocks = harness.compile_context(request_meta)
            compile_ms.append((time.perf_counter() - start) * 1000.0)

            retrieved_turns = parse_turns_from_blocks(blocks)
            retrieved_fact_ids = parse_fact_ids_from_blocks(blocks)
            for fact_id in retrieved_fact_ids:
                known_turn = scenario.turn_for_fact_id.get(fact_id)
                if known_turn is not None and known_turn not in retrieved_turns:
                    retrieved_turns.append(known_turn)

            if spec["eval_scope"] != "none":
                record = evaluate_query(spec, retrieved_turns, scenario)
                record.update(
                    {
                        "turn": turn,
                        "mode": spec["mode"],
                        "room_id": spec["room_id"],
                        "user_id": spec["user_id"],
                        "topic_id": spec["topic_id"],
                        "expected_turns": spec["expected_turns"],
                        "retrieved_turns": sorted(retrieved_turns),
                        "fragmented": bool(spec["fragmented"]),
                        "blocks_preview": blocks[:3],
                    }
                )
                query_records.append(record)
                if (not record["hit"] or record["cross_room"] or record["cross_user_personal"]) and len(
                    failure_examples
                ) < 8:
                    failure_examples.append(record)

            ingest_meta = dict(request_meta)
            ingest_meta["assistant_text"] = spec["assistant_text"]

            start = time.perf_counter()
            ingest_result = harness.ingest_turn(ingest_meta)
            ingest_ms.append((time.perf_counter() - start) * 1000.0)

            anchor = ingest_result["topic_anchor"]
            if anchor:
                anchors.append(anchor)
                if ingest_result["disentangle"]["use_local_sequence"]:
                    local_sequence_anchors.append(anchor)
            reason = ingest_result["disentangle"]["reason"]
            if ingest_result["disentangle"]["dropped"]:
                disentangle_stats["dropped"] += 1
            if reason == "reset_topic":
                disentangle_stats["reset_topic"] += 1
            if ingest_result["disentangle"]["merged"]:
                disentangle_stats["merged"] += 1
            if ingest_result["disentangle"]["is_new"]:
                disentangle_stats["is_new"] += 1

        harness.shutdown()

        unique_anchors = sorted(set(a for a in anchors if a))
        unique_local_anchors = sorted(set(a for a in local_sequence_anchors if a))
        nonempty_fp = sum(1 for anchor in unique_anchors if harness.fingerprint_memory_count(anchor) > 0)
        nonempty_local_fp = sum(
            1 for anchor in unique_local_anchors if harness.fingerprint_memory_count(anchor) > 0
        )

        shared_queries = [r for r in query_records if r["mode"] == "shared_query"]
        personal_queries = [r for r in query_records if r["mode"] == "personal_query"]
        fragmented_queries = [r for r in query_records if r["fragmented"]]

        def hit_rate(records: list[dict[str, Any]]) -> float:
            return float(sum(1 for item in records if item["hit"])) / float(len(records) or 1)

        def rate(records: list[dict[str, Any]], field: str) -> float:
            return float(sum(1 for item in records if item[field])) / float(len(records) or 1)

        memory_stats = harness.memory_stats()
        summary = {
            "seed": seed,
            "turns": args.turns,
            "queries": len(query_records),
            "shared_queries": len(shared_queries),
            "personal_queries": len(personal_queries),
            "fragmented_queries": len(fragmented_queries),
            "hit_rate": round(hit_rate(query_records), 4),
            "shared_hit_rate": round(hit_rate(shared_queries), 4),
            "personal_hit_rate": round(hit_rate(personal_queries), 4),
            "fragmented_hit_rate": round(hit_rate(fragmented_queries), 4),
            "empty_context_rate": round(rate(query_records, "empty"), 4),
            "cross_room_intrusion_rate": round(rate(query_records, "cross_room"), 4),
            "cross_user_personal_intrusion_rate": round(rate(personal_queries, "cross_user_personal"), 4),
            "avg_query_precision": round(mean_or_zero([r["precision"] for r in query_records]), 4),
            "compile_timing": summarize_ms(compile_ms),
            "ingest_timing": summarize_ms(ingest_ms),
            "disentangle": {
                "drop_rate": round(disentangle_stats["dropped"] / float(args.turns or 1), 4),
                "reset_rate": round(disentangle_stats["reset_topic"] / float(args.turns or 1), 4),
                "merge_rate": round(disentangle_stats["merged"] / float(args.turns or 1), 4),
                "new_stream_rate": round(disentangle_stats["is_new"] / float(args.turns or 1), 4),
            },
            "fingerprint_nonempty_rate": round(nonempty_fp / float(len(unique_anchors) or 1), 4),
            "local_sequence_fingerprint_nonempty_rate": round(
                nonempty_local_fp / float(len(unique_local_anchors) or 1), 4
            ),
            "anchors": len(unique_anchors),
            "local_sequence_anchors": len(unique_local_anchors),
        }
        summary.update(memory_stats)
        return {
            "summary": summary,
            "failure_examples": failure_examples,
        }
    finally:
        try:
            harness.shutdown()
        except Exception:
            pass


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    summary_fields = [
        "hit_rate",
        "shared_hit_rate",
        "personal_hit_rate",
        "fragmented_hit_rate",
        "empty_context_rate",
        "cross_room_intrusion_rate",
        "cross_user_personal_intrusion_rate",
        "avg_query_precision",
        "fingerprint_nonempty_rate",
        "local_sequence_fingerprint_nonempty_rate",
        "memory_total",
        "cross_scope_origin_count",
    ]
    out: dict[str, Any] = {"runs": len(runs)}
    for field in summary_fields:
        values = [float(run["summary"].get(field, 0.0)) for run in runs]
        out[field] = round(mean_or_zero(values), 4)

    compile_p95 = [float(run["summary"]["compile_timing"]["p95_ms"]) for run in runs]
    ingest_p95 = [float(run["summary"]["ingest_timing"]["p95_ms"]) for run in runs]
    out["compile_p95_ms"] = round(mean_or_zero(compile_p95), 3)
    out["ingest_p95_ms"] = round(mean_or_zero(ingest_p95), 3)
    return out


def make_workdir(root: Path | None, seed: int) -> tuple[Path, callable]:
    if root is None:
        tempdir = Path(tempfile.mkdtemp(prefix=f"mori-memory-sim-{seed}-"))
        return tempdir, lambda: shutil.rmtree(tempdir, ignore_errors=True)
    target = root / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    return target, lambda: None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bridge-backed multi-user live memory simulator adapted from the old single-user simulator."
    )
    parser.add_argument("--turns", type=int, default=240, help="Turns per seed")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--repeats", type=int, default=3, help="How many seeds to run")
    parser.add_argument("--rooms", type=int, default=3, help="Number of live rooms")
    parser.add_argument("--users-per-room", type=int, default=4, help="Users per room")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument(
        "--topics",
        type=str,
        default=",".join(DEFAULT_TOPICS),
        help="Comma-separated topic ids",
    )
    parser.add_argument("--max-selected-turns", type=int, default=6)
    parser.add_argument("--max-streams", type=int, default=4)
    parser.add_argument("--assign-threshold", type=float, default=0.80)
    parser.add_argument("--reset-threshold", type=float, default=0.62)
    parser.add_argument("--commit-idle-turns", type=int, default=2)
    parser.add_argument("--pending-context-turns", type=int, default=2)
    parser.add_argument("--stale-turns", type=int, default=80)
    parser.add_argument("--workdir", type=str, default="", help="Keep run artifacts under this directory")
    parser.add_argument("--output", type=str, default="sim_results_live.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.topics = [topic.strip() for topic in str(args.topics).split(",") if topic.strip()]
    unknown_topics = [topic for topic in args.topics if topic not in TOPIC_SPECS]
    if unknown_topics:
        raise SystemExit(f"Unknown topics: {', '.join(unknown_topics)}")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root_workdir = Path(args.workdir).resolve() if args.workdir else None

    runs: list[dict[str, Any]] = []
    cleanup_callbacks: list[callable] = []
    try:
        for idx in range(args.repeats):
            seed = int(args.seed) + idx
            workdir, cleanup = make_workdir(root_workdir, seed)
            cleanup_callbacks.append(cleanup)
            result = run_single(seed, args, workdir)
            result["summary"]["workdir"] = str(workdir)
            runs.append(result)
            summary = result["summary"]
            print(
                "seed={seed} hit={hit:.4f} shared={shared:.4f} personal={personal:.4f} "
                "frag={frag:.4f} cross_room={cross:.4f} cross_user={cross_user:.4f} "
                "compile_p95={compile_p95:.3f}ms ingest_p95={ingest_p95:.3f}ms memories={mem}".format(
                    seed=seed,
                    hit=summary["hit_rate"],
                    shared=summary["shared_hit_rate"],
                    personal=summary["personal_hit_rate"],
                    frag=summary["fragmented_hit_rate"],
                    cross=summary["cross_room_intrusion_rate"],
                    cross_user=summary["cross_user_personal_intrusion_rate"],
                    compile_p95=summary["compile_timing"]["p95_ms"],
                    ingest_p95=summary["ingest_timing"]["p95_ms"],
                    mem=summary["memory_total"],
                )
            )

        payload = {
            "config": {
                "turns": args.turns,
                "seed": args.seed,
                "repeats": args.repeats,
                "rooms": args.rooms,
                "users_per_room": args.users_per_room,
                "dim": args.dim,
                "topics": args.topics,
                "max_streams": args.max_streams,
                "assign_threshold": args.assign_threshold,
                "reset_threshold": args.reset_threshold,
                "commit_idle_turns": args.commit_idle_turns,
                "pending_context_turns": args.pending_context_turns,
                "stale_turns": args.stale_turns,
            },
            "aggregate": aggregate_runs(runs),
            "runs": runs,
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

        agg = payload["aggregate"]
        print(
            "aggregate hit={hit:.4f} shared={shared:.4f} personal={personal:.4f} "
            "frag={frag:.4f} cross_room={cross:.4f} cross_user={cross_user:.4f} "
            "compile_p95={compile_p95:.3f}ms ingest_p95={ingest_p95:.3f}ms output={output}".format(
                hit=agg.get("hit_rate", 0.0),
                shared=agg.get("shared_hit_rate", 0.0),
                personal=agg.get("personal_hit_rate", 0.0),
                frag=agg.get("fragmented_hit_rate", 0.0),
                cross=agg.get("cross_room_intrusion_rate", 0.0),
                cross_user=agg.get("cross_user_personal_intrusion_rate", 0.0),
                compile_p95=agg.get("compile_p95_ms", 0.0),
                ingest_p95=agg.get("ingest_p95_ms", 0.0),
                output=str(output_path),
            )
        )
        return 0
    finally:
        if root_workdir is None:
            for cleanup in cleanup_callbacks:
                cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
