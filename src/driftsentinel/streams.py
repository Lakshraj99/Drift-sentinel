"""Deterministic raw stream definitions shared by Phase 1 and recovery replay."""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Iterator
from collections.abc import Mapping

from river.datasets import synth

SEA_DRIFT_INSTANCES = (3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 29500)
SEA_REFERENCE_SIZE = 300
SEA_BATCH_SIZE = 50
SEA_MAX_INSTANCES = 30000
SEA_DRIFT_BATCHES = tuple((point - SEA_REFERENCE_SIZE) // SEA_BATCH_SIZE for point in SEA_DRIFT_INSTANCES)


def make_multi_sea_stream(kind: str, generation: Mapping[str, object] | None = None) -> Iterator[tuple[dict, int]]:
    """Yield alternating SEA concepts with abrupt or gradual recurring changes."""
    if kind not in {"abrupt", "gradual"}:
        raise ValueError("kind must be abrupt or gradual")
    settings = generation or {}
    drift_instances = tuple(int(value) for value in settings.get("event_instances", SEA_DRIFT_INSTANCES))
    variants = tuple(int(value) for value in settings.get("concept_variants", (0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1)))
    concept_seed_base = int(settings.get("concept_seed_base", 700))
    transition_seed = int(settings.get(f"{kind}_transition_seed", 991 if kind == "abrupt" else 992))
    width = int(settings.get("gradual_width_instances", 500))
    max_instances = int(settings.get("max_instances", SEA_MAX_INSTANCES))
    if len(variants) != len(drift_instances) + 1:
        raise ValueError("SEA concept_variants must contain one more item than event_instances")
    streams = [iter(synth.SEA(variant=variant, seed=concept_seed_base + i)) for i, variant in enumerate(variants)]
    rng = random.Random(transition_seed)
    for index in range(max_instances):
        segment = bisect.bisect_right(drift_instances, index)
        selected = segment
        if kind == "gradual":
            transition = min(range(len(drift_instances)), key=lambda j: abs(index - drift_instances[j]))
            onset = drift_instances[transition]
            if abs(index - onset) <= 2 * width:
                probability_next = 1.0 / (1.0 + math.exp(-4.0 * (index - onset) / width))
                selected = transition + 1 if rng.random() < probability_next else transition
        yield next(streams[selected])
