"""
cluster_labeler.py
"CRAZY FEATURE #2" — Auto-Labeled Chemistry Clusters.

Raw KMeans output is just cluster index 0..4 — meaningless to a judge or a
professor. This module inspects each cluster's centroid (in the original,
human-interpretable feature space: aggression, positioning, comms, support,
objective focus, tilt, fatigue, mmr) and generates a real label + one-line
description, e.g. "Volatile Duelists — high aggression, elevated tilt" or
"Steady Anchors — objective-focused, low tilt, tight MMR band".

This turns "5 unexplained colored dots" into "5 named psychological/tactical
player profiles" — the single biggest interpretability upgrade for a demo.
"""
import numpy as np

DIM_NAMES = ["aggression", "positioning", "comms", "support", "objective_focus",
             "tilt", "fatigue", "mmr_norm"]

# vocabulary banks used to compose a label from whichever dimensions stand out
HIGH_TRAIT_WORDS = {
    "aggression": "Duelist",
    "positioning": "Tactician",
    "comms": "Shotcaller",
    "support": "Anchor",
    "objective_focus": "Objective-Runner",
    "tilt": "Volatile",
    "fatigue": "Grinding",
    "mmr_norm": "Elite",
}
LOW_TRAIT_WORDS = {
    "tilt": "Steady",
    "fatigue": "Fresh",
    "aggression": "Passive",
    "mmr_norm": "Developing",
}


def label_clusters(players: list[dict], labels: list[int], centers: np.ndarray) -> dict:
    """Returns {cluster_id: {"label": str, "description": str, "size": int,
    "avg_tilt": float, "avg_mmr": float}}"""
    result = {}
    n_clusters = centers.shape[0]

    # global z-score normalization so "high"/"low" is relative to this queue,
    # not to arbitrary absolute thresholds
    all_vals = centers
    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0) + 1e-6

    for c in range(n_clusters):
        z = (centers[c] - mean) / std
        # rank dims by |z-score| to find what's distinctive about this cluster
        order = np.argsort(-np.abs(z))
        top_dims = order[:2]

        words = []
        for d in top_dims:
            dim_name = DIM_NAMES[d]
            if z[d] > 0.35 and dim_name in HIGH_TRAIT_WORDS:
                words.append(HIGH_TRAIT_WORDS[dim_name])
            elif z[d] < -0.35 and dim_name in LOW_TRAIT_WORDS:
                words.append(LOW_TRAIT_WORDS[dim_name])

        if not words:
            words = ["Balanced", "Flex"]
        label = " ".join(dict.fromkeys(words))  # dedupe, keep order

        members = [p for p, l in zip(players, labels) if l == c]
        avg_tilt = float(np.mean([p["tilt_score"] for p in members])) if members else 0.0
        avg_mmr = float(np.mean([p["mmr"] for p in members])) if members else 0.0
        avg_fatigue = float(np.mean([p["fatigue_index"] for p in members])) if members else 0.0

        description = (
            f"{len(members)} players · avg tilt {avg_tilt:.2f} · "
            f"avg fatigue {avg_fatigue:.2f} · avg MMR {avg_mmr:.0f}"
        )

        result[c] = {
            "label": label,
            "description": description,
            "size": len(members),
            "avg_tilt": round(avg_tilt, 3),
            "avg_fatigue": round(avg_fatigue, 3),
            "avg_mmr": round(avg_mmr, 1),
        }
    return result
