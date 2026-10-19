"""
gnn_chemistry.py (v2 — dataset-agnostic)
Same GraphSAGE-style aggregation + KMeans chemistry engine as v1, but
operating on plain dict records (from feature_engine.py) instead of a fixed
dataclass, so it works identically whether the players came from the
synthetic generator or an uploaded real dataset.
"""
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def feature_vector(p: dict) -> np.ndarray:
    return np.array([
        *p["playstyle_vector"],
        p["tilt_score"],
        p["fatigue_index"],
        p["mmr"] / 3000.0,
    ])


def build_similarity_graph(players: list[dict], k: int = 5) -> nx.Graph:
    X = np.array([feature_vector(p) for p in players])
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(players))).fit(X)
    _, indices = nbrs.kneighbors(X)
    G = nx.Graph()
    for i, p in enumerate(players):
        G.add_node(i, player=p)
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            G.add_edge(i, int(j))
    return G


def embed_players(players: list[dict], hops: int = 2) -> np.ndarray:
    G = build_similarity_graph(players)
    X = np.array([feature_vector(p) for p in players])
    H = X.copy()
    for _ in range(hops):
        H_new = np.zeros_like(H)
        for i in G.nodes:
            neighbors = list(G.neighbors(i))
            neighbor_mean = H[neighbors].mean(axis=0) if neighbors else H[i]
            H_new[i] = 0.5 * H[i] + 0.5 * neighbor_mean
        H = H_new
    return H


def cluster_chemistry(players: list[dict], n_clusters: int = 5):
    embeddings = embed_players(players)
    n_clusters = min(n_clusters, len(players))
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(embeddings)
    return labels.tolist(), km.cluster_centers_, embeddings


def chemistry_score(team: list[dict]) -> float:
    if len(team) < 2:
        return 1.0
    vectors = np.array([p["playstyle_vector"] for p in team])
    role_diversity = np.mean(np.std(vectors, axis=0))
    tilt_count = sum(1 for p in team if p["tilt_score"] > 0.6)
    tilt_penalty = max(0, tilt_count - 1) * 0.15
    mmrs = [p["mmr"] for p in team]
    mmr_spread_penalty = (max(mmrs) - min(mmrs)) / 3000.0
    score = 0.5 + role_diversity - tilt_penalty - mmr_spread_penalty * 0.3
    return round(max(0.0, min(1.0, score)), 3)


def assemble_slate(players: list[dict], slate_size: int = 10) -> dict:
    if len(players) < slate_size:
        slate_size = max(2, len(players) - (len(players) % 2))
    pool = sorted(players[:slate_size], key=lambda p: p["mmr"], reverse=True)

    team_a, team_b = [], []
    for i, p in enumerate(pool):
        if (i // 2) % 2 == 0:
            (team_a if i % 2 == 0 else team_b).append(p)
        else:
            (team_b if i % 2 == 0 else team_a).append(p)

    best_a, best_b = team_a[:], team_b[:]
    best_score = chemistry_score(best_a) + chemistry_score(best_b)
    for i in range(len(team_a)):
        for j in range(len(team_b)):
            trial_a, trial_b = team_a[:], team_b[:]
            trial_a[i], trial_b[j] = trial_b[j], trial_a[i]
            gap = abs(sum(p["mmr"] for p in trial_a) - sum(p["mmr"] for p in trial_b))
            if gap > 400:
                continue
            score = chemistry_score(trial_a) + chemistry_score(trial_b)
            if score > best_score:
                best_score = score
                best_a, best_b = trial_a[:], trial_b[:]

    mmr_gap = abs(sum(p["mmr"] for p in best_a) - sum(p["mmr"] for p in best_b))
    return {
        "team_a": best_a, "team_b": best_b,
        "team_a_chemistry": chemistry_score(best_a),
        "team_b_chemistry": chemistry_score(best_b),
        "mmr_gap": mmr_gap,
        "predicted_close_match": 1 - min(1.0, mmr_gap / 1000.0),
    }
