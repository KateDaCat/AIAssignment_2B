"""
Utilities for retrieving the k-shortest loopless paths between two nodes using Yen's algorithm.
"""

from __future__ import annotations

import heapq
from itertools import count
from typing import Dict, List, Sequence, Set, Tuple

Graph = Dict[int, List[Tuple[int, float]]]
Edge = Tuple[int, int]


def _build_edge_lookup(graph: Graph) -> Dict[Edge, float]:
    lookup: Dict[Edge, float] = {}
    for u, neighbors in graph.items():
        for v, cost in neighbors:
            lookup[(u, v)] = cost
    return lookup


def _path_cost(edge_lookup: Dict[Edge, float], path: Sequence[int]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        if edge not in edge_lookup:
            raise ValueError(f"No edge data for {edge}")
        total += edge_lookup[edge]
    return total


def _dijkstra_shortest_path(
    graph: Graph,
    start: int,
    goal: int,
    banned_nodes: Set[int] | None = None,
    banned_edges: Set[Edge] | None = None,
) -> Tuple[List[int] | None, float, int]:
    banned_nodes = banned_nodes or set()
    banned_edges = banned_edges or set()

    frontier: List[Tuple[float, int, List[int]]] = [(0.0, start, [start])]
    best_cost: Dict[int, float] = {}
    nodes_expanded = 0

    while frontier:
        cost, node, path = heapq.heappop(frontier)
        if node in banned_nodes:
            continue
        if cost > best_cost.get(node, float("inf")):
            continue
        best_cost[node] = cost
        nodes_expanded += 1

        if node == goal:
            return path, cost, nodes_expanded

        for neighbor, weight in sorted(graph.get(node, []), key=lambda x: x[0]):
            if neighbor in banned_nodes or (node, neighbor) in banned_edges:
                continue
            new_cost = cost + weight
            if new_cost < best_cost.get(neighbor, float("inf")):
                heapq.heappush(frontier, (new_cost, neighbor, path + [neighbor]))

    return None, float("inf"), nodes_expanded


def yen_k_shortest_paths(graph: Graph, start: int, goal: int, k: int) -> Tuple[List[Dict[str, object]], int]:
    if k <= 0:
        return [], 0
    if start == goal:
        return [{"path": [start], "cost": 0.0}], 1

    edge_lookup = _build_edge_lookup(graph)
    paths: List[Dict[str, object]] = []
    total_expanded = 0

    first_path, first_cost, expanded = _dijkstra_shortest_path(graph, start, goal)
    total_expanded += expanded
    if not first_path:
        return [], total_expanded

    paths.append({"path": first_path, "cost": first_cost})
    candidates: List[Tuple[float, int, List[int]]] = []
    candidate_seen: Set[Tuple[int, ...]] = set()
    path_ids = set([tuple(first_path)])
    tie_counter = count()

    for _ in range(1, k):
        previous_path = paths[-1]["path"]
        assert isinstance(previous_path, list)
        path_len = len(previous_path)

        for i in range(path_len - 1):
            spur_node = previous_path[i]
            root_path = previous_path[: i + 1]

            banned_nodes = set(root_path[:-1])
            banned_edges: Set[Edge] = set()
            for stored in paths:
                stored_path = stored["path"]
                assert isinstance(stored_path, list)
                if len(stored_path) > i and stored_path[: i + 1] == root_path:
                    banned_edges.add((stored_path[i], stored_path[i + 1]))

            spur_path, spur_cost, expanded = _dijkstra_shortest_path(
                graph, spur_node, goal, banned_nodes, banned_edges
            )
            total_expanded += expanded

            if not spur_path:
                continue

            new_path = root_path[:-1] + spur_path
            path_tuple = tuple(new_path)
            if path_tuple in path_ids or path_tuple in candidate_seen:
                continue

            total_cost = _path_cost(edge_lookup, new_path)
            heapq.heappush(candidates, (total_cost, next(tie_counter), new_path))
            candidate_seen.add(path_tuple)

        if not candidates:
            break

        cost, _, best_path = heapq.heappop(candidates)
        paths.append({"path": best_path, "cost": cost})
        path_ids.add(tuple(best_path))

    return paths, total_expanded
