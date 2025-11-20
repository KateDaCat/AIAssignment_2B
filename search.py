import argparse
import sys
from typing import List, Tuple

from graph import load_graph
from topk import yen_k_shortest_paths

from algorithms.cus1_ucs import cus1_ucs
from algorithms.dfs import dfs_search
from algorithms.bfs import bfs_search
from algorithms.gbfs import gbfs_search
from algorithms.astar import astar_search
from algorithms.cus2_hcs import cus2_hcs


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run routing algorithms and optionally retrieve the top-k shortest paths."
    )
    parser.add_argument("filename", help="Path to the map/test file.")
    parser.add_argument("method", help="Search method (CUS1, DFS, BFS, GBFS, ASTAR, CUS2).")
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Return up to k shortest paths (currently available for CUS1).",
    )
    parser.add_argument(
        "--target",
        type=int,
        help="Explicit destination node to use when k > 1. Defaults to the first destination listed in the file.",
    )
    parser.add_argument("--origin-id", type=int, help="Override origin node ID defined in the map file.")
    parser.add_argument(
        "--destination-id",
        type=int,
        help="Override destination node list with a single node ID.",
    )
    parser.add_argument("--origin-name", help="Override origin by landmark name (case-insensitive).")
    parser.add_argument("--destination-name", help="Override destination by landmark name (case-insensitive).")
    parser.add_argument(
        "--list-landmarks",
        action="store_true",
        help="Print all known landmarks for the supplied map file and exit.",
    )
    return parser.parse_args(argv)


def node_label(node: int, landmarks: dict) -> str:
    name = landmarks.get(node)
    if name:
        return f"{node} ({name})"
    return str(node)


def format_destination_list(destinations: List[int], landmarks: dict) -> str:
    if not destinations:
        return "[]"
    labels = ", ".join(node_label(node, landmarks) for node in destinations)
    return f"[{labels}]"


def print_run_header(filename: str, method: str, origin: int, destinations: List[int], landmarks: dict) -> None:
    print("=" * 40)
    print(f"File: {filename}")
    print(f"Algorithm: {method}")
    print(f"Origin: {node_label(origin, landmarks)}")
    print(f"Destinations: {format_destination_list(destinations, landmarks)}")
    print("-" * 40)


def print_accident_info(accident: dict) -> None:
    if accident["edge"]:
        print(f"Accident on edge: {accident['edge']}")
        print(f"Severity: {accident['severity']}")
        print(f"Multiplier applied: {accident['multiplier']}")
        print("-" * 40)


def handle_top_k(
    graph,
    origin: int,
    destinations: List[int],
    filename: str,
    accident: dict,
    k: int,
    target: int | None,
    landmarks: dict,
) -> None:
    if not destinations:
        print("No destinations provided in the input file.")
        return

    chosen_target = target
    if chosen_target is None:
        chosen_target = destinations[0]
        if len(destinations) > 1:
            print(f"[info] --target not supplied; defaulting to first destination {chosen_target}.")
    elif chosen_target not in destinations:
        print(f"[info] Destination {chosen_target} not listed in file; proceeding anyway.")

    print_run_header(filename, "CUS1 (k-shortest)", origin, destinations, landmarks)
    print_accident_info(accident)

    paths, nodes_expanded = yen_k_shortest_paths(graph, origin, chosen_target, k)
    if not paths:
        print(f"No path found from {origin} to {chosen_target}")
        print("=" * 40)
        return

    for idx, info in enumerate(paths, start=1):
        path = info["path"]
        cost = info["cost"]
        labels = " -> ".join(node_label(n, landmarks) for n in path)
        print(f"{idx}) Path: {labels}")
        print(f"    Travel time: {cost:.6f}")
    print(f"Nodes expanded across runs: {nodes_expanded}")
    print("=" * 40)


def run_single_path_method(method: str, graph, origin, destinations, coords) -> Tuple[List[int] | None, float, int]:
    if method == "CUS1":
        return cus1_ucs(graph, origin, destinations)
    if method == "DFS":
        return dfs_search(graph, origin, destinations)
    if method == "BFS":
        return bfs_search(graph, origin, destinations)
    if method == "GBFS":
        return gbfs_search(graph, origin, destinations, coords)
    if method == "ASTAR":
        return astar_search(graph, origin, destinations, coords)
    if method == "CUS2":
        return cus2_hcs(graph, origin, destinations, coords)
    raise ValueError(f"Unknown method '{method}'")


def build_name_index(landmarks: dict) -> dict:
    return {name.lower(): node for node, name in landmarks.items()}


def resolve_node_override(
    *,
    id_override: int | None,
    name_override: str | None,
    default_value: int | None,
    landmarks: dict,
    label: str,
) -> int | None:
    if id_override is not None:
        return id_override
    if name_override:
        lookup = build_name_index(landmarks)
        node_id = lookup.get(name_override.strip().lower())
        if node_id is None:
            print(f"[error] Unknown {label} name '{name_override}'. Available names: {', '.join(landmarks.values())}")
            sys.exit(1)
        return node_id
    return default_value


def main():
    args = parse_args(sys.argv[1:])
    method = args.method.upper()

    graph, origin, destinations, coords, accident, landmarks = load_graph(args.filename)

    if args.list_landmarks:
        if not landmarks:
            print("No landmark metadata supplied with this map.")
        else:
            print("Available landmarks:")
            for node_id in sorted(landmarks):
                print(f"- {node_label(node_id, landmarks)}")
        return

    origin = resolve_node_override(
        id_override=args.origin_id,
        name_override=args.origin_name,
        default_value=origin,
        landmarks=landmarks,
        label="origin",
    )

    destination_override = resolve_node_override(
        id_override=args.destination_id,
        name_override=args.destination_name,
        default_value=None,
        landmarks=landmarks,
        label="destination",
    )
    if destination_override is not None:
        destinations = [destination_override]

    if origin is None:
        print("[error] Origin is not defined in the map file and was not provided via CLI.")
        return
    if not destinations:
        print("[error] Destination list is empty. Provide --destination-id or --destination-name.")
        return

    if args.k > 1 and method != "CUS1":
        print("[warn] Top-k paths currently require the CUS1 method. Falling back to k=1.")
        args.k = 1

    if args.k > 1:
        handle_top_k(
            graph=graph,
            origin=origin,
            destinations=destinations,
            filename=args.filename,
            accident=accident,
            k=args.k,
            target=args.target,
            landmarks=landmarks,
        )
        return

    try:
        path, cost, nodes = run_single_path_method(method, graph, origin, destinations, coords)
    except ValueError as exc:
        print(exc)
        return

    print_run_header(args.filename, method, origin, destinations, landmarks)
    print_accident_info(accident)

    if path:
        labels = " -> ".join(node_label(n, landmarks) for n in path)
        print(f"Path found: {labels}")
        print(f"Total travel time: {cost:.6f}")
        print(f"Nodes expanded: {nodes}")
    else:
        print("No path found")

    print("=" * 40)


if __name__ == "__main__":
    main()
