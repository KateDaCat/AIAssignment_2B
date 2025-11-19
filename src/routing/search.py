import sys
from map.graph import load_graph

from algorithms.cus1_ucs import cus1_ucs
from algorithms.dfs import dfs_search
from algorithms.bfs import bfs_search
from algorithms.gbfs import gbfs_search
from algorithms.astar import astar_search
from algorithms.cus2_hcs import cus2_hcs

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        return

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    graph, origin, destinations, coords, accident = load_graph(filename)

    if method == "CUS1":
        path, cost, nodes = cus1_ucs(graph, origin, destinations)
    elif method == "DFS":
        path, cost, nodes = dfs_search(graph, origin, destinations)
    elif method == "BFS":
        path, cost, nodes = bfs_search(graph, origin, destinations)
    elif method == "GBFS":
        path, cost, nodes = gbfs_search(graph, origin, destinations, coords)
    elif method == "ASTAR":
        path, cost, nodes = astar_search(graph, origin, destinations, coords)
    elif method == "CUS2":
        path, cost, nodes = cus2_hcs(graph, origin, destinations, coords)
    else:
        print("Unknown method")
        return

    print("=" * 40)
    print(f"File: {filename}")
    print(f"Algorithm: {method}")
    print(f"Origin: {origin}")
    print(f"Destinations: {destinations}")
    print("-" * 40)

    if accident["edge"]:
        print(f"Accident on edge: {accident['edge']}")
        print(f"Severity: {accident['severity']}")
        print(f"Multiplier applied: {accident['multiplier']}")
        print("-" * 40)

    if path:
        print(f"Path found: {' -> '.join(map(str, path))}")
        print(f"Total travel time: {cost:.6f}")
        print(f"Nodes expanded: {nodes}")
    else:
        print("No path found")

    print("=" * 40)

if __name__ == "__main__":
    main()
