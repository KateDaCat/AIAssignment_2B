# algorithms/bfs.py

from collections import deque

def bfs_search(graph, origin, destinations):
    """
    Breadth-First Search (Uninformed Search)
    """

    # Queue holds tuples: (current_node, path_so_far, cost_so_far)
    queue = deque([(origin, [origin], 0)])
    visited = set()
    nodes_created = 1

    while queue:
        # Pop from the front of the queue (FIFO)
        current, path, cost = queue.popleft()

        # Check goal
        if current in destinations:
            return path, cost, nodes_created

        # Skip already visited
        if current in visited:
            continue
        visited.add(current)

        # Expand neighbors
        if current in graph:
            # Sort neighbors ascending by node id
            for neighbor, edge_cost in sorted(graph[current], key=lambda x: x[0]):
                if neighbor not in visited:
                    nodes_created += 1
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    queue.append((neighbor, new_path, new_cost))

    # No path found
    return None, 0, nodes_created


