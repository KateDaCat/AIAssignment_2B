# algorithms/astar.py
import math        # for Euclidean distance calculation
import heapq       # for priority queue (min-heap)

def astar_search(graph, origin, destinations, coords):
    """
    A* Search (Informed Search)
    - Uses f(n) = g(n) + h(n)
    - Avoids cycles on the same branch (repeated-state check)
    - Returns (path, total_cost, nodes_created)
    """

    # Convert list of destinations into a set for faster goal checking
    goal_set = set(destinations)

    # Heuristic function: calculates straight-line distance to nearest goal
    def heuristic(node):
        x1, y1 = coords[node]            # get (x, y) of current node
        best = float("inf")              # start with large value
        for g in goal_set:               # check each goal node
            x2, y2 = coords[g]           # get (x, y) of goal node
            d = math.hypot(x1 - x2, y1 - y2)  # calculate Euclidean distance
            if d < best:                 # keep the smallest distance
                best = d
        return best                      # return nearest goal distance

    # Priority queue will store (f_value, node_id, order, current_node, path, g_value)
    frontier = []                        # create an empty priority queue
    order = 0                            # tie-breaker order counter
    g_start = 0                          # cost from start to start = 0
    f_start = g_start + heuristic(origin)  # f = g + h
    heapq.heappush(frontier, (f_start, origin, order, origin, [origin], g_start))  # push start node
    nodes_created = 1                    # count number of nodes added to frontier

    # Main loop: run until no more nodes left to explore
    while len(frontier) > 0:
        # pop the node with the smallest f-value (best estimated total cost)
        _, _, _, current, path_taken, g_value = heapq.heappop(frontier)

        # Goal test: check if current node is one of the destination nodes
        if current in goal_set:
            return path_taken, g_value, nodes_created  # goal found, return results

        # Check if current node has neighbors in the graph
        if current in graph:
            # sort neighbors by node id to ensure consistent tie-breaking order
            for (neighbor, step_cost) in sorted(graph[current], key=lambda x: x[0]):

                # Repeated-state check: skip if neighbor already in current path
                if neighbor in path_taken:
                    continue              # prevents cycles like A -> B -> A -> C

                # Calculate new path cost (g) and total estimated cost (f)
                new_g = g_value + step_cost          # cost so far + edge cost
                new_f = new_g + heuristic(neighbor)   # total estimated cost

                # Create a new path list with this neighbor added 
                new_path = path_taken + [neighbor]

                # Increase order count for tie-breaking
                order += 1

                # Add the neighbor to the frontier with updated values
                heapq.heappush(frontier, (new_f, neighbor, order, neighbor, new_path, new_g))
                nodes_created += 1                   # increase created nodes count
        else:
            # if no outgoing edges, just skip this node
            continue

    # If frontier is empty and no goal was found
    return [], 0, nodes_created
