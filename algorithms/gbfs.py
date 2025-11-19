# algorithms/gbfs.py
import math      # for Euclidean distance calculation
import heapq     # for priority queue (min-heap)

def gbfs_search(graph, origin, destinations, coords):
    """
    Greedy Best-First Search (Informed Search)
    Uses f(n) = h(n) only (ignores path cost g)
    Avoids cycles using same-branch repeated-state check
    Returns (path, total_cost, nodes_created)
    """

    goal_set = set(destinations)  # convert list to set for faster checking

    # heuristic function to find straight-line distance to nearest goal
    def heuristic(node):
        x1, y1 = coords[node]      # get coordinates of current node
        best = float("inf")        # start with large number
        for g in goal_set:         # check each goal node
            x2, y2 = coords[g]     # get coordinates of goal
            d = math.hypot(x1 - x2, y1 - y2)  # Euclidean distance formula
            if d < best:           # keep the smallest distance
                best = d
        return best                # return nearest goal distance

    frontier = []                  # create an empty priority queue
    order = 0                      # tie-breaker counter
    g_start = 0                    # path cost from start is 0
    h_start = heuristic(origin)    # heuristic value for start node
    heapq.heappush(frontier, (h_start, origin, order, origin, [origin], g_start))  # add start node to queue
    nodes_created = 1              # count how many nodes are created

    # main loop runs while there are nodes left to explore
    while len(frontier) > 0:
        # get node with the smallest heuristic value
        _, _, _, current, path_taken, g_value = heapq.heappop(frontier)

        # check if current node is one of the destination nodes
        if current in goal_set:
            return path_taken, int(g_value), nodes_created

        # check if current node has outgoing edges
        if current in graph:
            # sort neighbors by node id for consistent expansion order
            for (neighbor, step_cost) in sorted(graph[current], key=lambda x: x[0]):

                # avoid cycles by skipping nodes already in the current path
                if neighbor in path_taken:
                    continue

                h_value = heuristic(neighbor)        # calculate heuristic for neighbor
                new_g = g_value + step_cost          # update path cost (not used for priority)
                new_path = path_taken + [neighbor]   # extend the current path
                order += 1                           # increase order for tie-breaking

                # push neighbor into frontier with priority = heuristic value
                heapq.heappush(frontier, (h_value, neighbor, order, neighbor, new_path, new_g))
                nodes_created += 1                   # increase nodes created count
        else:
            # node has no neighbors, skip
            continue

    # no path found after exploring all nodes
    return [], 0, nodes_created
