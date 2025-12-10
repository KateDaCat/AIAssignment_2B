# algorithms/cus2_hcs.py
import math  # for Euclidean distance

def cus2_hcs(graph, origin, destinations, coords):
    """
    Hill-Climbing Search (CUS2) - Informed Search
    Uses only the heuristic h(n) and always moves to the best neighbor.
    Stops if no neighbor is strictly better (local maximum or plateau).
    Avoids cycles on the same branch using repeated-state check.
    Returns (path, total_cost, nodes_created)
    """

    goal_set = set(destinations)            # convert list to set for faster checks

    # heuristic: straight-line distance to the nearest goal
    def heuristic(node):
        x1, y1 = coords[node]               # get (x, y) for current node
        best = float("inf")                 # start with a large number
        for g in goal_set:                  # check each goal node
            x2, y2 = coords[g]              # get (x, y) of goal node
            d = math.hypot(x1 - x2, y1 - y2)  # Euclidean distance
            if d < best:                    # keep the smallest distance
                best = d
        return best                         # return nearest goal distance

    current = origin                         # start at origin
    path = [origin]                          # path starts with origin
    total_cost = 0                           # path cost so far
    nodes_created = 0                        # count of "generated" neighbors we examined

    # if start is already a goal, return immediately
    if current in goal_set:
        return path, total_cost, nodes_created

    # loop until we reach a goal or get stuck
    while True:
        current_h = heuristic(current)       # heuristic value of current node

        # if no outgoing edges, we are stuck
        if current not in graph:
            return [], 0, nodes_created      # no path to any goal

        # pick the best neighbor based on the lowest heuristic value
        best_neighbor = None                 # store the chosen neighbor
        best_h = float("inf")                # best heuristic seen so far
        best_step_cost = None                # track step cost for the chosen neighbor

        # sort neighbors by node id to keep tie-breaking stable
        for (neighbor, step_cost) in sorted(graph[current], key=lambda x: x[0]):

            # avoid cycles on the same branch
            if neighbor in path:
                continue                     # skip if neighbor already in current path

            # evaluate this neighbor
            h_value = heuristic(neighbor)    # compute heuristic for neighbor
            nodes_created += 1               # count this evaluation as a created node

            # choose neighbor with strictly smaller h than current (greedy improvement)
            if h_value < best_h:             # better than current best candidate
                best_h = h_value             # update best heuristic
                best_neighbor = neighbor     # update best neighbor
                best_step_cost = step_cost   # remember edge cost to add later

        # if no valid neighbor found, we are stuck
        if best_neighbor is None:
            return [], 0, nodes_created      # failure to find any improving move

        # if best neighbor is NOT strictly better than current, stop (local maximum/plateau)
        if not (best_h < current_h):
            return [], 0, nodes_created      # no improvement possible

        # move to the chosen neighbor
        path.append(best_neighbor)           # extend path
        total_cost += best_step_cost         # add edge cost to total
        current = best_neighbor              # update current node

        # check goal after the move
        if current in goal_set:
            return path, total_cost, nodes_created
