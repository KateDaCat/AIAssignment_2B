# algorithms/cus1_ucs.py
import heapq
from itertools import count

def cus1_ucs(graph, origin, destinations):
    """
    Uniform Cost Search (CUS1)
    Returns: (path:list[int] or None, total_cost:int, nodes_created:int)
    """
    # frontier items: (g_cost, node_id, tiebreak, node, parent)
    # order by g_cost, then node_id, then insertion order (tiebreak)
    tiebreaker = count()
    frontier = []
    heapq.heappush(frontier, (0, origin, next(tiebreaker), origin, None))

    best_cost = {}         # node -> best g seen
    parent = {}            # node -> parent node
    nodes_created = 1      # count nodes you push (or pop)—choose one & keep consistent across all methods (document in report). :contentReference[oaicite:3]{index=3}

    while frontier:
        g, _, _, u, par = heapq.heappop(frontier)

        if u in best_cost and g > best_cost[u]:
            continue

        parent[u] = par
        best_cost[u] = g

        if u in destinations:
            # reconstruct path
            path = []
            cur = u
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return path, g, nodes_created  # keep node-creation metric consistent across algorithms

        # expand neighbors in ascending node-id to satisfy tie-break (then insertion order)
        for v, w in sorted(graph.get(u, []), key=lambda x: x[0]):
            new_g = g + w
            if v not in best_cost or new_g < best_cost[v]:
                heapq.heappush(frontier, (new_g, v, next(tiebreaker), v, u))
                nodes_created += 1

    return None, float("inf"), nodes_created
