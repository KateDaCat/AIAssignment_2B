# algorithms/dfs.py

def dfs_search(graph, origin, destinations):
    """
    Depth-First Search (Uninformed Search)
    """
    # Stack: (current_node, path, cost)
    stack = [(origin, [origin], 0)]
    visited = set()
    nodes_created = 1
    
    while stack:
        current, path, cost = stack.pop()
        
        # Check if goal
        if current in destinations:
            return path, cost, nodes_created
        
        # Skip if visited
        if current in visited:
            continue
            
        visited.add(current)
        
        # Add neighbors to stack
        if current in graph:
            # Sort neighbors in ascending order
            neighbors = sorted(graph[current], key=lambda x: x[0])
            
            for neighbor, edge_cost in neighbors:
                if neighbor not in visited:
                    nodes_created += 1
                    new_path = path + [neighbor]
                    new_cost = cost + edge_cost
                    stack.append((neighbor, new_path, new_cost))
    
    return None, 0, nodes_created