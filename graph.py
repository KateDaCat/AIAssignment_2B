# graph.py  (Assignment 2B version)

def load_graph(filename):
    """
    Reads the updated test file and returns:
      - graph: {node: [(neighbor, cost), ...]}
      - origin: int
      - destinations: list[int]
      - coords: {node: (x,y)}
      - accident: { "edge": (u,v), "severity": str, "multiplier": float }
    """

    with open(filename, "r") as file:
        lines = file.readlines()

    graph = {}
    origin = None
    destinations = []
    coords = {}
    accident = {"edge": None, "severity": None, "multiplier": 1.0}

    section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.endswith(":"):
            section = line[:-1]
            continue

        # --------------------
        # Parse Nodes
        # --------------------
        if section == "Nodes":
            # Format:  1: (0,0)
            node_str, xy_str = line.split(":")
            node_id = int(node_str.strip())

            xy_str = xy_str.strip().strip("()")
            x, y = map(float, xy_str.split(","))
            coords[node_id] = (x, y)

            if node_id not in graph:
                graph[node_id] = []

        # --------------------
        # Parse Directed Edges
        # --------------------
        elif section == "Edges":
            # Format: (u,v): value
            pair, val = line.split(":")
            u, v = map(int, pair.strip().strip("()").split(","))
            cost = float(val.strip())

            if u not in graph:
                graph[u] = []

            # directed edge (u → v)
            graph[u].append((v, cost))

        # --------------------
        # Parse Origin/Destination
        # --------------------
        elif section == "Origin":
            origin = int(line)

        elif section == "Destinations":
            destinations = [int(x.strip()) for x in line.split(";")]

        # --------------------
        # Parse accident update
        # --------------------
        elif section == "Accident Update":
            if line.startswith("ACCIDENT_EDGE"):
                edge_str = line.split(":")[1].strip()
                u, v = map(int, edge_str.strip("()").split(","))
                accident["edge"] = (u, v)

            elif line.startswith("SEVERITY"):
                accident["severity"] = line.split(":")[1].strip()

            elif line.startswith("MULTIPLIER"):
                accident["multiplier"] = float(line.split(":")[1].strip())

    # --------------------
    # Apply accident multiplier
    # --------------------
    if accident["edge"]:
        u, v = accident["edge"]

        if u in graph:
            new_list = []
            for (nbr, cost) in graph[u]:
                if nbr == v:
                    cost = cost * accident["multiplier"]
                new_list.append((nbr, cost))
            graph[u] = new_list

    return graph, origin, destinations, coords, accident
