#!/usr/bin/env python3
"""Utility to generate ICS-formatted Kuching maps from teacher assets.

This script converts the provided heritage dataset and slices of the
OpenStreetMap export into the map format expected by graph.py/search.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
from PIL import Image

try:
    import osmnx as ox
except ImportError as exc:  # pragma: no cover
    raise SystemExit("osmnx is required. Install with `pip install osmnx`.") from exc

try:
    import contextily as cx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("contextily is required. Install with `pip install contextily`.") from exc


Number = float


def minutes_to_hours(value: Number) -> float:
    return float(value) / 60.0


def fmt_float(value: float) -> str:
    return f"{value:.6f}"


def write_ics(
    path: Path,
    *,
    nodes: Dict[int, Tuple[float, float]],
    edges: Dict[Tuple[int, int], float],
    origin: int,
    destinations: List[int],
    landmarks: Dict[int, str],
    accident_edge: Tuple[int, int] | None,
    accident_severity: str,
    accident_multiplier: float,
    accident_base_cost: float | None,
) -> None:
    lines: List[str] = []
    lines.append("Nodes:")
    for node_id in sorted(nodes):
        x, y = nodes[node_id]
        lines.append(f"{node_id}: ({x:.6f},{y:.6f})")
    lines.append("")

    if landmarks:
        lines.append("Landmarks:")
        for node_id in sorted(landmarks):
            lines.append(f"{node_id}: {landmarks[node_id]}")
        lines.append("")

    lines.append("Edges:")
    for (u, v) in sorted(edges):
        lines.append(f"({u},{v}): {fmt_float(edges[(u, v)])}")
    lines.append("")

    lines.append("Origin:")
    lines.append(str(origin))
    lines.append("")

    lines.append("Destinations:")
    lines.append(";".join(str(dest) for dest in destinations))
    lines.append("")

    if accident_edge and accident_base_cost is not None:
        lines.append("Accident Update:")
        lines.append(f"ACCIDENT_EDGE: ({accident_edge[0]},{accident_edge[1]})")
        lines.append(f"SEVERITY: {accident_severity}")
        lines.append(f"MULTIPLIER: {accident_multiplier}")
        lines.append("")
        lines.append("Final Travel Time:")
        final_time = accident_base_cost * accident_multiplier
        lines.append(
            f"FINAL_TRAVEL_TIME: {fmt_float(final_time)} ({fmt_float(accident_base_cost)} * {accident_multiplier})"
        )
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class HeritageData:
    nodes: Dict[int, Tuple[float, float]]
    landmarks: Dict[int, str]
    edges: Dict[Tuple[int, int], float]
    way_time_lookup: Dict[int, Tuple[int, int, float]]
    start: int
    goals: List[int]
    accident_multiplier: float
    camera_way_ids: List[int]


def parse_heritage(raw_path: Path) -> HeritageData:
    section = None
    nodes: Dict[int, Tuple[float, float]] = {}
    landmarks: Dict[int, str] = {}
    edges: Dict[Tuple[int, int], float] = {}
    way_lookup: Dict[int, Tuple[int, int, float]] = {}
    start = None
    goals: List[int] = []
    accident_multiplier = 1.0
    camera_way_ids: List[int] = []

    for raw_line in raw_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
            continue

        if section == "NODES":
            node_id_str, lat_str, lon_str, name = line.split(",", 3)
            node_id = int(node_id_str)
            lat = float(lat_str)
            lon = float(lon_str)
            nodes[node_id] = (lon, lat)
            landmarks[node_id] = name.strip()
        elif section == "WAYS":
            parts = line.split(",")
            if len(parts) < 6:
                continue
            way_id = int(parts[0])
            u = int(parts[1])
            v = int(parts[2])
            minutes = float(parts[5])
            hours = minutes_to_hours(minutes)
            edges[(u, v)] = hours
            way_lookup[way_id] = (u, v, hours)
        elif section == "CAMERAS":
            way_id_str, _image = line.split(",", 1)
            camera_way_ids.append(int(way_id_str))
        elif section == "META":
            key, value = line.split(",", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "START":
                start = int(value)
            elif key == "GOAL":
                parts = value.replace(";", ",").split(",")
                goals = [int(x.strip()) for x in parts if x.strip()]
            elif key == "ACCIDENT_MULTIPLIER":
                accident_multiplier = float(value)

    if start is None or not goals:
        raise ValueError("Heritage file missing START/GOAL metadata")

    return HeritageData(
        nodes=nodes,
        landmarks=landmarks,
        edges=edges,
        way_time_lookup=way_lookup,
        start=start,
        goals=goals,
        accident_multiplier=accident_multiplier,
        camera_way_ids=camera_way_ids,
    )


def convert_heritage(src: Path, dest: Path) -> None:
    data = parse_heritage(src)
    accident_edge = None
    accident_cost = None

    for way_id in data.camera_way_ids:
        if way_id in data.way_time_lookup:
            u, v, hours = data.way_time_lookup[way_id]
            accident_edge = (u, v)
            accident_cost = hours
            break

    if accident_edge is None:
        print("[warn] No camera-matched edge found; skipping accident block")

    write_ics(
        dest,
        nodes=data.nodes,
        edges=data.edges,
        origin=data.start,
        destinations=data.goals,
        landmarks=data.landmarks,
        accident_edge=accident_edge,
        accident_severity="Severe" if data.accident_multiplier >= 1.5 else "Moderate",
        accident_multiplier=data.accident_multiplier,
        accident_base_cost=accident_cost,
    )
    print(f"[ok] Wrote heritage map to {dest}")

    image_path = dest.with_suffix(".png")
    meta_path = dest.with_suffix(".meta.json")
    generate_basemap_image(
        nodes=data.nodes,
        image_path=image_path,
        meta_path=meta_path,
        zoom=17,
    )


@dataclass
class OSMSpec:
    name: str
    bbox: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)
    landmarks: Dict[str, Tuple[float, float]]
    origin_name: str
    destination_names: List[str]
    connections: List[Tuple[str, str]]
    accident_connection: Tuple[str, str]
    accident_severity: str
    accident_multiplier: float
    zoom: int = 16


def compute_bbox(nodes: Dict[int, Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [coord[0] for coord in nodes.values()]
    ys = [coord[1] for coord in nodes.values()]
    return min(xs), max(xs), min(ys), max(ys)


def generate_basemap_image(
    *,
    nodes: Dict[int, Tuple[float, float]],
    image_path: Path,
    meta_path: Path,
    zoom: int,
):
    if not nodes:
        return

    west, east, south, north = compute_bbox(nodes)
    if math.isclose(west, east) or math.isclose(south, north):
        return

    try:
        img, extent = cx.bounds2img(
            west,
            south,
            east,
            north,
            zoom=zoom,
            source=cx.providers.OpenStreetMap.Mapnik,
            ll=True,
        )
    except Exception as exc:
        print(f"[warn] Failed to fetch basemap tiles: {exc}")
        return
    arr = img
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4 and arr.shape[2] > 4:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    Image.fromarray(arr).save(image_path)

    meta = {
        "image": os.path.basename(image_path),
        "projection": "web_mercator",
        "extent": {
            "xmin": float(extent[0]),
            "xmax": float(extent[1]),
            "ymin": float(extent[2]),
            "ymax": float(extent[3]),
        },
        "bounds": {
            "west": float(west),
            "east": float(east),
            "south": float(south),
            "north": float(north),
        },
        "zoom": zoom,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_osm_graph(osm_path: Path):
    print(f"[info] Loading OSM graph from {osm_path}")
    G = ox.graph_from_xml(osm_path, simplify=True, retain_all=False)
    for u, v, key, data in G.edges(keys=True, data=True):
        if data.get("length") is None:
            y1 = G.nodes[u].get("y")
            x1 = G.nodes[u].get("x")
            y2 = G.nodes[v].get("y")
            x2 = G.nodes[v].get("x")
            if None not in (x1, y1, x2, y2):
                data["length"] = ox.distance.great_circle_vec(y1, x1, y2, x2)
            else:
                data["length"] = 10.0
    G = ox.add_edge_speeds(G)
    for _u, _v, _key, data in G.edges(keys=True, data=True):
        if data.get("length") is None:
            data["length"] = 10.0
        if data.get("speed_kph") is None:
            data["speed_kph"] = 30.0
        speed = max(data.get("speed_kph", 30.0), 1.0)
        length = max(data.get("length", 10.0), 1.0)
        data["travel_time"] = (length / 1000.0) / speed * 3600.0
    return G


def slice_graph(G, bbox: Tuple[float, float, float, float]):
    lat_min, lat_max, lon_min, lon_max = bbox
    nodes_to_keep = [
        n
        for n, data in G.nodes(data=True)
        if lat_min <= data.get("y", 0) <= lat_max and lon_min <= data.get("x", 0) <= lon_max
    ]
    if not nodes_to_keep:
        raise ValueError("No nodes found in the requested bounding box.")
    sub = G.subgraph(nodes_to_keep).copy()
    if sub.number_of_nodes() == 0:
        raise ValueError("Subgraph is empty after slicing.")
    undirected = sub.to_undirected()
    components = list(nx.connected_components(undirected))
    if not components:
        return sub
    largest_nodes = max(components, key=len)
    return sub.subgraph(largest_nodes).copy()


def nearest_node_simple(G, lat: float, lon: float):
    best = None
    best_dist = float("inf")
    for node, data in G.nodes(data=True):
        x = data.get("x")
        y = data.get("y")
        if x is None or y is None:
            continue
        dist = (x - lon) ** 2 + (y - lat) ** 2
        if dist < best_dist:
            best = node
            best_dist = dist
    return best


def map_landmarks(G, landmarks_spec: Dict[str, Tuple[float, float]]):
    mapped: Dict[str, int] = {}
    for name, (lat, lon) in landmarks_spec.items():
        nearest = nearest_node_simple(G, lat, lon)
        if nearest is None:
            continue
        mapped[name] = nearest
    return mapped


def build_connection_graph(
    G,
    *,
    spec: OSMSpec,
    name_to_node: Dict[str, int],
) -> Tuple[Dict[int, Tuple[float, float]], Dict[Tuple[int, int], float], Dict[int, str], Dict[str, int]]:
    if not name_to_node:
        raise ValueError("No landmarks mapped to OSM nodes.")

    # Build a weighted simple DiGraph for shortest-path queries
    weighted = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        travel_time = data.get("travel_time")
        if travel_time is None:
            length = data.get("length", 0)
            if not length:
                continue
            travel_time = (length / 1000.0) / 40.0 * 3600.0
        travel_time = float(travel_time)
        if not math.isfinite(travel_time) or travel_time < 0:
            continue
        existing = weighted.get_edge_data(u, v)
        if existing is None or travel_time < existing["weight"]:
            weighted.add_edge(u, v, weight=travel_time)

    nodes: Dict[int, Tuple[float, float]] = {}
    landmarks: Dict[int, str] = {}
    name_to_newid: Dict[str, int] = {}
    for idx, name in enumerate(spec.landmarks.keys(), start=1):
        original = name_to_node.get(name)
        if original is None:
            continue
        data = G.nodes[original]
        nodes[idx] = (data["x"], data["y"])
        landmarks[idx] = name
        name_to_newid[name] = idx

    edges: Dict[Tuple[int, int], float] = {}
    for name_a, name_b in spec.connections:
        if name_a not in name_to_newid or name_b not in name_to_newid:
            print(f"[warn] Connection uses unknown landmark ({name_a} -> {name_b})")
            continue
        origin_node = name_to_node.get(name_a)
        target_node = name_to_node.get(name_b)
        if origin_node is None or target_node is None:
            print(f"[warn] Unable to map connection ({name_a} -> {name_b}) to OSM nodes")
            continue
        try:
            cost_seconds = nx.shortest_path_length(
                weighted, origin_node, target_node, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"[warn] No path between {name_a} and {name_b}")
            continue
        new_u = name_to_newid[name_a]
        new_v = name_to_newid[name_b]
        edges[(new_u, new_v)] = cost_seconds / 3600.0

    return nodes, edges, landmarks, name_to_newid


def build_osm_map(base_graph, spec: OSMSpec, dest: Path) -> None:
    sub = slice_graph(
        base_graph,
        (
            spec.bbox[0],
            spec.bbox[1],
            spec.bbox[2],
            spec.bbox[3],
        ),
    )
    name_to_node = map_landmarks(sub, spec.landmarks)
    if len(name_to_node) < len(spec.landmarks):
        missing = set(spec.landmarks) - set(name_to_node)
        print(f"[warn] Could not map landmarks for {spec.name}: {', '.join(missing)}")

    nodes, edges, landmark_map, name_to_newid = build_connection_graph(
        sub, spec=spec, name_to_node=name_to_node
    )
    if not nodes or not edges:
        raise ValueError(f"Simplified graph for {spec.name} is empty.")

    def resolve_newid(name: str) -> int:
        node_id = name_to_newid.get(name)
        if node_id is None:
            raise ValueError(f"Landmark '{name}' not mapped in {spec.name}")
        return node_id

    origin = resolve_newid(spec.origin_name)
    destinations = [resolve_newid(name) for name in spec.destination_names]

    accident_edge = tuple(resolve_newid(name) for name in spec.accident_connection)
    accident_base = edges.get(accident_edge)
    if accident_base is None:
        (u, v), accident_base = next(iter(edges.items()))
        accident_edge = (u, v)

    write_ics(
        dest,
        nodes=nodes,
        edges=edges,
        origin=origin,
        destinations=destinations,
        landmarks=landmark_map,
        accident_edge=accident_edge,
        accident_severity=spec.accident_severity,
        accident_multiplier=spec.accident_multiplier,
        accident_base_cost=accident_base,
    )
    print(f"[ok] Wrote OSM-derived map to {dest}")

    image_path = dest.with_suffix(".png")
    meta_path = dest.with_suffix(".meta.json")
    generate_basemap_image(
        nodes=nodes,
        image_path=image_path,
        meta_path=meta_path,
        zoom=spec.zoom,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate Kuching maps from assets")
    parser.add_argument("--heritage", action="store_true", help="Only rebuild heritage map")
    parser.add_argument("--osm", action="store_true", help="Only rebuild OSM-based maps")
    parser.add_argument("--output-dir", default="maps", help="Directory for generated maps")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heritage_src = Path("maps/heritage_assignment_15_time_asymmetric-1.txt")
    heritage_dest = output_dir / "kuching_heritage.txt"

    if not args.osm:
        convert_heritage(heritage_src, heritage_dest)

    if not args.heritage:
        osm_path = Path("maps/map.osm")
        base_graph = load_osm_graph(osm_path)

        specs = [
            OSMSpec(
                name="kuching_central_osm",
                bbox=(1.5540, 1.5612, 110.3390, 110.3495),
                landmarks={
                    "Padang Merdeka": (1.557014, 110.343616),
                    "Kuching Waterfront": (1.557950, 110.347900),
                    "Old Courthouse Auditorium": (1.558856, 110.344629),
                    "Carpenter Street": (1.558200, 110.346300),
                    "Simpang Tiga Interchange": (1.555500, 110.346800),
                    "Chinese History Museum": (1.557650, 110.346300),
                    "Darul Hana Bridge": (1.557200, 110.348300),
                    "Tua Pek Kong Temple": (1.557620, 110.346700),
                    "India Street": (1.557700, 110.344600),
                    "Main Bazaar": (1.558050, 110.345900),
                    "Plaza Merdeka": (1.558558, 110.344180),
                    "Electra House": (1.557300, 110.345300),
                },
                origin_name="Padang Merdeka",
                destination_names=["Carpenter Street", "Kuching Waterfront"],
                connections=[
                    ("Padang Merdeka", "Old Courthouse Auditorium"),
                    ("Old Courthouse Auditorium", "Padang Merdeka"),
                    ("Old Courthouse Auditorium", "Kuching Waterfront"),
                    ("Kuching Waterfront", "Old Courthouse Auditorium"),
                    ("Padang Merdeka", "Carpenter Street"),
                    ("Carpenter Street", "Padang Merdeka"),
                    ("Carpenter Street", "Kuching Waterfront"),
                    ("Kuching Waterfront", "Carpenter Street"),
                    ("Padang Merdeka", "Simpang Tiga Interchange"),
                    ("Simpang Tiga Interchange", "Padang Merdeka"),
                    ("Simpang Tiga Interchange", "Kuching Waterfront"),
                    ("Kuching Waterfront", "Simpang Tiga Interchange"),
                    ("Padang Merdeka", "Chinese History Museum"),
                    ("Chinese History Museum", "Padang Merdeka"),
                    ("Chinese History Museum", "Tua Pek Kong Temple"),
                    ("Tua Pek Kong Temple", "Chinese History Museum"),
                    ("Tua Pek Kong Temple", "Darul Hana Bridge"),
                    ("Darul Hana Bridge", "Tua Pek Kong Temple"),
                    ("Padang Merdeka", "India Street"),
                    ("India Street", "Padang Merdeka"),
                    ("India Street", "Plaza Merdeka"),
                    ("Plaza Merdeka", "India Street"),
                    ("Plaza Merdeka", "Old Courthouse Auditorium"),
                    ("Old Courthouse Auditorium", "Plaza Merdeka"),
                    ("Main Bazaar", "Kuching Waterfront"),
                    ("Kuching Waterfront", "Main Bazaar"),
                    ("Electra House", "Padang Merdeka"),
                    ("Padang Merdeka", "Electra House"),
                    ("Electra House", "India Street"),
                    ("India Street", "Electra House"),
                ],
                accident_connection=("Padang Merdeka", "Old Courthouse Auditorium"),
                accident_severity="Moderate",
                accident_multiplier=1.3,
                zoom=16,
            ),
            OSMSpec(
                name="kuching_museum_loop_osm",
                bbox=(1.5500, 1.5585, 110.3390, 110.3455),
                landmarks={
                    "Sarawak Museum": (1.551910, 110.343500),
                    "Sarawak Islamic Heritage Museum": (1.551450, 110.342800),
                    "Sarawak Art Museum": (1.553400, 110.341900),
                    "Heroes Monument": (1.553000, 110.340800),
                    "Padang Merdeka": (1.557014, 110.343616),
                    "Wisma Hopoh": (1.556000, 110.340200),
                    "Textile Museum": (1.557400, 110.343900),
                    "Natural History Museum": (1.551800, 110.343100),
                    "Taman Budaya Theater": (1.552800, 110.341200),
                    "Bukit Siol Viewpoint": (1.555200, 110.342200),
                    "St. Thomas' Cathedral": (1.557108, 110.345049),
                },
                origin_name="Sarawak Museum",
                destination_names=["Padang Merdeka", "Sarawak Islamic Heritage Museum"],
                connections=[
                    ("Sarawak Museum", "Sarawak Islamic Heritage Museum"),
                    ("Sarawak Islamic Heritage Museum", "Sarawak Museum"),
                    ("Sarawak Museum", "Padang Merdeka"),
                    ("Padang Merdeka", "Sarawak Museum"),
                    ("Sarawak Museum", "Natural History Museum"),
                    ("Natural History Museum", "Sarawak Museum"),
                    ("Natural History Museum", "Sarawak Islamic Heritage Museum"),
                    ("Sarawak Islamic Heritage Museum", "Natural History Museum"),
                    ("Sarawak Art Museum", "Padang Merdeka"),
                    ("Padang Merdeka", "Sarawak Art Museum"),
                    ("Sarawak Art Museum", "Heroes Monument"),
                    ("Heroes Monument", "Sarawak Art Museum"),
                    ("Sarawak Islamic Heritage Museum", "Padang Merdeka"),
                    ("Padang Merdeka", "Sarawak Islamic Heritage Museum"),
                    ("Sarawak Art Museum", "Wisma Hopoh"),
                    ("Wisma Hopoh", "Sarawak Art Museum"),
                    ("Wisma Hopoh", "Padang Merdeka"),
                    ("Padang Merdeka", "Wisma Hopoh"),
                    ("Padang Merdeka", "Textile Museum"),
                    ("Textile Museum", "Padang Merdeka"),
                    ("Textile Museum", "St. Thomas' Cathedral"),
                    ("St. Thomas' Cathedral", "Textile Museum"),
                    ("Taman Budaya Theater", "Sarawak Art Museum"),
                    ("Sarawak Art Museum", "Taman Budaya Theater"),
                    ("Bukit Siol Viewpoint", "Padang Merdeka"),
                    ("Padang Merdeka", "Bukit Siol Viewpoint"),
                ],
                accident_connection=("Sarawak Museum", "Sarawak Islamic Heritage Museum"),
                accident_severity="Severe",
                accident_multiplier=1.5,
                zoom=17,
            ),
        ]

        for spec in specs:
            dest = output_dir / f"{spec.name}.txt"
            build_osm_map(base_graph, spec, dest)


if __name__ == "__main__":
    main()
