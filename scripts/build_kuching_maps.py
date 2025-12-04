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


def build_weighted_graph(osm_graph):
    weighted = nx.DiGraph()
    for u, v, data in osm_graph.edges(data=True):
        travel_time = data.get("travel_time")
        if travel_time is None:
            length = data.get("length", 0)
            if not length:
                continue
            travel_time = (length / 1000.0) / 40.0 * 3600.0
        travel_time = float(travel_time)
        if not math.isfinite(travel_time) or travel_time <= 0:
            continue
        existing = weighted.get_edge_data(u, v)
        if existing is None or travel_time < existing["weight"]:
            weighted.add_edge(u, v, weight=travel_time)
    return weighted


def compute_polylines_for_edges(
    osm_graph,
    weighted_graph,
    coords: Dict[int, Tuple[float, float]],
    edges: Dict[Tuple[int, int], float],
):
    mapping: Dict[int, int] = {}
    for node_id, (lon, lat) in coords.items():
        mapped = nearest_node_simple(osm_graph, lat, lon)
        if mapped is not None:
            mapping[node_id] = mapped

    polylines: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    for (u, v) in edges:
        start = mapping.get(u)
        end = mapping.get(v)
        if start is None or end is None:
            continue
        try:
            path_nodes = nx.shortest_path(
                weighted_graph, start, end, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        points = []
        for node_id in path_nodes:
            node_data = osm_graph.nodes[node_id]
            points.append((node_data["x"], node_data["y"]))
        polylines[(u, v)] = points
    return polylines


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


def convert_heritage(src: Path, dest: Path, osm_graph=None, weighted_graph=None) -> None:
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
    heritage_polylines: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
    if osm_graph is not None and weighted_graph is not None:
        heritage_polylines = compute_polylines_for_edges(
            osm_graph,
            weighted_graph,
            data.nodes,
            data.edges,
        )

    extra_points = [pt for poly in heritage_polylines.values() for pt in poly]
    generate_basemap_image(
        nodes=data.nodes,
        image_path=image_path,
        meta_path=meta_path,
        zoom=17,
        extra_points=extra_points,
    )
    poly_path = dest.with_suffix(".paths.json")
    payload = {
        f"{u}-{v}": [{"lon": float(px), "lat": float(py)} for px, py in points]
        for (u, v), points in heritage_polylines.items()
    }
    poly_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    extra_points: List[Tuple[float, float]] | None = None,
):
    if not nodes:
        return

    bbox_nodes = dict(nodes)
    if extra_points:
        extra_index = {-(idx + 1): pt for idx, pt in enumerate(extra_points)}
        bbox_nodes.update(extra_index)

    west, east, south, north = compute_bbox(bbox_nodes)
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
    weighted_graph,
    *,
    spec: OSMSpec,
    name_to_node: Dict[str, int],
) -> Tuple[
    Dict[int, Tuple[float, float]],
    Dict[Tuple[int, int], float],
    Dict[int, str],
    Dict[str, int],
    Dict[Tuple[int, int], List[Tuple[float, float]]],
]:
    if not name_to_node:
        raise ValueError("No landmarks mapped to OSM nodes.")

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
    polylines: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}
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
            path_nodes = nx.shortest_path(
                weighted_graph, origin_node, target_node, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            print(f"[warn] No path between {name_a} and {name_b}")
            continue
        cost_seconds = 0.0
        polyline_points: List[Tuple[float, float]] = []
        for idx in range(len(path_nodes) - 1):
            u = path_nodes[idx]
            v = path_nodes[idx + 1]
            data = weighted_graph.get_edge_data(u, v)
            if data is None:
                continue
            cost_seconds += data["weight"]
        for node_id in path_nodes:
            node_data = G.nodes[node_id]
            polyline_points.append((node_data["x"], node_data["y"]))
        new_u = name_to_newid[name_a]
        new_v = name_to_newid[name_b]
        edges[(new_u, new_v)] = cost_seconds / 3600.0
        polylines[(new_u, new_v)] = polyline_points
    return nodes, edges, landmarks, name_to_newid, polylines


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
    weighted_sub = build_weighted_graph(sub)
    name_to_node = map_landmarks(sub, spec.landmarks)
    if len(name_to_node) < len(spec.landmarks):
        missing = set(spec.landmarks) - set(name_to_node)
        print(f"[warn] Could not map landmarks for {spec.name}: {', '.join(missing)}")

    nodes, edges, landmark_map, name_to_newid, polylines = build_connection_graph(
        sub, weighted_sub, spec=spec, name_to_node=name_to_node
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
    extra_points = [pt for polyline in polylines.values() for pt in polyline]
    generate_basemap_image(
        nodes=nodes,
        image_path=image_path,
        meta_path=meta_path,
        zoom=spec.zoom,
        extra_points=extra_points,
    )

    poly_path = dest.with_suffix(".paths.json")
    path_payload = {
        f"{u}-{v}": [{"lon": float(px), "lat": float(py)} for px, py in points]
        for (u, v), points in polylines.items()
    }
    poly_path.write_text(json.dumps(path_payload, indent=2), encoding="utf-8")


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
    osm_path = Path("maps/map.osm")
    base_graph = load_osm_graph(osm_path)
    weighted_graph = build_weighted_graph(base_graph)

    if not args.osm:
        convert_heritage(
            heritage_src,
            heritage_dest,
            osm_graph=base_graph,
            weighted_graph=weighted_graph,
        )

    if not args.heritage:
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
            OSMSpec(
                name="kuching_padungan_waterfront",
                bbox=(1.5530, 1.5590, 110.3410, 110.3488),
                landmarks={
                    "Kuching Waterfront": (1.55749, 110.34428),
                    "Darul Hana Bridge": (1.55728, 110.34752),
                    "Grand Margherita Hotel": (1.55878, 110.34340),
                    "Kuching Esplanade": (1.55700, 110.34380),
                    "Tua Pek Kong Temple": (1.55712, 110.34483),
                    "Carpenter Street Entrance": (1.55644, 110.34401),
                    "Plaza Merdeka Entrance": (1.55665, 110.34455),
                    "Jalan Padungan Arch": (1.55720, 110.34270),
                    "Riverside Majestic Hotel": (1.55802, 110.34498),
                    "Jalan Main Bazaar": (1.55701, 110.34520),
                },
                origin_name="Kuching Waterfront",
                destination_names=["Plaza Merdeka Entrance", "Jalan Main Bazaar"],
                connections=[
                    ("Kuching Waterfront", "Darul Hana Bridge"),
                    ("Darul Hana Bridge", "Kuching Waterfront"),
                    ("Kuching Waterfront", "Grand Margherita Hotel"),
                    ("Grand Margherita Hotel", "Kuching Waterfront"),
                    ("Kuching Waterfront", "Kuching Esplanade"),
                    ("Kuching Esplanade", "Kuching Waterfront"),
                    ("Kuching Waterfront", "Tua Pek Kong Temple"),
                    ("Tua Pek Kong Temple", "Kuching Waterfront"),
                    ("Tua Pek Kong Temple", "Carpenter Street Entrance"),
                    ("Carpenter Street Entrance", "Tua Pek Kong Temple"),
                    ("Carpenter Street Entrance", "Plaza Merdeka Entrance"),
                    ("Plaza Merdeka Entrance", "Carpenter Street Entrance"),
                    ("Plaza Merdeka Entrance", "Jalan Padungan Arch"),
                    ("Jalan Padungan Arch", "Plaza Merdeka Entrance"),
                    ("Jalan Padungan Arch", "Grand Margherita Hotel"),
                    ("Grand Margherita Hotel", "Jalan Padungan Arch"),
                    ("Grand Margherita Hotel", "Riverside Majestic Hotel"),
                    ("Riverside Majestic Hotel", "Grand Margherita Hotel"),
                    ("Darul Hana Bridge", "Jalan Main Bazaar"),
                    ("Jalan Main Bazaar", "Darul Hana Bridge"),
                    ("Tua Pek Kong Temple", "Jalan Main Bazaar"),
                    ("Jalan Main Bazaar", "Tua Pek Kong Temple"),
                    ("Carpenter Street Entrance", "Jalan Main Bazaar"),
                    ("Jalan Main Bazaar", "Carpenter Street Entrance"),
                    ("Riverside Majestic Hotel", "Jalan Main Bazaar"),
                    ("Jalan Main Bazaar", "Riverside Majestic Hotel"),
                    ("Darul Hana Bridge", "Riverside Majestic Hotel"),
                    ("Riverside Majestic Hotel", "Darul Hana Bridge"),
                    ("Kuching Esplanade", "Carpenter Street Entrance"),
                    ("Carpenter Street Entrance", "Kuching Esplanade"),
                    ("Kuching Esplanade", "Tua Pek Kong Temple"),
                    ("Tua Pek Kong Temple", "Kuching Esplanade"),
                ],
                accident_connection=("Kuching Waterfront", "Tua Pek Kong Temple"),
                accident_severity="Severe",
                accident_multiplier=1.8,
                zoom=17,
            ),
            OSMSpec(
                name="kuching_petra_jaya",
                bbox=(1.5700, 1.5800, 110.3300, 110.3450),
                landmarks={
                    "Masjid Jamek Petra Jaya": (1.57428, 110.33640),
                    "Sarawak State Library": (1.57710, 110.33822),
                    "Pustaka Lake Park": (1.57600, 110.33870),
                    "Wisma Bapa Malaysia": (1.57344, 110.34041),
                    "State Assembly Viewpoint": (1.57350, 110.34410),
                    "Stadium Sarawak": (1.57760, 110.33070),
                    "Petra Jaya Community Centre": (1.57610, 110.33350),
                    "Kampung Tupong Market": (1.57190, 110.33490),
                    "Sarawak Heart Centre": (1.57940, 110.33570),
                    "Mini Garden Petra Jaya": (1.57290, 110.33790),
                },
                origin_name="Sarawak State Library",
                destination_names=["Stadium Sarawak", "State Assembly Viewpoint"],
                connections=[
                    ("Sarawak State Library", "Masjid Jamek Petra Jaya"),
                    ("Masjid Jamek Petra Jaya", "Sarawak State Library"),
                    ("Sarawak State Library", "Pustaka Lake Park"),
                    ("Pustaka Lake Park", "Sarawak State Library"),
                    ("Sarawak State Library", "Wisma Bapa Malaysia"),
                    ("Wisma Bapa Malaysia", "Sarawak State Library"),
                    ("Sarawak State Library", "State Assembly Viewpoint"),
                    ("State Assembly Viewpoint", "Sarawak State Library"),
                    ("Sarawak State Library", "Mini Garden Petra Jaya"),
                    ("Mini Garden Petra Jaya", "Sarawak State Library"),
                    ("Pustaka Lake Park", "Mini Garden Petra Jaya"),
                    ("Mini Garden Petra Jaya", "Pustaka Lake Park"),
                    ("Wisma Bapa Malaysia", "Petra Jaya Community Centre"),
                    ("Petra Jaya Community Centre", "Wisma Bapa Malaysia"),
                    ("State Assembly Viewpoint", "Wisma Bapa Malaysia"),
                    ("Wisma Bapa Malaysia", "State Assembly Viewpoint"),
                    ("State Assembly Viewpoint", "Petra Jaya Community Centre"),
                    ("Petra Jaya Community Centre", "State Assembly Viewpoint"),
                    ("State Assembly Viewpoint", "Kampung Tupong Market"),
                    ("Kampung Tupong Market", "State Assembly Viewpoint"),
                    ("Petra Jaya Community Centre", "Stadium Sarawak"),
                    ("Stadium Sarawak", "Petra Jaya Community Centre"),
                    ("Petra Jaya Community Centre", "Kampung Tupong Market"),
                    ("Kampung Tupong Market", "Petra Jaya Community Centre"),
                    ("Petra Jaya Community Centre", "Mini Garden Petra Jaya"),
                    ("Mini Garden Petra Jaya", "Petra Jaya Community Centre"),
                    ("Stadium Sarawak", "Sarawak Heart Centre"),
                    ("Sarawak Heart Centre", "Stadium Sarawak"),
                    ("Sarawak Heart Centre", "Mini Garden Petra Jaya"),
                    ("Mini Garden Petra Jaya", "Sarawak Heart Centre"),
                    ("Masjid Jamek Petra Jaya", "Kampung Tupong Market"),
                    ("Kampung Tupong Market", "Masjid Jamek Petra Jaya"),
                ],
                accident_connection=("Sarawak State Library", "Pustaka Lake Park"),
                accident_severity="Minor",
                accident_multiplier=1.2,
                zoom=16,
            ),
            OSMSpec(
                name="kuching_pending_industrial",
                bbox=(1.5500, 1.5680, 110.3800, 110.3980),
                landmarks={
                    "Pending Industrial Gate A": (1.55650, 110.38920),
                    "Jalan Semangat Roundabout": (1.55790, 110.39210),
                    "Pending Port Road": (1.55850, 110.39400),
                    "Senari Port Cargo Entrance": (1.56130, 110.39590),
                    "Pending Food Centre": (1.56220, 110.39080),
                    "Kampung Tabuan Hilir Entrance": (1.56020, 110.38770),
                    "Industrial Block C": (1.55930, 110.39190),
                    "Jalan Kemajuan Junction": (1.55710, 110.39370),
                    "Shell Pending Depot": (1.55900, 110.39750),
                    "Timber Industrial Lots": (1.55790, 110.39510),
                },
                origin_name="Pending Industrial Gate A",
                destination_names=["Pending Food Centre", "Shell Pending Depot"],
                connections=[
                    ("Pending Industrial Gate A", "Jalan Semangat Roundabout"),
                    ("Jalan Semangat Roundabout", "Pending Industrial Gate A"),
                    ("Pending Industrial Gate A", "Pending Food Centre"),
                    ("Pending Food Centre", "Pending Industrial Gate A"),
                    ("Pending Industrial Gate A", "Kampung Tabuan Hilir Entrance"),
                    ("Kampung Tabuan Hilir Entrance", "Pending Industrial Gate A"),
                    ("Jalan Semangat Roundabout", "Pending Port Road"),
                    ("Pending Port Road", "Jalan Semangat Roundabout"),
                    ("Jalan Semangat Roundabout", "Kampung Tabuan Hilir Entrance"),
                    ("Kampung Tabuan Hilir Entrance", "Jalan Semangat Roundabout"),
                    ("Jalan Semangat Roundabout", "Industrial Block C"),
                    ("Industrial Block C", "Jalan Semangat Roundabout"),
                    ("Pending Port Road", "Senari Port Cargo Entrance"),
                    ("Senari Port Cargo Entrance", "Pending Port Road"),
                    ("Pending Port Road", "Shell Pending Depot"),
                    ("Shell Pending Depot", "Pending Port Road"),
                    ("Pending Port Road", "Jalan Kemajuan Junction"),
                    ("Jalan Kemajuan Junction", "Pending Port Road"),
                    ("Senari Port Cargo Entrance", "Shell Pending Depot"),
                    ("Shell Pending Depot", "Senari Port Cargo Entrance"),
                    ("Senari Port Cargo Entrance", "Pending Food Centre"),
                    ("Pending Food Centre", "Senari Port Cargo Entrance"),
                    ("Pending Food Centre", "Industrial Block C"),
                    ("Industrial Block C", "Pending Food Centre"),
                    ("Pending Food Centre", "Kampung Tabuan Hilir Entrance"),
                    ("Kampung Tabuan Hilir Entrance", "Pending Food Centre"),
                    ("Pending Food Centre", "Shell Pending Depot"),
                    ("Shell Pending Depot", "Pending Food Centre"),
                    ("Kampung Tabuan Hilir Entrance", "Industrial Block C"),
                    ("Industrial Block C", "Kampung Tabuan Hilir Entrance"),
                    ("Industrial Block C", "Jalan Kemajuan Junction"),
                    ("Jalan Kemajuan Junction", "Industrial Block C"),
                    ("Jalan Kemajuan Junction", "Timber Industrial Lots"),
                    ("Timber Industrial Lots", "Jalan Kemajuan Junction"),
                    ("Shell Pending Depot", "Timber Industrial Lots"),
                    ("Timber Industrial Lots", "Shell Pending Depot"),
                ],
                accident_connection=("Jalan Semangat Roundabout", "Pending Port Road"),
                accident_severity="Moderate",
                accident_multiplier=1.4,
                zoom=15,
            ),
            OSMSpec(
                name="kuching_airport_corridor",
                bbox=(1.4750, 1.4930, 110.3300, 110.3540),
                landmarks={
                    "Kuching International Airport": (1.48450, 110.34750),
                    "Airport Junction": (1.48720, 110.34510),
                    "Simpang Tiga Roundabout": (1.49210, 110.34310),
                    "CityONE Megamall": (1.49250, 110.34740),
                    "Jalan Stutong Junction": (1.49180, 110.34900),
                    "Tabuan Tranquility Entrance": (1.48810, 110.35140),
                    "KIA Car Rental Zone": (1.48390, 110.34870),
                    "Airport Mosque": (1.48570, 110.34620),
                    "Airport Fire Station": (1.48420, 110.34800),
                    "Premier Food Court": (1.48950, 110.34450),
                },
                origin_name="Kuching International Airport",
                destination_names=["CityONE Megamall", "Premier Food Court"],
                connections=[
                    ("Kuching International Airport", "Airport Junction"),
                    ("Airport Junction", "Kuching International Airport"),
                    ("Kuching International Airport", "KIA Car Rental Zone"),
                    ("KIA Car Rental Zone", "Kuching International Airport"),
                    ("Kuching International Airport", "Airport Mosque"),
                    ("Airport Mosque", "Kuching International Airport"),
                    ("Kuching International Airport", "Airport Fire Station"),
                    ("Airport Fire Station", "Kuching International Airport"),
                    ("Airport Junction", "Simpang Tiga Roundabout"),
                    ("Simpang Tiga Roundabout", "Airport Junction"),
                    ("Airport Junction", "Jalan Stutong Junction"),
                    ("Jalan Stutong Junction", "Airport Junction"),
                    ("Airport Junction", "Premier Food Court"),
                    ("Premier Food Court", "Airport Junction"),
                    ("Simpang Tiga Roundabout", "CityONE Megamall"),
                    ("CityONE Megamall", "Simpang Tiga Roundabout"),
                    ("Simpang Tiga Roundabout", "Jalan Stutong Junction"),
                    ("Jalan Stutong Junction", "Simpang Tiga Roundabout"),
                    ("Simpang Tiga Roundabout", "Premier Food Court"),
                    ("Premier Food Court", "Simpang Tiga Roundabout"),
                    ("CityONE Megamall", "Jalan Stutong Junction"),
                    ("Jalan Stutong Junction", "CityONE Megamall"),
                    ("CityONE Megamall", "Tabuan Tranquility Entrance"),
                    ("Tabuan Tranquility Entrance", "CityONE Megamall"),
                    ("Jalan Stutong Junction", "Tabuan Tranquility Entrance"),
                    ("Tabuan Tranquility Entrance", "Jalan Stutong Junction"),
                    ("Tabuan Tranquility Entrance", "Premier Food Court"),
                    ("Premier Food Court", "Tabuan Tranquility Entrance"),
                    ("KIA Car Rental Zone", "Airport Mosque"),
                    ("Airport Mosque", "KIA Car Rental Zone"),
                    ("KIA Car Rental Zone", "Airport Fire Station"),
                    ("Airport Fire Station", "KIA Car Rental Zone"),
                    ("Airport Mosque", "Airport Fire Station"),
                    ("Airport Fire Station", "Airport Mosque"),
                ],
                accident_connection=("Simpang Tiga Roundabout", "Airport Junction"),
                accident_severity="Severe",
                accident_multiplier=2.0,
                zoom=15,
            ),
            OSMSpec(
                name="kuching_batu_kawa",
                bbox=(1.5000, 1.5150, 110.3000, 110.3230),
                landmarks={
                    "MJC Batu Kawa Township": (1.50820, 110.31090),
                    "Batu Kawa Old Town Market": (1.50560, 110.31170),
                    "Lee Onn Supermarket": (1.50950, 110.31280),
                    "Moyan Square": (1.50810, 110.31690),
                    "Esplanade Batu Kawa": (1.50720, 110.30940),
                    "Jalan Stapok Junction": (1.51030, 110.30470),
                    "Batu Kawa Bridge": (1.50640, 110.30760),
                    "Jalan Kong Ping Roundabout": (1.51080, 110.31210),
                    "Brighton Square": (1.50970, 110.31740),
                    "Kuching City Mall": (1.51050, 110.32010),
                },
                origin_name="MJC Batu Kawa Township",
                destination_names=["Kuching City Mall", "Batu Kawa Bridge"],
                connections=[
                    ("MJC Batu Kawa Township", "Batu Kawa Old Town Market"),
                    ("Batu Kawa Old Town Market", "MJC Batu Kawa Township"),
                    ("MJC Batu Kawa Township", "Lee Onn Supermarket"),
                    ("Lee Onn Supermarket", "MJC Batu Kawa Township"),
                    ("MJC Batu Kawa Township", "Esplanade Batu Kawa"),
                    ("Esplanade Batu Kawa", "MJC Batu Kawa Township"),
                    ("MJC Batu Kawa Township", "Jalan Kong Ping Roundabout"),
                    ("Jalan Kong Ping Roundabout", "MJC Batu Kawa Township"),
                    ("Batu Kawa Old Town Market", "Esplanade Batu Kawa"),
                    ("Esplanade Batu Kawa", "Batu Kawa Old Town Market"),
                    ("Batu Kawa Old Town Market", "Batu Kawa Bridge"),
                    ("Batu Kawa Bridge", "Batu Kawa Old Town Market"),
                    ("Lee Onn Supermarket", "Moyan Square"),
                    ("Moyan Square", "Lee Onn Supermarket"),
                    ("Lee Onn Supermarket", "Jalan Kong Ping Roundabout"),
                    ("Jalan Kong Ping Roundabout", "Lee Onn Supermarket"),
                    ("Lee Onn Supermarket", "Brighton Square"),
                    ("Brighton Square", "Lee Onn Supermarket"),
                    ("Moyan Square", "Brighton Square"),
                    ("Brighton Square", "Moyan Square"),
                    ("Moyan Square", "Kuching City Mall"),
                    ("Kuching City Mall", "Moyan Square"),
                    ("Esplanade Batu Kawa", "Batu Kawa Bridge"),
                    ("Batu Kawa Bridge", "Esplanade Batu Kawa"),
                    ("Jalan Stapok Junction", "Batu Kawa Bridge"),
                    ("Batu Kawa Bridge", "Jalan Stapok Junction"),
                    ("Jalan Stapok Junction", "Jalan Kong Ping Roundabout"),
                    ("Jalan Kong Ping Roundabout", "Jalan Stapok Junction"),
                    ("Jalan Stapok Junction", "Kuching City Mall"),
                    ("Kuching City Mall", "Jalan Stapok Junction"),
                    ("Batu Kawa Bridge", "Kuching City Mall"),
                    ("Kuching City Mall", "Batu Kawa Bridge"),
                    ("Jalan Kong Ping Roundabout", "Brighton Square"),
                    ("Brighton Square", "Jalan Kong Ping Roundabout"),
                    ("Brighton Square", "Kuching City Mall"),
                    ("Kuching City Mall", "Brighton Square"),
                ],
                accident_connection=("Moyan Square", "Lee Onn Supermarket"),
                accident_severity="Minor",
                accident_multiplier=1.1,
                zoom=16,
            ),
            OSMSpec(
                name="kuching_matang_kubah",
                bbox=(1.5610, 1.5830, 110.2600, 110.2850),
                landmarks={
                    "Kubah National Park Entrance": (1.57290, 110.26650),
                    "Matang Wildlife Centre": (1.57140, 110.26900),
                    "Jalan Matang Mall": (1.57330, 110.27940),
                    "Matang Mid-Century Park": (1.57730, 110.27900),
                    "Taman Sri Matang": (1.57260, 110.27480),
                    "E-Mart Matang": (1.57210, 110.28060),
                    "Surau Darul Muttaqin": (1.57120, 110.27740),
                    "Matang Hospital": (1.57840, 110.27510),
                    "Taman Palma Matang": (1.57560, 110.27780),
                    "Jalan Depo Matang Junction": (1.57360, 110.26870),
                },
                origin_name="Kubah National Park Entrance",
                destination_names=["E-Mart Matang", "Jalan Matang Mall"],
                connections=[
                    ("Kubah National Park Entrance", "Matang Wildlife Centre"),
                    ("Matang Wildlife Centre", "Kubah National Park Entrance"),
                    ("Kubah National Park Entrance", "Taman Sri Matang"),
                    ("Taman Sri Matang", "Kubah National Park Entrance"),
                    ("Kubah National Park Entrance", "Jalan Depo Matang Junction"),
                    ("Jalan Depo Matang Junction", "Kubah National Park Entrance"),
                    ("Matang Wildlife Centre", "Jalan Matang Mall"),
                    ("Jalan Matang Mall", "Matang Wildlife Centre"),
                    ("Matang Wildlife Centre", "Taman Sri Matang"),
                    ("Taman Sri Matang", "Matang Wildlife Centre"),
                    ("Matang Wildlife Centre", "Surau Darul Muttaqin"),
                    ("Surau Darul Muttaqin", "Matang Wildlife Centre"),
                    ("Jalan Matang Mall", "Matang Mid-Century Park"),
                    ("Matang Mid-Century Park", "Jalan Matang Mall"),
                    ("Jalan Matang Mall", "E-Mart Matang"),
                    ("E-Mart Matang", "Jalan Matang Mall"),
                    ("Jalan Matang Mall", "Surau Darul Muttaqin"),
                    ("Surau Darul Muttaqin", "Jalan Matang Mall"),
                    ("Matang Mid-Century Park", "E-Mart Matang"),
                    ("E-Mart Matang", "Matang Mid-Century Park"),
                    ("Matang Mid-Century Park", "Taman Palma Matang"),
                    ("Taman Palma Matang", "Matang Mid-Century Park"),
                    ("Taman Sri Matang", "Surau Darul Muttaqin"),
                    ("Surau Darul Muttaqin", "Taman Sri Matang"),
                    ("Taman Sri Matang", "Jalan Depo Matang Junction"),
                    ("Jalan Depo Matang Junction", "Taman Sri Matang"),
                    ("E-Mart Matang", "Surau Darul Muttaqin"),
                    ("Surau Darul Muttaqin", "E-Mart Matang"),
                    ("E-Mart Matang", "Taman Palma Matang"),
                    ("Taman Palma Matang", "E-Mart Matang"),
                    ("Surau Darul Muttaqin", "Taman Palma Matang"),
                    ("Taman Palma Matang", "Surau Darul Muttaqin"),
                    ("Matang Hospital", "Taman Palma Matang"),
                    ("Taman Palma Matang", "Matang Hospital"),
                    ("Matang Hospital", "Jalan Depo Matang Junction"),
                    ("Jalan Depo Matang Junction", "Matang Hospital"),
                ],
                accident_connection=("Matang Wildlife Centre", "Taman Sri Matang"),
                accident_severity="Moderate",
                accident_multiplier=1.3,
                zoom=15,
            ),
            OSMSpec(
                name="kuching_saradise_stutong",
                bbox=(1.5050, 1.5200, 110.3500, 110.3700),
                landmarks={
                    "Saradise Kuching": (1.51280, 110.35910),
                    "Stutong Forest Park": (1.51210, 110.36340),
                    "Jalan Setia Raja Junction": (1.51710, 110.35810),
                    "Vivacity Megamall": (1.52020, 110.35370),
                    "Stutong Market": (1.51570, 110.35850),
                    "KPJ Kuching Specialist Hospital": (1.51500, 110.36610),
                    "One Residency Stutong": (1.51440, 110.36280),
                    "Lorong 7B Stutong": (1.51620, 110.36090),
                    "CityOne Link Road": (1.51890, 110.35520),
                    "Taman Sri Stutong": (1.51830, 110.36300),
                },
                origin_name="Vivacity Megamall",
                destination_names=["Saradise Kuching", "KPJ Kuching Specialist Hospital"],
                connections=[
                    ("Vivacity Megamall", "Saradise Kuching"),
                    ("Saradise Kuching", "Vivacity Megamall"),
                    ("Vivacity Megamall", "Jalan Setia Raja Junction"),
                    ("Jalan Setia Raja Junction", "Vivacity Megamall"),
                    ("Vivacity Megamall", "CityOne Link Road"),
                    ("CityOne Link Road", "Vivacity Megamall"),
                    ("Vivacity Megamall", "Taman Sri Stutong"),
                    ("Taman Sri Stutong", "Vivacity Megamall"),
                    ("Saradise Kuching", "Stutong Market"),
                    ("Stutong Market", "Saradise Kuching"),
                    ("Saradise Kuching", "One Residency Stutong"),
                    ("One Residency Stutong", "Saradise Kuching"),
                    ("Jalan Setia Raja Junction", "Stutong Market"),
                    ("Stutong Market", "Jalan Setia Raja Junction"),
                    ("Jalan Setia Raja Junction", "CityOne Link Road"),
                    ("CityOne Link Road", "Jalan Setia Raja Junction"),
                    ("Jalan Setia Raja Junction", "Taman Sri Stutong"),
                    ("Taman Sri Stutong", "Jalan Setia Raja Junction"),
                    ("Stutong Market", "Stutong Forest Park"),
                    ("Stutong Forest Park", "Stutong Market"),
                    ("Stutong Market", "One Residency Stutong"),
                    ("One Residency Stutong", "Stutong Market"),
                    ("Stutong Market", "Lorong 7B Stutong"),
                    ("Lorong 7B Stutong", "Stutong Market"),
                    ("Stutong Forest Park", "KPJ Kuching Specialist Hospital"),
                    ("KPJ Kuching Specialist Hospital", "Stutong Forest Park"),
                    ("Stutong Forest Park", "One Residency Stutong"),
                    ("One Residency Stutong", "Stutong Forest Park"),
                    ("One Residency Stutong", "Lorong 7B Stutong"),
                    ("Lorong 7B Stutong", "One Residency Stutong"),
                    ("Lorong 7B Stutong", "Taman Sri Stutong"),
                    ("Taman Sri Stutong", "Lorong 7B Stutong"),
                    ("CityOne Link Road", "KPJ Kuching Specialist Hospital"),
                    ("KPJ Kuching Specialist Hospital", "CityOne Link Road"),
                    ("CityOne Link Road", "Taman Sri Stutong"),
                    ("Taman Sri Stutong", "CityOne Link Road"),
                    ("Taman Sri Stutong", "KPJ Kuching Specialist Hospital"),
                    ("KPJ Kuching Specialist Hospital", "Taman Sri Stutong"),
                ],
                accident_connection=("Stutong Market", "Jalan Setia Raja Junction"),
                accident_severity="Severe",
                accident_multiplier=1.7,
                zoom=16,
            ),
        ]

        for spec in specs:
            dest = output_dir / f"{spec.name}.txt"
            build_osm_map(base_graph, spec, dest)


if __name__ == "__main__":
    main()
