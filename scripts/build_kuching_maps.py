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
                accident_connection=None,
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
                accident_connection=None,
                zoom=17,
            ),
            OSMSpec(
                name="kuching_padungan_waterfront",
                bbox=(1.5550, 1.5585, 110.3420, 110.3485),
                landmarks={
                    "Kuching Waterfront": (1.55749, 110.34428),
                    "Main Bazaar": (1.55701, 110.34520),
                    "Carpenter Street": (1.55655, 110.34428),
                    "Plaza Merdeka": (1.55690, 110.34360),
                    "India Street": (1.55680, 110.34400),
                    "Tua Pek Kong Temple": (1.55712, 110.34483),
                    "Darul Hana Bridge": (1.55728, 110.34752),
                    "Electra House": (1.55670, 110.34470),
                },
                origin_name="Kuching Waterfront",
                destination_names=["Darul Hana Bridge", "Main Bazaar"],
                connections=[
                    ("Kuching Waterfront", "Main Bazaar"),
                    ("Main Bazaar", "Kuching Waterfront"),
                    ("Main Bazaar", "Carpenter Street"),
                    ("Carpenter Street", "Main Bazaar"),
                    ("Carpenter Street", "Plaza Merdeka"),
                    ("Plaza Merdeka", "Carpenter Street"),
                    ("Plaza Merdeka", "India Street"),
                    ("India Street", "Plaza Merdeka"),
                    ("India Street", "Tua Pek Kong Temple"),
                    ("Tua Pek Kong Temple", "India Street"),
                    ("Tua Pek Kong Temple", "Darul Hana Bridge"),
                    ("Darul Hana Bridge", "Tua Pek Kong Temple"),
                    ("Kuching Waterfront", "Electra House"),
                    ("Electra House", "Kuching Waterfront"),
                    ("Electra House", "India Street"),
                    ("India Street", "Electra House"),
                    ("Main Bazaar", "Tua Pek Kong Temple"),
                    ("Tua Pek Kong Temple", "Main Bazaar"),
                ],
                accident_connection=None,
                zoom=17,
            ),
            OSMSpec(
                name="kuching_petra_jaya",
                bbox=(1.5710, 1.5820, 110.3280, 110.3445),
                landmarks={
                    "Sarawak State Library": (1.57710, 110.33822),
                    "Masjid Jamek Petra Jaya": (1.57428, 110.33640),
                    "Petra Jaya Sports Complex": (1.57680, 110.33480),
                    "Mini Garden Petra Jaya": (1.57320, 110.33790),
                    "Sarawak State Assembly": (1.57290, 110.34290),
                    "Normah Medical Centre": (1.57920, 110.32940),
                    "PETRONAS Jalan Semariang": (1.58074, 110.33226),
                    "Hawa BBQ Steamboat House": (1.57659, 110.33166),
                },
                origin_name="Sarawak State Library",
                destination_names=["Normah Medical Centre", "Sarawak State Assembly"],
                connections=[
                    ("Sarawak State Library", "Masjid Jamek Petra Jaya"),
                    ("Masjid Jamek Petra Jaya", "Sarawak State Library"),
                    ("Masjid Jamek Petra Jaya", "Petra Jaya Sports Complex"),
                    ("Petra Jaya Sports Complex", "Masjid Jamek Petra Jaya"),
                    ("Petra Jaya Sports Complex", "Hawa BBQ Steamboat House"),
                    ("Hawa BBQ Steamboat House", "Petra Jaya Sports Complex"),
                    ("Hawa BBQ Steamboat House", "PETRONAS Jalan Semariang"),
                    ("PETRONAS Jalan Semariang", "Hawa BBQ Steamboat House"),
                    ("PETRONAS Jalan Semariang", "Normah Medical Centre"),
                    ("Normah Medical Centre", "PETRONAS Jalan Semariang"),
                    ("Hawa BBQ Steamboat House", "Normah Medical Centre"),
                    ("Normah Medical Centre", "Hawa BBQ Steamboat House"),
                    ("Petra Jaya Sports Complex", "Mini Garden Petra Jaya"),
                    ("Mini Garden Petra Jaya", "Petra Jaya Sports Complex"),
                    ("Mini Garden Petra Jaya", "Sarawak State Assembly"),
                    ("Sarawak State Assembly", "Mini Garden Petra Jaya"),
                    ("Sarawak State Library", "Mini Garden Petra Jaya"),
                    ("Mini Garden Petra Jaya", "Sarawak State Library"),
                ],
                accident_connection=None,
                zoom=16,
            ),
            OSMSpec(
                name="kuching_pending_industrial",
                bbox=(1.5580, 1.5630, 110.3870, 110.3965),
                landmarks={
                    "Pending Industrial Gate": (1.55880, 110.38980),
                    "Jalan Pending Junction": (1.55980, 110.39200),
                    "Pending Port Roundabout": (1.56100, 110.39380),
                    "Senari Terminal": (1.56280, 110.39550),
                    "Pending Market": (1.56200, 110.39050),
                    "Kampung Tabuan Hilir Access": (1.56060, 110.38800),
                },
                origin_name="Pending Industrial Gate",
                destination_names=["Senari Terminal", "Pending Market"],
                connections=[
                    ("Pending Industrial Gate", "Jalan Pending Junction"),
                    ("Jalan Pending Junction", "Pending Industrial Gate"),
                    ("Jalan Pending Junction", "Pending Port Roundabout"),
                    ("Pending Port Roundabout", "Jalan Pending Junction"),
                    ("Pending Port Roundabout", "Senari Terminal"),
                    ("Senari Terminal", "Pending Port Roundabout"),
                    ("Jalan Pending Junction", "Pending Market"),
                    ("Pending Market", "Jalan Pending Junction"),
                    ("Pending Industrial Gate", "Kampung Tabuan Hilir Access"),
                    ("Kampung Tabuan Hilir Access", "Pending Industrial Gate"),
                    ("Kampung Tabuan Hilir Access", "Pending Market"),
                    ("Pending Market", "Kampung Tabuan Hilir Access"),
                ],
                accident_connection=None,
                zoom=15,
            ),
            OSMSpec(
                name="kuching_airport_corridor",
                bbox=(1.4680, 1.4935, 110.3240, 110.3500),
                landmarks={
                    "Kuching International Airport": (1.4873996468695103, 110.34177081614432),
                    "Raia Hotel & Convention Centre": (1.4905402301555795, 110.33995920924441),
                    "Jalan Liu Shan Bang Junction": (1.47820, 110.33780),
                    "Sarawak Forestry Corporation": (1.4727491562706154, 110.33532680720366),
                    "Farley Kuching": (1.484422037459582, 110.3329256689021),
                    "Big Canteen Food Court": (1.4825888651134826, 110.33056468161541),
                    "PETRONAS Batu 7 Jalan Penrissen": (1.4734832305104373, 110.32827718683153),
                },
                origin_name="Kuching International Airport",
                destination_names=[
                    "Sarawak Forestry Corporation",
                    "Farley Kuching",
                ],
                connections=[
                    ("Kuching International Airport", "Raia Hotel & Convention Centre"),
                    ("Raia Hotel & Convention Centre", "Kuching International Airport"),
                    ("Raia Hotel & Convention Centre", "Jalan Liu Shan Bang Junction"),
                    ("Jalan Liu Shan Bang Junction", "Raia Hotel & Convention Centre"),
                    ("Jalan Liu Shan Bang Junction", "Sarawak Forestry Corporation"),
                    ("Sarawak Forestry Corporation", "Jalan Liu Shan Bang Junction"),
                    ("Sarawak Forestry Corporation", "PETRONAS Batu 7 Jalan Penrissen"),
                    ("PETRONAS Batu 7 Jalan Penrissen", "Sarawak Forestry Corporation"),
                    ("Sarawak Forestry Corporation", "Big Canteen Food Court"),
                    ("Big Canteen Food Court", "Farley Kuching"),
                    ("Farley Kuching", "Kuching International Airport"),
                    ("Big Canteen Food Court", "Kuching International Airport"),
                ],
                accident_connection=None,
                zoom=15,
            ),
            OSMSpec(
                name="kuching_batu_kawa",
                bbox=(1.5040, 1.5125, 110.3040, 110.3210),
                landmarks={
                    "MJC Batu Kawa": (1.50820, 110.31090),
                    "Batu Kawa Old Town": (1.50560, 110.31170),
                    "Batu Kawa Bridge": (1.50640, 110.30760),
                    "Jalan Stapok Junction": (1.51030, 110.30470),
                    "Moyan Square": (1.50810, 110.31690),
                    "Kuching City Mall": (1.51050, 110.32010),
                },
                origin_name="MJC Batu Kawa",
                destination_names=["Kuching City Mall", "Batu Kawa Bridge"],
                connections=[
                    ("MJC Batu Kawa", "Batu Kawa Old Town"),
                    ("Batu Kawa Old Town", "MJC Batu Kawa"),
                    ("Batu Kawa Old Town", "Batu Kawa Bridge"),
                    ("Batu Kawa Bridge", "Batu Kawa Old Town"),
                    ("Batu Kawa Bridge", "Jalan Stapok Junction"),
                    ("Jalan Stapok Junction", "Batu Kawa Bridge"),
                    ("MJC Batu Kawa", "Moyan Square"),
                    ("Moyan Square", "MJC Batu Kawa"),
                    ("Moyan Square", "Kuching City Mall"),
                    ("Kuching City Mall", "Moyan Square"),
                    ("Jalan Stapok Junction", "Kuching City Mall"),
                    ("Kuching City Mall", "Jalan Stapok Junction"),
                ],
                accident_connection=None,
                zoom=16,
            ),
            OSMSpec(
                name="kuching_saradise_stutong",
                bbox=(1.5110, 1.5215, 110.3530, 110.3665),
                landmarks={
                    "Vivacity Megamall": (1.52020, 110.35370),
                    "CityOne Link": (1.51890, 110.35520),
                    "Saradise": (1.51280, 110.35910),
                    "Stutong Market": (1.51570, 110.35850),
                    "Lorong 7B Stutong": (1.51620, 110.36090),
                    "Stutong Forest Park": (1.51210, 110.36340),
                    "KPJ Kuching Specialist Hospital": (1.51500, 110.36610),
                    "Taman Sri Stutong": (1.51830, 110.36300),
                },
                origin_name="Vivacity Megamall",
                destination_names=["Saradise", "KPJ Kuching Specialist Hospital"],
                connections=[
                    ("Vivacity Megamall", "CityOne Link"),
                    ("CityOne Link", "Vivacity Megamall"),
                    ("CityOne Link", "Taman Sri Stutong"),
                    ("Taman Sri Stutong", "CityOne Link"),
                    ("Taman Sri Stutong", "KPJ Kuching Specialist Hospital"),
                    ("KPJ Kuching Specialist Hospital", "Taman Sri Stutong"),
                    ("Vivacity Megamall", "Stutong Market"),
                    ("Stutong Market", "Vivacity Megamall"),
                    ("Stutong Market", "Lorong 7B Stutong"),
                    ("Lorong 7B Stutong", "Stutong Market"),
                    ("Lorong 7B Stutong", "Saradise"),
                    ("Saradise", "Lorong 7B Stutong"),
                    ("Saradise", "Stutong Forest Park"),
                    ("Stutong Forest Park", "Saradise"),
                ],
                accident_connection=None,
                zoom=16,
            ),
            OSMSpec(
                name="kuching_matang_kubah",
                bbox=(1.5680, 1.5800, 110.2650, 110.3055),
                landmarks={
                    "Kubah National Park Entrance": (1.57424, 110.26633),
                    "Matang Wildlife Centre": (1.57205, 110.26810),
                    "Jalan Matang West": (1.57320, 110.27290),
                    "Taman Sri Matang": (1.57395, 110.27650),
                    "Jalan Matang Mall": (1.57395, 110.27810),
                    "Emart Matang": (1.57200, 110.30321),
                    "Matang Clinic": (1.57429, 110.29714),
                    "Matang Jaya Commercial Centre": (1.57530, 110.29398),
                    "Matang Avenue": (1.57619, 110.28040),
                    "Matang Hospital": (1.57665, 110.28341),
                },
                origin_name="Matang Jaya Commercial Centre",
                destination_names=["Emart Matang", "Matang Hospital"],
                connections=[
                    ("Kubah National Park Entrance", "Matang Wildlife Centre"),
                    ("Matang Wildlife Centre", "Kubah National Park Entrance"),
                    ("Matang Wildlife Centre", "Jalan Matang West"),
                    ("Jalan Matang West", "Matang Wildlife Centre"),
                    ("Jalan Matang West", "Taman Sri Matang"),
                    ("Taman Sri Matang", "Jalan Matang West"),
                    ("Taman Sri Matang", "Jalan Matang Mall"),
                    ("Jalan Matang Mall", "Taman Sri Matang"),
                    ("Jalan Matang Mall", "Emart Matang"),
                    ("Emart Matang", "Jalan Matang Mall"),
                    ("Emart Matang", "Matang Clinic"),
                    ("Matang Clinic", "Matang Jaya Commercial Centre"),
                    ("Matang Jaya Commercial Centre", "Matang Hospital"),
                    ("Matang Hospital", "Matang Avenue"),
                    ("Matang Avenue", "Matang Hospital"),
                    ("Emart Matang", "Matang Avenue"),
                    ("Matang Avenue", "Emart Matang"),
                    ("Taman Sri Matang", "Matang Avenue"),
                    ("Matang Avenue", "Taman Sri Matang"),
                ],
                accident_connection=None,
                zoom=16,
            )
        ]

        for spec in specs:
            dest = output_dir / f"{spec.name}.txt"
            build_osm_map(base_graph, spec, dest)


if __name__ == "__main__":
    main()
