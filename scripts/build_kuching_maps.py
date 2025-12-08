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

FALLBACK_SPEED_KMPH = 35.0


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
    accident_connection: Tuple[str, str] | None = None
    accident_severity: str = ""
    accident_multiplier: float = 1.0
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


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns the great-circle distance between two lat/lon pairs in kilometers.
    """
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


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
    if weighted_graph.number_of_nodes() == 0:
        raise ValueError("Weighted graph is empty.")

    largest_component = max(
        nx.connected_components(weighted_graph.to_undirected()), key=len
    )
    component_nodes = set(largest_component)

    for name, node_id in list(name_to_node.items()):
        if node_id not in component_nodes:
            lat, lon = spec.landmarks.get(name, (None, None))
            if lat is None or lon is None:
                continue
            best = None
            best_dist = float("inf")
            for candidate in component_nodes:
                cx = G.nodes[candidate].get("x")
                cy = G.nodes[candidate].get("y")
                if cx is None or cy is None:
                    continue
                dist = (cx - lon) ** 2 + (cy - lat) ** 2
                if dist < best_dist:
                    best = candidate
                    best_dist = dist
            if best is not None:
                name_to_node[name] = best

    undirected_weighted = weighted_graph.to_undirected()

    for name_a, name_b in spec.connections:
        if name_a not in name_to_newid or name_b not in name_to_newid:
            print(f"[warn] Connection uses unknown landmark ({name_a} -> {name_b})")
            continue
        origin_node = name_to_node.get(name_a)
        target_node = name_to_node.get(name_b)
        if origin_node is None or target_node is None:
            print(f"[warn] Unable to map connection ({name_a} -> {name_b}) to OSM nodes")
            continue
        new_u = name_to_newid[name_a]
        new_v = name_to_newid[name_b]
        path_nodes = None
        try:
            path_nodes = nx.shortest_path(
                weighted_graph, origin_node, target_node, weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            try:
                path_nodes = nx.shortest_path(
                    undirected_weighted, origin_node, target_node, weight="weight"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                print(f"[warn] No path between {name_a} and {name_b}. Falling back to straight-line edge.")
                lon_a, lat_a = nodes.get(new_u, (None, None))
                lon_b, lat_b = nodes.get(new_v, (None, None))
                if None in (lon_a, lat_a, lon_b, lat_b):
                    continue
                distance_km = haversine_distance(lat_a, lon_a, lat_b, lon_b)
                if distance_km <= 0:
                    distance_km = 0.05
                cost_hours = max(distance_km / FALLBACK_SPEED_KMPH, 0.005)
                edges[(new_u, new_v)] = cost_hours
                polylines[(new_u, new_v)] = [(lon_a, lat_a), (lon_b, lat_b)]
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

    accident_edge: Tuple[int, int] | None = None
    accident_base: float | None = None
    if spec.accident_connection:
        try:
            accident_edge = tuple(resolve_newid(name) for name in spec.accident_connection)  # type: ignore[arg-type]
        except ValueError:
            accident_edge = None
        else:
            accident_base = edges.get(accident_edge)
            if accident_base is None and edges:
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
                name="kuching_green_heights",
                bbox=(1.4980, 1.5155, 110.3420, 110.3585),
                landmarks={
                    "Premier Food Republic": (1.503600, 110.349496),
                    "Sin Chong Choon Cafe": (1.502671, 110.347235),
                    "Shell Green Heights": (1.506678, 110.350221),
                    "Taman BDC Park": (1.503737, 110.356130),
                    "Petronas Green Heights": (1.501908, 110.344762),
                    "Green Heights Park": (1.500698, 110.346407),
                    "Maguro Buffet": (1.511854, 110.352739),
                    "TalentSeed Daycare": (1.509690, 110.354338),
                    "Hong Leong Bank": (1.503458, 110.351430),
                    "BIG Pharmacy Hui Sing": (1.513580, 110.344643),
                    "Borneoartifact Gallery": (1.512456, 110.347041),
                },
                origin_name="Premier Food Republic",
                destination_names=["Taman BDC Park", "Maguro Buffet"],
                connections=[
                    ("Premier Food Republic", "Sin Chong Choon Cafe"),
                    ("Sin Chong Choon Cafe", "Premier Food Republic"),
                    ("Sin Chong Choon Cafe", "Shell Green Heights"),
                    ("Shell Green Heights", "Sin Chong Choon Cafe"),
                    ("Shell Green Heights", "Hong Leong Bank"),
                    ("Hong Leong Bank", "Shell Green Heights"),
                    ("Hong Leong Bank", "Taman BDC Park"),
                    ("Taman BDC Park", "Hong Leong Bank"),
                    ("Premier Food Republic", "Petronas Green Heights"),
                    ("Petronas Green Heights", "Premier Food Republic"),
                    ("Petronas Green Heights", "Green Heights Park"),
                    ("Green Heights Park", "Petronas Green Heights"),
                    ("Green Heights Park", "Sin Chong Choon Cafe"),
                    ("Premier Food Republic", "Hong Leong Bank"),
                    ("Hong Leong Bank", "Premier Food Republic"),
                    ("Taman BDC Park", "TalentSeed Daycare"),
                    ("TalentSeed Daycare", "Taman BDC Park"),
                    ("TalentSeed Daycare", "Maguro Buffet"),
                    ("Maguro Buffet", "TalentSeed Daycare"),
                    ("Maguro Buffet", "BIG Pharmacy Hui Sing"),
                    ("BIG Pharmacy Hui Sing", "Maguro Buffet"),
                    ("BIG Pharmacy Hui Sing", "Borneoartifact Gallery"),
                    ("Borneoartifact Gallery", "BIG Pharmacy Hui Sing"),
                    ("Borneoartifact Gallery", "Maguro Buffet"),
                    ("Maguro Buffet", "Borneoartifact Gallery"),
                    ("Shell Green Heights", "Maguro Buffet"),
                    ("Maguro Buffet", "Shell Green Heights"),
                ],
                accident_connection=None,
                zoom=17,
            ),
            OSMSpec(
                name="kuching_swinburne_corridor",
                bbox=(1.5200, 1.5450, 110.3500, 110.3750),
                landmarks={
                    "Swinburne Sarawak": (1.532751, 110.357184),
                    "Kuching District Police HQ": (1.534657, 110.359207),
                    "The Spring Mall": (1.535765, 110.358653),
                    "Box Chicken Kopitiam": (1.530439, 110.357355),
                    "Arena Sukan": (1.529505, 110.362212),
                    "Emart King Centre": (1.527860, 110.360513),
                    "Citadines Uplands": (1.536088, 110.356089),
                    "56 Hotel": (1.526570, 110.357605),
                    "Borneo Medical Centre": (1.529235, 110.357657),
                },
                origin_name="Swinburne Sarawak",
                destination_names=["The Spring Mall", "Citadines Uplands"],
                connections=[
                    ("Swinburne Sarawak", "Kuching District Police HQ"),
                    ("Kuching District Police HQ", "Swinburne Sarawak"),
                    ("The Spring Mall", "Swinburne Sarawak"),
                    ("Swinburne Sarawak", "The Spring Mall"),
                    ("Citadines Uplands", "Box Chicken Kopitiam"),
                    ("Box Chicken Kopitiam", "Citadines Uplands"),
                    ("Box Chicken Kopitiam", "Borneo Medical Centre"),
                    ("Borneo Medical Centre", "Box Chicken Kopitiam"),
                    ("Borneo Medical Centre", "The Spring Mall"),
                    ("The Spring Mall", "Borneo Medical Centre"),
                    ("Borneo Medical Centre", "56 Hotel"),
                    ("56 Hotel", "Borneo Medical Centre"),
                    ("56 Hotel", "Emart King Centre"),
                    ("Emart King Centre", "56 Hotel"),
                    ("Emart King Centre", "Arena Sukan"),
                    ("Arena Sukan", "Emart King Centre"),
                    ("Arena Sukan", "Swinburne Sarawak"),
                    ("Swinburne Sarawak", "Arena Sukan"),
                    ("Kuching District Police HQ", "Borneo Medical Centre"),
                    ("Borneo Medical Centre", "Kuching District Police HQ"),
                ],
                accident_connection=None,
                zoom=17,
            ),
            OSMSpec(
                name="kuching_petra_jaya",
                bbox=(1.5680, 1.5850, 110.3240, 110.3480),
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
                bbox=(1.5440, 1.5695, 110.3840, 110.3995),
                landmarks={
                    "Kuching Port Authority": (1.555435, 110.395463),
                    "Penview Hotel": (1.550949, 110.387003),
                    "Stadium MBKS": (1.564502, 110.387954),
                    "See Hua Daily News": (1.566310, 110.389892),
                    "Shin Yang Shipping": (1.550196, 110.391960),
                    "Kilang Borneo": (1.547688, 110.390595),
                    "Rajah Cafe": (1.552429, 110.385883),
                    "Peaches Garden Food Court": (1.549655, 110.386044),
                    "Padang Futsal Pending": (1.560839, 110.391350),
                    "Kuching Frozen Food": (1.559153, 110.386440),
                    "Bintawa Police Station": (1.566798, 110.387141),
                    "Hong Yong Seafood": (1.551156, 110.391618),
                },
                origin_name="Penview Hotel",
                destination_names=["Kuching Port Authority", "Stadium MBKS"],
                connections=[
                    ("Penview Hotel", "Rajah Cafe"),
                    ("Rajah Cafe", "Penview Hotel"),
                    ("Rajah Cafe", "Peaches Garden Food Court"),
                    ("Peaches Garden Food Court", "Rajah Cafe"),
                    ("Penview Hotel", "Peaches Garden Food Court"),
                    ("Peaches Garden Food Court", "Penview Hotel"),
                    ("Peaches Garden Food Court", "Shin Yang Shipping"),
                    ("Shin Yang Shipping", "Peaches Garden Food Court"),
                    ("Shin Yang Shipping", "Hong Yong Seafood"),
                    ("Hong Yong Seafood", "Shin Yang Shipping"),
                    ("Hong Yong Seafood", "Kuching Frozen Food"),
                    ("Kuching Frozen Food", "Hong Yong Seafood"),
                    ("Hong Yong Seafood", "Kuching Port Authority"),
                    ("Kuching Port Authority", "Hong Yong Seafood"),
                    ("Penview Hotel", "Kuching Frozen Food"),
                    ("Kuching Frozen Food", "Penview Hotel"),
                    ("Penview Hotel", "Shin Yang Shipping"),
                    ("Shin Yang Shipping", "Penview Hotel"),
                    ("Rajah Cafe", "Kilang Borneo"),
                    ("Kilang Borneo", "Rajah Cafe"),
                    ("Kilang Borneo", "Kuching Frozen Food"),
                    ("Kuching Frozen Food", "Kilang Borneo"),
                    ("Peaches Garden Food Court", "Kilang Borneo"),
                    ("Kilang Borneo", "Peaches Garden Food Court"),
                    ("Kuching Port Authority", "Padang Futsal Pending"),
                    ("Padang Futsal Pending", "Kuching Port Authority"),
                    ("Padang Futsal Pending", "Stadium MBKS"),
                    ("Stadium MBKS", "Padang Futsal Pending"),
                    ("Stadium MBKS", "See Hua Daily News"),
                    ("See Hua Daily News", "Stadium MBKS"),
                    ("See Hua Daily News", "Bintawa Police Station"),
                    ("Bintawa Police Station", "See Hua Daily News"),
                    ("Bintawa Police Station", "Kuching Port Authority"),
                    ("Kuching Port Authority", "Bintawa Police Station"),
                    ("Bintawa Police Station", "Padang Futsal Pending"),
                    ("Padang Futsal Pending", "Bintawa Police Station"),
                    ("Rajah Cafe", "Stadium MBKS"),
                    ("Stadium MBKS", "Rajah Cafe"),
                ],
                accident_connection=None,
                zoom=15,
            ),
            OSMSpec(
                name="kuching_sheraton_cbd",
                bbox=(1.5490, 1.5595, 110.3475, 110.3615),
                landmarks={
                    "Sheraton Kuching Hotel": (1.556628, 110.353365),
                    "Pullman Kuching": (1.555859, 110.351316),
                    "Ceylonese Restaurant": (1.556366, 110.349069),
                    "Little Hainan": (1.554402, 110.357430),
                    "Cat Statue": (1.557788, 110.352831),
                    "Song Kheng Hai Field": (1.554950, 110.354822),
                    "Mita Cake House": (1.553268, 110.354487),
                    "Sharing Downtown": (1.553760, 110.351660),
                    "Lok-Lok Ban Hock": (1.553366, 110.349226),
                    "Chinese Temple": (1.551212, 110.354270),
                    "St. Peter's Church": (1.551979, 110.358787),
                    "Shell Ban Hock": (1.553080, 110.360432),
                    "RHB Bank Yung Kong": (1.555506, 110.357640),
                },
                origin_name="Sheraton Kuching Hotel",
                destination_names=["Cat Statue", "Little Hainan"],
                connections=[
                    ("Sheraton Kuching Hotel", "Pullman Kuching"),
                    ("Pullman Kuching", "Sheraton Kuching Hotel"),
                    ("Pullman Kuching", "Ceylonese Restaurant"),
                    ("Ceylonese Restaurant", "Pullman Kuching"),
                    ("Ceylonese Restaurant", "Lok-Lok Ban Hock"),
                    ("Lok-Lok Ban Hock", "Ceylonese Restaurant"),
                    ("Lok-Lok Ban Hock", "Chinese Temple"),
                    ("Chinese Temple", "Lok-Lok Ban Hock"),
                    ("Chinese Temple", "St. Peter's Church"),
                    ("St. Peter's Church", "Chinese Temple"),
                    ("St. Peter's Church", "Shell Ban Hock"),
                    ("Shell Ban Hock", "St. Peter's Church"),
                    ("Shell Ban Hock", "RHB Bank Yung Kong"),
                    ("RHB Bank Yung Kong", "Shell Ban Hock"),
                    ("RHB Bank Yung Kong", "Little Hainan"),
                    ("Little Hainan", "RHB Bank Yung Kong"),
                    ("Little Hainan", "Song Kheng Hai Field"),
                    ("Song Kheng Hai Field", "Little Hainan"),
                    ("Song Kheng Hai Field", "Mita Cake House"),
                    ("Mita Cake House", "Song Kheng Hai Field"),
                    ("Mita Cake House", "Sharing Downtown"),
                    ("Sharing Downtown", "Mita Cake House"),
                    ("Sharing Downtown", "Pullman Kuching"),
                    ("Pullman Kuching", "Sharing Downtown"),
                    ("Sheraton Kuching Hotel", "Cat Statue"),
                    ("Cat Statue", "Sheraton Kuching Hotel"),
                ],
                accident_connection=None,
                zoom=17,
            ),
            OSMSpec(
                name="kuching_airport_corridor",
                bbox=(1.4580, 1.5035, 110.3150, 110.3600),
                landmarks={
                    "Kuching International Airport": (1.487399, 110.341770),
                    "Raia Hotel & Convention Centre": (1.490540, 110.339959),
                    "Jalan Liu Shan Bang Junction": (1.47820, 110.33780),
                    "Sarawak Forestry Corporation": (1.472749, 110.335326),
                    "Farley Kuching": (1.484422, 110.332925),
                    "Big Canteen Food Court": (1.482588, 110.330564),
                    "PETRONAS Batu 7 Jalan Penrissen": (1.473483, 110.328277),
                    "Public Bank": (1.475815, 110.330860),
                    "Affin Bank": (1.482142, 110.332108),
                    "Sacred Heart Catholic Church": (1.472928, 110.329862),
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
                    ("Big Canteen Food Court", "Affin Bank"),
                    ("Affin Bank", "Big Canteen Food Court"),
                    ("Affin Bank", "Farley Kuching"),
                    ("Farley Kuching", "Affin Bank"),
                    ("Kuching International Airport", "Affin Bank"),
                    ("Affin Bank", "Kuching International Airport"),
                    ("Jalan Liu Shan Bang Junction", "Public Bank"),
                    ("Public Bank", "Jalan Liu Shan Bang Junction"),
                    ("Public Bank", "Sacred Heart Catholic Church"),
                    ("Sacred Heart Catholic Church", "Public Bank"),
                    ("Public Bank", "Sarawak Forestry Corporation"),
                    ("Sarawak Forestry Corporation", "Public Bank"),
                    ("Sacred Heart Catholic Church", "Sarawak Forestry Corporation"),
                    ("Sarawak Forestry Corporation", "Sacred Heart Catholic Church"),
                    ("Public Bank", "Affin Bank"),
                    ("Affin Bank", "Public Bank"),
                    ("Affin Bank", "Sarawak Forestry Corporation"),
                    ("Sarawak Forestry Corporation", "Affin Bank"),
                    ("Affin Bank", "Jalan Liu Shan Bang Junction"),
                    ("Jalan Liu Shan Bang Junction", "Affin Bank"),
                ],
                accident_connection=None,
                zoom=15,
            ),
            OSMSpec(
                name="kuching_batu_kawa",
                bbox=(1.5015, 1.5192, 110.2945, 110.3250),
                landmarks={
                    "ZUS Coffee - Pines Square, Kuching": (1.511019, 110.307922),
                    "PETRONAS Batu Kawah": (1.513879, 110.312037),
                    "Emart Batu Kawa": (1.507132, 110.299509),
                    "29 Taman Botanika Batu Kawa": (1.510205, 110.299725),
                    "McDonald's Batu Kawa DT": (1.511574, 110.304945),
                    "RICE GARDEN by OYES FOOD CORNER": (1.515164, 110.308372),
                    "PINES SQUARE": (1.514101, 110.316444),
                    "JJ Pet Shop Kuching @MJC Batu Kawa": (1.516490, 110.311947),
                    "LED Word Marketing Sdn Bhd": (1.510402, 110.314951),
                },
                origin_name="Emart Batu Kawa",
                destination_names=["PINES SQUARE", "PETRONAS Batu Kawah"],
                connections=[
                    ("Emart Batu Kawa", "29 Taman Botanika Batu Kawa"),
                    ("29 Taman Botanika Batu Kawa", "Emart Batu Kawa"),
                    ("29 Taman Botanika Batu Kawa", "ZUS Coffee - Pines Square, Kuching"),
                    ("ZUS Coffee - Pines Square, Kuching", "29 Taman Botanika Batu Kawa"),
                    ("ZUS Coffee - Pines Square, Kuching", "McDonald's Batu Kawa DT"),
                    ("McDonald's Batu Kawa DT", "ZUS Coffee - Pines Square, Kuching"),
                    ("McDonald's Batu Kawa DT", "RICE GARDEN by OYES FOOD CORNER"),
                    ("RICE GARDEN by OYES FOOD CORNER", "McDonald's Batu Kawa DT"),
                    ("RICE GARDEN by OYES FOOD CORNER", "PETRONAS Batu Kawah"),
                    ("PETRONAS Batu Kawah", "RICE GARDEN by OYES FOOD CORNER"),
                    ("PETRONAS Batu Kawah", "JJ Pet Shop Kuching @MJC Batu Kawa"),
                    ("JJ Pet Shop Kuching @MJC Batu Kawa", "PETRONAS Batu Kawah"),
                    ("JJ Pet Shop Kuching @MJC Batu Kawa", "PINES SQUARE"),
                    ("PINES SQUARE", "JJ Pet Shop Kuching @MJC Batu Kawa"),
                    ("PINES SQUARE", "LED Word Marketing Sdn Bhd"),
                    ("LED Word Marketing Sdn Bhd", "PINES SQUARE"),
                    ("LED Word Marketing Sdn Bhd", "PETRONAS Batu Kawah"),
                    ("PETRONAS Batu Kawah", "LED Word Marketing Sdn Bhd"),
                    ("PETRONAS Batu Kawah", "ZUS Coffee - Pines Square, Kuching"),
                    ("ZUS Coffee - Pines Square, Kuching", "PETRONAS Batu Kawah"),
                    ("RICE GARDEN by OYES FOOD CORNER", "PINES SQUARE"),
                    ("PINES SQUARE", "RICE GARDEN by OYES FOOD CORNER"),
                    ("McDonald's Batu Kawa DT", "LED Word Marketing Sdn Bhd"),
                    ("LED Word Marketing Sdn Bhd", "McDonald's Batu Kawa DT"),
                    ("Emart Batu Kawa", "McDonald's Batu Kawa DT"),
                    ("McDonald's Batu Kawa DT", "Emart Batu Kawa"),
                ],
                accident_connection=None,
                zoom=16,
            ),
            OSMSpec(
                name="kuching_saradise_stutong",
                bbox=(1.5030, 1.5115, 110.3570, 110.3685),
                landmarks={
                    "KPJ Kuching Specialist Hospital": (1.506610, 110.365838),
                    "Sushi Tie Saradise": (1.504274, 110.359564),
                    "MySmile Dental Clinic": (1.505336, 110.358682),
                    "Blendsmiths Cafe": (1.506057, 110.361309),
                    "Rice King Saradise": (1.504977, 110.362546),
                    "Level Up Fitness BDC": (1.507035, 110.360995),
                    "Medsave Pharmacy": (1.507726, 110.362277),
                    "MSU College Sarawak": (1.509329, 110.361443),
                    "Masjid Darul Barakah": (1.508814, 110.360326),
                    "BDC Football Field": (1.507554, 110.364603),
                },
                origin_name="Sushi Tie Saradise",
                destination_names=["KPJ Kuching Specialist Hospital", "Masjid Darul Barakah"],
                connections=[
                    ("Sushi Tie Saradise", "MySmile Dental Clinic"),
                    ("MySmile Dental Clinic", "Sushi Tie Saradise"),
                    ("MySmile Dental Clinic", "Blendsmiths Cafe"),
                    ("Blendsmiths Cafe", "MySmile Dental Clinic"),
                    ("Blendsmiths Cafe", "Rice King Saradise"),
                    ("Rice King Saradise", "Blendsmiths Cafe"),
                    ("Rice King Saradise", "Level Up Fitness BDC"),
                    ("Level Up Fitness BDC", "Rice King Saradise"),
                    ("Level Up Fitness BDC", "Medsave Pharmacy"),
                    ("Medsave Pharmacy", "Level Up Fitness BDC"),
                    ("Medsave Pharmacy", "MSU College Sarawak"),
                    ("MSU College Sarawak", "Medsave Pharmacy"),
                    ("MSU College Sarawak", "Masjid Darul Barakah"),
                    ("Masjid Darul Barakah", "MSU College Sarawak"),
                    ("Masjid Darul Barakah", "BDC Football Field"),
                    ("BDC Football Field", "Masjid Darul Barakah"),
                    ("BDC Football Field", "KPJ Kuching Specialist Hospital"),
                    ("KPJ Kuching Specialist Hospital", "BDC Football Field"),
                    ("KPJ Kuching Specialist Hospital", "Medsave Pharmacy"),
                    ("Medsave Pharmacy", "KPJ Kuching Specialist Hospital"),
                    ("Blendsmiths Cafe", "Level Up Fitness BDC"),
                    ("Level Up Fitness BDC", "Blendsmiths Cafe"),
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
 