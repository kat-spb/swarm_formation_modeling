from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .models import ObstacleSpec
from .utils import Vec, add, clamp, dist, dot, ensure_ccw, norm, rotate, sub, unit


def obstacle_id(obs: ObstacleSpec, idx: int = 0) -> str:
    return obs.id or f"o{idx + 1:02d}"


def obstacle_active(obs: ObstacleSpec, time_s: float) -> bool:
    if not obs.active_by_default:
        return False
    if time_s < obs.appear_time_s:
        return False
    if obs.disappear_time_s is not None and time_s > obs.disappear_time_s:
        return False
    return True


def obstacle_center(obs: ObstacleSpec, time_s: float = 0.0) -> Vec:
    return (obs.x + obs.vx_mps * time_s, obs.y + obs.vy_mps * time_s)


def rectangle_vertices(obs: ObstacleSpec, inflation: float = 0.0, time_s: float = 0.0) -> List[Vec]:
    cx, cy = obstacle_center(obs, time_s)
    w = obs.width_m + 2.0 * inflation
    h = obs.height_m + 2.0 * inflation
    pts = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    a = math.radians(obs.rotation_deg)
    return [(cx + rotate(p, a)[0], cy + rotate(p, a)[1]) for p in pts]


def polygon_vertices(obs: ObstacleSpec, inflation: float = 0.0, time_s: float = 0.0) -> List[Vec]:
    if obs.kind == "rectangle":
        return rectangle_vertices(obs, inflation=inflation, time_s=time_s)
    if obs.kind == "circle":
        n = 24
        r = obs.radius_m + inflation
        cx, cy = obstacle_center(obs, time_s)
        return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
    if obs.vertices:
        cx, cy = obstacle_center(obs, time_s)
        verts = [(cx + v.x, cy + v.y) for v in obs.vertices]
        if inflation <= 1e-12:
            return ensure_ccw(verts)
        return inflate_polygon_approx(ensure_ccw(verts), inflation)
    if obs.side_lengths_m and obs.interior_angles_deg:
        verts = build_polygon_from_sides_angles(obs.side_lengths_m, obs.interior_angles_deg)
        cx, cy = obstacle_center(obs, time_s)
        verts = [(x + cx, y + cy) for x, y in verts]
        if inflation > 1e-12:
            verts = inflate_polygon_approx(ensure_ccw(verts), inflation)
        return verts
    return rectangle_vertices(obs, inflation=inflation, time_s=time_s)


def obstacle_approx_radius(obs: ObstacleSpec) -> float:
    if obs.kind == "circle":
        return obs.radius_m
    if obs.kind == "rectangle":
        return 0.5 * math.sqrt(obs.width_m * obs.width_m + obs.height_m * obs.height_m)
    verts = polygon_vertices(obs, 0.0, 0.0)
    if not verts:
        return obs.radius_m
    c = (sum(p[0] for p in verts) / len(verts), sum(p[1] for p in verts) / len(verts))
    return max(dist(c, p) for p in verts)


def build_polygon_from_sides_angles(sides: Sequence[float], interior_angles_deg: Sequence[float]) -> List[Vec]:
    # Turtle reconstruction. It is used for validation/editing, not for exact CAD.
    n = min(len(sides), len(interior_angles_deg))
    if n < 3:
        return []
    pts: List[Vec] = [(0.0, 0.0)]
    heading = 0.0
    x, y = 0.0, 0.0
    for i in range(n - 1):
        x += sides[i] * math.cos(heading)
        y += sides[i] * math.sin(heading)
        pts.append((x, y))
        turn = math.radians(180.0 - interior_angles_deg[(i + 1) % n])
        heading += turn
    # Center the polygon and preserve approximate shape.
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return ensure_ccw([(p[0] - cx, p[1] - cy) for p in pts])


def inflate_polygon_approx(poly: Sequence[Vec], margin: float) -> List[Vec]:
    if len(poly) < 3 or margin <= 0:
        return list(poly)
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    out = []
    for p in poly:
        v = sub(p, (cx, cy))
        out.append(add(p, tuple(t * margin for t in unit(v, fallback=(1.0, 0.0)))))
    return out


def point_segment_distance(p: Vec, a: Vec, b: Vec) -> float:
    ab = sub(b, a)
    l2 = dot(ab, ab)
    if l2 < 1e-12:
        return dist(p, a)
    t = clamp(dot(sub(p, a), ab) / l2, 0.0, 1.0)
    q = (a[0] + ab[0] * t, a[1] + ab[1] * t)
    return dist(p, q)


def point_in_convex_polygon(p: Vec, poly: Sequence[Vec]) -> bool:
    if len(poly) < 3:
        return False
    sign = None
    eps = 1e-10
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < eps:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif sign != s:
            return False
    return True


def point_clearance_to_obstacle(p: Vec, obs: ObstacleSpec, time_s: float = 0.0) -> float:
    if obs.kind == "circle":
        return dist(p, obstacle_center(obs, time_s)) - obs.radius_m
    poly = polygon_vertices(obs, inflation=0.0, time_s=time_s)
    if not poly:
        return float("inf")
    d = min(point_segment_distance(p, poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly)))
    return -d if point_in_convex_polygon(p, poly) else d


def segment_clearance_to_obstacle(a: Vec, b: Vec, obs: ObstacleSpec, time_s: float = 0.0, samples_per_m: float = 12.0) -> float:
    length = dist(a, b)
    samples = max(2, int(math.ceil(length * samples_per_m)))
    best = float("inf")
    for i in range(samples + 1):
        t = i / samples
        p = (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)
        best = min(best, point_clearance_to_obstacle(p, obs, time_s))
    return best


def point_clearance_to_any_obstacle(p: Vec, obstacles: Iterable[ObstacleSpec], time_s: float, active_only: bool = True) -> float:
    vals = []
    for obs in obstacles:
        if active_only and not obstacle_active(obs, time_s):
            continue
        vals.append(point_clearance_to_obstacle(p, obs, time_s))
    return min(vals) if vals else float("inf")


def segment_collision_free(a: Vec, b: Vec, obstacles: Iterable[ObstacleSpec], time_s: float, inflation: float, lane_half: float, samples_per_m: float = 12.0) -> bool:
    length = dist(a, b)
    samples = max(2, int(math.ceil(length * samples_per_m)))
    for i in range(samples + 1):
        t = i / samples
        p = (a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t)
        if abs(p[1]) > lane_half - inflation:
            return False
        for obs in obstacles:
            if not obstacle_active(obs, time_s):
                continue
            if point_clearance_to_obstacle(p, obs, time_s) <= inflation:
                return False
    return True


def validate_obstacle_geometry(obs: ObstacleSpec) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    if obs.kind == "rectangle":
        if obs.width_m <= 0 or obs.height_m <= 0:
            errors.append("rectangle dimensions must be positive")
    elif obs.kind == "circle":
        if obs.radius_m <= 0:
            errors.append("circle radius must be positive")
    elif obs.kind == "polygon":
        if obs.vertices and len(obs.vertices) > 6:
            warnings.append("polygon has more than 6 vertices; consider using kind='circle' for a high-k obstacle")
        verts = polygon_vertices(obs, 0.0, 0.0)
        if len(verts) < 3:
            errors.append("polygon must have at least 3 vertices")
        if len(verts) >= 3:
            ccw = ensure_ccw(verts)
            if abs(sum(ccw[i][0] * ccw[(i + 1) % len(ccw)][1] - ccw[(i + 1) % len(ccw)][0] * ccw[i][1] for i in range(len(ccw)))) < 1e-8:
                errors.append("polygon area is close to zero")
            # Simple convexity check.
            signs = []
            for i in range(len(ccw)):
                a, b, c = ccw[i], ccw[(i + 1) % len(ccw)], ccw[(i + 2) % len(ccw)]
                cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
                if abs(cross) > 1e-9:
                    signs.append(cross > 0)
            if signs and not all(s == signs[0] for s in signs):
                errors.append("polygon must be convex")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def obstacle_to_public(obs: ObstacleSpec, idx: int, time_s: float) -> Dict[str, object]:
    oid = obstacle_id(obs, idx)
    c = obstacle_center(obs, time_s)
    active = obstacle_active(obs, time_s)
    verts = polygon_vertices(obs, 0.0, time_s) if obs.kind in ("rectangle", "polygon") else []
    return {
        "id": oid,
        "label": obs.label or oid,
        "kind": obs.kind,
        "x": c[0],
        "y": c[1],
        "width_m": obs.width_m,
        "height_m": obs.height_m,
        "radius_m": obs.radius_m,
        "rotation_deg": obs.rotation_deg,
        "vertices": [{"x": x, "y": y} for x, y in verts],
        "appear_time_s": obs.appear_time_s,
        "disappear_time_s": obs.disappear_time_s,
        "active": active,
        "opacity": obs.opacity,
        "approx_radius_m": obstacle_approx_radius(obs),
    }
