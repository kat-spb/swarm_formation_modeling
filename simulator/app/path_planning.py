from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .geometry import (
    obstacle_active,
    obstacle_center,
    point_clearance_to_obstacle,
    polygon_vertices,
    rectangle_vertices,
    segment_collision_free,
)
from .models import ObstacleSpec
from .utils import Vec, add, clamp, dist, sub, unit


@dataclass
class PathResult:
    path: List[Vec]
    length_m: float
    status: str
    expanded: int = 0
    reason: str = ""


def path_length(path: Sequence[Vec]) -> float:
    return sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1)) if len(path) > 1 else 0.0


def resample_polyline(path: Sequence[Vec], max_points: int = 28) -> List[Vec]:
    """Return a compact but visually continuous representation of a path.

    This is used for the UI route preview; it does not change the physical
    collision-checked path used by the planner.
    """
    if not path:
        return []
    if len(path) <= max_points:
        return list(path)
    total = path_length(path)
    if total <= 1e-9:
        return [path[0]]
    out: List[Vec] = [path[0]]
    target_ds = [total * i / (max_points - 1) for i in range(1, max_points - 1)]
    seg_i = 0
    acc = 0.0
    for td in target_ds:
        while seg_i < len(path) - 2 and acc + dist(path[seg_i], path[seg_i + 1]) < td:
            acc += dist(path[seg_i], path[seg_i + 1])
            seg_i += 1
        a, b = path[seg_i], path[seg_i + 1]
        seg = max(1e-9, dist(a, b))
        t = max(0.0, min(1.0, (td - acc) / seg))
        out.append((a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t))
    out.append(path[-1])
    return out


def chaikin_smooth(path: Sequence[Vec], iterations: int = 2, max_points: int = 28) -> List[Vec]:
    """Chaikin corner-cutting preview path for rendering smooth bypasses.

    The planner still follows the collision-checked polyline; this curve is a
    visual and setpoint-smoothing aid, not an optimizer certificate.
    """
    pts = list(path)
    if len(pts) <= 2:
        return resample_polyline(pts, max_points)
    for _ in range(max(0, iterations)):
        new_pts: List[Vec] = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            q = (0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1])
            r = (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1])
            new_pts.extend([q, r])
        new_pts.append(pts[-1])
        pts = new_pts
    return resample_polyline(pts, max_points)


def simplify_path(path: List[Vec], obstacles: Sequence[ObstacleSpec], time_s: float, inflation: float, lane_half: float) -> List[Vec]:
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if segment_collision_free(path[i], path[j], obstacles, time_s, inflation, lane_half, samples_per_m=10.0):
                break
            j -= 1
        out.append(path[j])
        i = j
    return out


def astar_grid(
    start: Vec,
    goal: Vec,
    mission_length_m: float,
    corridor_width_m: float,
    obstacles: Sequence[ObstacleSpec],
    time_s: float,
    agent_radius_m: float,
    safety_margin_m: float,
    resolution_m: float = 0.12,
    x_pad_m: float = 0.75,
    simplify: bool = True,
) -> PathResult:
    lane_half = corridor_width_m / 2.0
    inflation = agent_radius_m + safety_margin_m
    xmin = min(-x_pad_m, start[0] - x_pad_m)
    xmax = max(mission_length_m + x_pad_m, goal[0] + x_pad_m)
    ymin = -lane_half + inflation
    ymax = lane_half - inflation
    if ymin >= ymax:
        return PathResult([], float("inf"), "no_path", reason="corridor narrower than inflated agent")

    nx = max(3, int(math.ceil((xmax - xmin) / resolution_m)) + 1)
    ny = max(3, int(math.ceil((ymax - ymin) / resolution_m)) + 1)

    active_obs = [o for o in obstacles if obstacle_active(o, time_s)]

    def to_idx(p: Vec) -> Tuple[int, int]:
        ix = int(round((p[0] - xmin) / resolution_m))
        iy = int(round((p[1] - ymin) / resolution_m))
        return (clamp(ix, 0, nx - 1), clamp(iy, 0, ny - 1))

    def to_xy(idx: Tuple[int, int]) -> Vec:
        ix, iy = idx
        return (xmin + ix * resolution_m, ymin + iy * resolution_m)

    blocked_cache: Dict[Tuple[int, int], bool] = {}

    def blocked(idx: Tuple[int, int]) -> bool:
        if idx in blocked_cache:
            return blocked_cache[idx]
        x, y = to_xy(idx)
        if y < ymin - 1e-9 or y > ymax + 1e-9:
            blocked_cache[idx] = True
            return True
        for obs in active_obs:
            if point_clearance_to_obstacle((x, y), obs, time_s) <= inflation:
                blocked_cache[idx] = True
                return True
        blocked_cache[idx] = False
        return False

    def nearest_free(idx: Tuple[int, int], max_ring: int = 12) -> Optional[Tuple[int, int]]:
        if not blocked(idx):
            return idx
        best = None
        best_d = float("inf")
        for r in range(1, max_ring + 1):
            for dx in range(-r, r + 1):
                for dy in (-r, r):
                    cand = (idx[0] + dx, idx[1] + dy)
                    if 0 <= cand[0] < nx and 0 <= cand[1] < ny and not blocked(cand):
                        d = dist(to_xy(idx), to_xy(cand))
                        if d < best_d:
                            best = cand
                            best_d = d
                for dy in range(-r + 1, r):
                    for dx in (-r, r):
                        cand = (idx[0] + dx, idx[1] + dy)
                        if 0 <= cand[0] < nx and 0 <= cand[1] < ny and not blocked(cand):
                            d = dist(to_xy(idx), to_xy(cand))
                            if d < best_d:
                                best = cand
                                best_d = d
            if best is not None:
                return best
        return None

    start_idx = nearest_free(to_idx(start), max_ring=18)
    goal_idx = nearest_free(to_idx(goal), max_ring=18)
    if start_idx is None or goal_idx is None:
        return PathResult([], float("inf"), "no_path", reason="start or goal is inside inflated obstacles")

    # Fast straight-line case.
    if segment_collision_free(start, goal, active_obs, time_s, inflation, lane_half, samples_per_m=10.0):
        return PathResult([start, goal], dist(start, goal), "ok", expanded=0)

    dirs = [
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (-1, -1, math.sqrt(2)),
    ]

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
    heapq.heappush(open_heap, (dist(to_xy(start_idx), to_xy(goal_idx)), 0.0, start_idx))
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    gscore: Dict[Tuple[int, int], float] = {start_idx: 0.0}
    closed = set()
    expanded = 0

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded += 1
        if cur == goal_idx:
            cells = [cur]
            while cells[-1] != start_idx:
                cells.append(came[cells[-1]])
            cells.reverse()
            pts = [start] + [to_xy(c) for c in cells[1:-1]] + [goal]
            if simplify:
                pts = simplify_path(pts, active_obs, time_s, inflation, lane_half)
            return PathResult(pts, path_length(pts), "ok", expanded=expanded)
        for dx, dy, step_cost in dirs:
            nb = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if nb in closed or blocked(nb):
                continue
            # Avoid diagonal squeezing through obstacle corners.
            if dx != 0 and dy != 0 and (blocked((cur[0] + dx, cur[1])) or blocked((cur[0], cur[1] + dy))):
                continue
            tentative = gscore[cur] + step_cost * resolution_m
            if tentative < gscore.get(nb, float("inf")):
                came[nb] = cur
                gscore[nb] = tentative
                h = dist(to_xy(nb), to_xy(goal_idx))
                # Small bias toward mission progress in x but never at the cost of blocking.
                heapq.heappush(open_heap, (tentative + h, tentative, nb))
    return PathResult([], float("inf"), "no_path", expanded=expanded, reason="A* open set exhausted")


def visibility_shortest_path(
    start: Vec,
    goal: Vec,
    mission_length_m: float,
    corridor_width_m: float,
    obstacles: Sequence[ObstacleSpec],
    time_s: float,
    agent_radius_m: float,
    safety_margin_m: float,
) -> PathResult:
    lane_half = corridor_width_m / 2.0
    inflation = agent_radius_m + safety_margin_m
    active_obs = [o for o in obstacles if obstacle_active(o, time_s)]
    nodes: List[Vec] = [start, goal]

    # Obstacle corners / circle samples in configuration space.
    for obs in active_obs:
        if obs.kind == "rectangle":
            candidates = rectangle_vertices(obs, inflation=inflation + 0.015, time_s=time_s)
        elif obs.kind == "circle":
            cx, cy = obstacle_center(obs, time_s)
            r = obs.radius_m + inflation + 0.015
            candidates = [(cx + r * math.cos(2 * math.pi * i / 24), cy + r * math.sin(2 * math.pi * i / 24)) for i in range(24)]
        else:
            candidates = polygon_vertices(obs, inflation=inflation + 0.015, time_s=time_s)
        for p in candidates:
            if p[0] < -0.5 or p[0] > mission_length_m + 0.5:
                continue
            if abs(p[1]) > lane_half - inflation:
                continue
            if all(point_clearance_to_obstacle(p, o, time_s) > inflation for o in active_obs):
                nodes.append(p)

    # Add corridor wall tangency candidates for wide obstacles.
    nodes.extend([(-0.0, -lane_half + inflation), (-0.0, lane_half - inflation), (mission_length_m, -lane_half + inflation), (mission_length_m, lane_half - inflation)])

    n = len(nodes)
    adj: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if segment_collision_free(nodes[i], nodes[j], active_obs, time_s, inflation, lane_half, samples_per_m=16.0):
                d = dist(nodes[i], nodes[j])
                adj[i].append((j, d))
                adj[j].append((i, d))

    pq: List[Tuple[float, int]] = [(0.0, 0)]
    best = {0: 0.0}
    prev: Dict[int, int] = {}
    seen = set()
    while pq:
        g, i = heapq.heappop(pq)
        if i in seen:
            continue
        seen.add(i)
        if i == 1:
            seq = [1]
            while seq[-1] != 0:
                seq.append(prev[seq[-1]])
            seq.reverse()
            path = [nodes[k] for k in seq]
            return PathResult(path, g, "ok", expanded=len(seen), reason="visibility_graph")
        for j, w in adj[i]:
            ng = g + w
            if ng < best.get(j, float("inf")):
                best[j] = ng
                prev[j] = i
                heapq.heappush(pq, (ng, j))
    # Fallback to grid for non-rectangular or difficult scenes.
    grid = astar_grid(start, goal, mission_length_m, corridor_width_m, obstacles, time_s, agent_radius_m, safety_margin_m, resolution_m=0.06)
    if grid.status == "ok":
        grid.reason = "grid_fallback"
    return grid


def pick_next_waypoint(path: Sequence[Vec], pos: Vec, lookahead_m: float = 0.35) -> Vec:
    if not path:
        return pos
    if len(path) == 1:
        return path[0]
    acc = 0.0
    last = path[0]
    for p in path[1:]:
        seg = dist(last, p)
        if acc + seg >= lookahead_m:
            # interpolate along this segment.
            remain = lookahead_m - acc
            t = remain / seg if seg > 1e-9 else 0.0
            return (last[0] * (1 - t) + p[0] * t, last[1] * (1 - t) + p[1] * t)
        acc += seg
        last = p
    return path[-1]


def route_to_commands(path: Sequence[Vec], speed_mps: float = 0.30, dt_s: float = 0.20, max_power: int = 1000) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if len(path) < 2:
        return rows
    t = 0.0
    ts = 1784462000
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        length = dist(a, b)
        if length < 1e-9:
            continue
        heading = math.atan2(b[1] - a[1], b[0] - a[0])
        steps = max(1, int(math.ceil(length / max(1e-6, speed_mps * dt_s))))
        for k in range(steps):
            frac = min(1.0, (k + 1) / steps)
            x = a[0] * (1 - frac) + b[0] * frac
            y = a[1] * (1 - frac) + b[1] * frac
            pwr = int(clamp(round(speed_mps / 0.38 * max_power), -max_power, max_power))
            rows.append({
                "ts": ts + int(round(t * 1000)),
                "t_s": round(t, 3),
                "cmd": "w",
                "pwr_left": pwr,
                "pwr_right": pwr,
                "x_m": round(x, 4),
                "y_m": round(y, 4),
                "theta_rad": round(heading, 4),
                "segment": i,
            })
            t += dt_s
    rows.append({"ts": ts + int(round(t * 1000)), "t_s": round(t, 3), "cmd": "e", "pwr_left": 0, "pwr_right": 0, "x_m": round(path[-1][0], 4), "y_m": round(path[-1][1], 4), "theta_rad": 0.0, "segment": len(path) - 1})
    return rows
