from __future__ import annotations

import random
from typing import Dict, List

from .models import ObstacleRandomRequest, ObstacleSpec
from .utils import clamp


def _rect(oid: str, x: float, y: float, w: float, h: float, label: str | None = None, appear: float = 0.0, disappear: float | None = None) -> ObstacleSpec:
    return ObstacleSpec(
        id=oid,
        label=label or oid,
        kind="rectangle",
        x=x,
        y=y,
        width_m=w,
        height_m=h,
        radius_m=max(w, h) / 2,
        rotation_deg=0.0,
        appear_time_s=appear,
        disappear_time_s=disappear,
        active_by_default=True,
        opacity=0.36,
    )


def _circle(oid: str, x: float, y: float, r: float, label: str | None = None, appear: float = 0.0, disappear: float | None = None) -> ObstacleSpec:
    return ObstacleSpec(
        id=oid,
        label=label or oid,
        kind="circle",
        x=x,
        y=y,
        radius_m=r,
        width_m=2 * r,
        height_m=2 * r,
        appear_time_s=appear,
        disappear_time_s=disappear,
        active_by_default=True,
        opacity=0.36,
    )


def list_obstacle_maps() -> List[Dict[str, object]]:
    return [
        {"key": "clear_lane", "title": "Пустая полоса / эталон", "description": "Без препятствий: контроль длины 12 м и времени завершения.", "bench_focus": "reference_completion"},
        {"key": "single_gate", "title": "Один прямоугольник и боковой проход", "description": "Проверка одиночного обтекания и выбора верхнего/нижнего кластера.", "bench_focus": "simple_bypass"},
        {"key": "two_gates", "title": "Две шахматные заслонки", "description": "Проверка плавного split/rejoin и личных маршрутов агентов.", "bench_focus": "cluster_split_rejoin"},
        {"key": "rect_gate", "title": "Ворота из прямоугольников", "description": "Два стационарных прямоугольника образуют проверяемый проход.", "bench_focus": "physical_rectangles"},
        {"key": "central_block", "title": "Центральный блок", "description": "Один прямоугольник на оси полосы: тест выбора верх/низ.", "bench_focus": "side_choice"},
        {"key": "zigzag_rects", "title": "Зигзаг прямоугольников", "description": "Несколько прямоугольников заставляют рой менять сторону обхода.", "bench_focus": "multi_bypass"},
        {"key": "split_rejoin", "title": "Split/rejoin", "description": "Карта для разделения на верхний и нижний кластеры с последующим сбором.", "bench_focus": "adaptive_clusters"},
        {"key": "narrow_slalom", "title": "Узкий слалом", "description": "Серия прямоугольников у краев полосы, провоцирует перестроение.", "bench_focus": "corridor_constraints"},
        {"key": "dynamic_popup", "title": "Динамические pop-up препятствия", "description": "Часть препятствий появляется и исчезает после старта миссии.", "bench_focus": "local_perception_replanning"},
        {"key": "mixed_circles_rects", "title": "Смешанная карта: круги + прямоугольники", "description": "Проверка разных геометрий препятствий в одном бенче.", "bench_focus": "geometry_coverage"},
    ]


def obstacle_maps_index() -> Dict[str, object]:
    return {"maps": list_obstacle_maps()}


def obstacle_map(key: str, mission_length_m: float = 12.0, corridor_width_m: float = 2.4) -> List[ObstacleSpec]:
    L = mission_length_m
    lane = corridor_width_m / 2.0
    x = lambda frac: L * frac
    ycap = lambda y, h=0.0: clamp(y, -lane + h / 2 + 0.05, lane - h / 2 - 0.05)
    key = key or "clear_lane"
    if key in ("clear_lane", "empty"):
        return []
    if key == "single_gate":
        return [_rect("gate-left", x(0.44), ycap(-0.32, 0.78), 0.62, 0.78, "gate-left")]
    if key == "two_gates":
        return [
            _rect("gate-a", x(0.33), ycap(-0.48, 0.70), 0.58, 0.70, "gate-a"),
            _rect("gate-b", x(0.57), ycap(0.42, 0.76), 0.64, 0.76, "gate-b"),
        ]
    if key == "rect_gate":
        return [
            _rect("gate-upper", x(0.48), ycap(lane * 0.48, 0.56), 0.80, 0.56, "gate-upper"),
            _rect("gate-lower", x(0.48), ycap(-lane * 0.48, 0.56), 0.80, 0.56, "gate-lower"),
        ]
    if key == "central_block":
        return [_rect("central-block", x(0.48), 0.0, 0.70, 0.62, "central-block")]
    if key == "zigzag_rects":
        return [
            _rect("zig-1", x(0.24), ycap(-lane * 0.36, 0.54), 0.62, 0.54, "zig-1"),
            _rect("zig-2", x(0.40), ycap(lane * 0.34, 0.58), 0.58, 0.58, "zig-2"),
            _rect("zig-3", x(0.58), ycap(-lane * 0.30, 0.54), 0.64, 0.54, "zig-3"),
            _rect("zig-4", x(0.74), ycap(lane * 0.28, 0.50), 0.56, 0.50, "zig-4"),
        ]
    if key == "split_rejoin":
        return [
            _rect("split-center-1", x(0.38), 0.0, 0.82, 0.45, "split-center-1"),
            _rect("split-upper", x(0.52), ycap(lane * 0.46, 0.52), 0.62, 0.52, "split-upper"),
            _rect("split-lower", x(0.52), ycap(-lane * 0.46, 0.52), 0.62, 0.52, "split-lower"),
            _rect("rejoin-center", x(0.66), 0.0, 0.70, 0.40, "rejoin-center"),
        ]
    if key == "narrow_slalom":
        return [
            _rect("slalom-1", x(0.26), ycap(-0.64, 0.75), 0.56, 0.75, "slalom-1"),
            _rect("slalom-2", x(0.42), ycap(0.62, 0.72), 0.52, 0.72, "slalom-2"),
            _rect("slalom-3", x(0.58), ycap(-0.58, 0.68), 0.54, 0.68, "slalom-3"),
            _rect("slalom-4", x(0.74), ycap(0.56, 0.62), 0.50, 0.62, "slalom-4"),
        ]
    if key == "dynamic_popup":
        return [
            _rect("static-1", x(0.36), ycap(-0.42, 0.68), 0.54, 0.68, "static-1", appear=0.0),
            _rect("pop-up-1", x(0.52), ycap(0.28, 0.50), 0.52, 0.50, "pop-up-1", appear=5.0, disappear=14.0),
            _rect("pop-up-2", x(0.66), ycap(-0.26, 0.48), 0.50, 0.48, "pop-up-2", appear=9.0, disappear=18.0),
            _rect("static-2", x(0.80), ycap(0.44, 0.58), 0.44, 0.58, "static-2", appear=0.0),
        ]
    if key == "mixed_circles_rects":
        return [
            _rect("rect-1", x(0.30), ycap(-0.34, 0.62), 0.58, 0.62, "rect-1"),
            _circle("circle-1", x(0.47), ycap(0.42, 0.58), 0.29, "circle-1"),
            _rect("rect-2", x(0.64), ycap(-0.46, 0.58), 0.50, 0.58, "rect-2"),
            _circle("circle-2", x(0.76), ycap(0.12, 0.46), 0.23, "circle-2"),
        ]
    raise KeyError(key)


def generate_random_obstacles(req: ObstacleRandomRequest) -> List[ObstacleSpec]:
    rng = random.Random(req.seed)
    max_x = req.max_x_m if req.max_x_m is not None else max(req.min_x_m + 0.1, req.mission_length_m - 1.0)
    lane = req.corridor_width_m / 2.0
    obstacles: List[ObstacleSpec] = []
    attempts = 0
    while len(obstacles) < req.count and attempts < max(1000, req.count * 80):
        attempts += 1
        w = rng.uniform(min(req.width_min_m, req.width_max_m), max(req.width_min_m, req.width_max_m))
        h = rng.uniform(min(req.height_min_m, req.height_max_m), max(req.height_min_m, req.height_max_m))
        h = min(h, max(0.05, req.corridor_width_m - 0.18))
        x = rng.uniform(req.min_x_m, max_x)
        y_low = -lane + h / 2 + 0.08
        y_high = lane - h / 2 - 0.08
        if y_low > y_high:
            continue
        y = rng.uniform(y_low, y_high)
        ok = True
        for o in obstacles:
            ow = o.width_m if o.kind == "rectangle" else o.radius_m * 2
            oh = o.height_m if o.kind == "rectangle" else o.radius_m * 2
            if abs(x - o.x) < (w + ow) / 2 + req.min_gap_m and abs(y - o.y) < (h + oh) / 2 + req.min_gap_m:
                ok = False
                break
        if not ok:
            continue
        i = len(obstacles) + 1
        dynamic = rng.random() < req.dynamic_fraction
        appear = req.appear_default_s if not dynamic else round(rng.uniform(2.0, 12.0), 2)
        disappear = None if not dynamic else round(appear + rng.uniform(5.0, 14.0), 2)
        kind = req.kind
        if kind == "mixed":
            kind = "circle" if rng.random() < 0.35 else "rectangle"
        if kind == "circle":
            r = max(0.08, min(w, h) / 2.0)
            obs = _circle(f"rand-{i:02d}", x, y, r, f"rand-{i:02d}", appear=appear, disappear=disappear)
        else:
            obs = _rect(f"rand-{i:02d}", x, y, w, h, f"rand-{i:02d}", appear=appear, disappear=disappear)
        obstacles.append(obs)
    return obstacles




def random_obstacles(req: ObstacleRandomRequest) -> List[ObstacleSpec]:
    return generate_random_obstacles(req)


def available_obstacle_maps() -> List[Dict[str, object]]:
    return list_obstacle_maps()


def map_obstacles(req) -> List[ObstacleSpec]:
    return obstacle_map(req.key, req.mission_length_m, req.corridor_width_m)
