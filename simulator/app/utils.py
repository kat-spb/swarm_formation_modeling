from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

Vec = Tuple[float, float]


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k)


def dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def norm(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit(a: Vec, fallback: Vec = (1.0, 0.0)) -> Vec:
    n = norm(a)
    if n < 1e-12:
        return fallback
    return (a[0] / n, a[1] / n)


def clamp_vec(v: Vec, max_len: float) -> Vec:
    n = norm(v)
    if n <= max_len or n < 1e-12:
        return v
    return (v[0] * max_len / n, v[1] * max_len / n)


def angle_wrap(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def centroid(points: Sequence[Vec]) -> Vec:
    if not points:
        return (0.0, 0.0)
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


def rms(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def pairwise_matrix(points: Sequence[Vec]) -> List[List[float]]:
    n = len(points)
    return [[0.0 if i == j else dist(points[i], points[j]) for j in range(n)] for i in range(n)]


def pairwise_distances(points: Sequence[Vec]) -> List[float]:
    vals: List[float] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            vals.append(dist(points[i], points[j]))
    vals.sort()
    return vals


def model_dump_compat(model) -> Dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def fmt_float(x: float, ndigits: int = 4) -> float:
    if math.isinf(x) or math.isnan(x):
        return x
    return round(float(x), ndigits)


def rotate(v: Vec, angle_rad: float) -> Vec:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def polygon_area(poly: Sequence[Vec]) -> float:
    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        s += p[0] * q[1] - q[0] * p[1]
    return 0.5 * s


def ensure_ccw(poly: List[Vec]) -> List[Vec]:
    return poly if polygon_area(poly) >= 0 else list(reversed(poly))
