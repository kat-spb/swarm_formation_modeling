from __future__ import annotations

import itertools
import math
from typing import Dict, List, Sequence, Tuple

from .models import FormationNode, FormationSpec
from .utils import Vec, centroid, dist, pairwise_matrix, rms


def make_offsets(shape: str, n: int, spacing: float) -> Dict[str, Vec]:
    n = max(1, int(n))
    if shape == "line":
        pts = [((i - (n - 1) / 2) * spacing, 0.0) for i in range(n)]
    elif shape == "circle":
        if n == 1:
            pts = [(0.0, 0.0)]
        else:
            radius = spacing * max(1.0, n / (2.0 * math.pi))
            pts = [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n)]
    elif shape == "square":
        side = math.ceil(math.sqrt(n))
        pts = []
        for row in range(side):
            for col in range(side):
                if len(pts) >= n:
                    break
                pts.append(((col - (side - 1) / 2) * spacing, (row - (side - 1) / 2) * spacing))
    elif shape == "hex":
        pts = [(0.0, 0.0)]
        rings = 1
        while len(pts) < n:
            # axial hex grid ring.
            q, r = rings, 0
            dirs = [(0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0)]
            for dq, dr in dirs:
                for _ in range(rings):
                    if len(pts) >= n:
                        break
                    x = spacing * (q + r / 2.0)
                    y = spacing * (math.sqrt(3) / 2.0 * r)
                    pts.append((x, y))
                    q += dq
                    r += dr
                if len(pts) >= n:
                    break
            rings += 1
    else:  # triangle
        pts = []
        rows = 1
        while len(pts) < n:
            for col in range(rows):
                if len(pts) >= n:
                    break
                x = (col - (rows - 1) / 2) * spacing
                y = (rows - 1) * spacing * math.sqrt(3) / 2
                pts.append((x, y))
            rows += 1
    c = centroid(pts)
    pts = [(x - c[0], y - c[1]) for x, y in pts]
    return {f"r{i + 1:02d}": p for i, p in enumerate(pts)}


def make_formation_spec(shape: str, n: int, spacing: float) -> FormationSpec:
    offsets = make_offsets(shape, n, spacing)
    nodes = [FormationNode(id=rid, name=rid, x=p[0], y=p[1], powered=True) for rid, p in offsets.items()]
    matrix = pairwise_matrix([(node.x, node.y) for node in nodes])
    return FormationSpec(name=f"{shape}_{n}", nodes=nodes, distance_matrix=matrix, tolerance_m=max(0.06, spacing * 0.25), edge_tolerance_m=max(0.05, spacing * 0.22))


def formation_points(spec: FormationSpec) -> Dict[str, Vec]:
    pts = {n.id: (n.x, n.y) for n in spec.nodes}
    c = centroid(list(pts.values()))
    return {rid: (p[0] - c[0], p[1] - c[1]) for rid, p in pts.items()}


def desired_distance_matrix(spec: FormationSpec) -> List[List[float]]:
    pts = [(n.x, n.y) for n in spec.nodes]
    n = len(pts)
    if spec.distance_matrix and len(spec.distance_matrix) == n and all(len(row) == n for row in spec.distance_matrix):
        return [[float(v) for v in row] for row in spec.distance_matrix]
    return pairwise_matrix(pts)


def validate_formation(spec: FormationSpec, corridor_width: float, robot_radius: float, safety_margin: float) -> Dict[str, object]:
    errors: List[str] = []
    warnings: List[str] = []
    nodes = spec.nodes
    ids = [n.id for n in nodes]
    if len(set(ids)) != len(ids):
        errors.append("agent ids in formation must be unique")
    if not nodes:
        errors.append("formation must contain at least one node")
        return {"ok": False, "errors": errors, "warnings": warnings, "ids": [], "matrix": []}
    pts = [(n.x, n.y) for n in nodes]
    c = centroid(pts)
    span_y = (max(p[1] for p in pts) - min(p[1] for p in pts)) if len(pts) > 1 else 0.0
    if span_y + 2 * (robot_radius + safety_margin) > corridor_width:
        warnings.append("formation is wider than the corridor with the current robot radius and margin")
    geom_matrix = pairwise_matrix(pts)
    matrix = desired_distance_matrix(spec)
    n = len(nodes)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        errors.append("distance_matrix must be NxN and match node count")
    else:
        for i in range(n):
            if abs(matrix[i][i]) > 1e-8:
                errors.append("distance_matrix diagonal must be zero")
                break
        for i in range(n):
            for j in range(i + 1, n):
                if matrix[i][j] < 0:
                    errors.append("distance_matrix must be non-negative")
                    break
                if abs(matrix[i][j] - matrix[j][i]) > 1e-6:
                    errors.append("distance_matrix must be symmetric")
                    break
        # Triangle inequalities: O(n^3), fine for editor sizes.
        tri_violations = 0
        for i, j, k in itertools.combinations(range(n), 3):
            a, b, cc = matrix[i][j], matrix[j][k], matrix[i][k]
            if a + b + 1e-6 < cc or a + cc + 1e-6 < b or b + cc + 1e-6 < a:
                tri_violations += 1
        if tri_violations:
            warnings.append(f"distance matrix violates triangle inequality in {tri_violations} triplets")
        diffs = [geom_matrix[i][j] - matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        embedding_rms = rms(diffs)
        if embedding_rms > spec.tolerance_m:
            warnings.append(f"coordinate embedding differs from distance_matrix: RMS={embedding_rms:.3f} m")
        min_d = min([matrix[i][j] for i in range(n) for j in range(i + 1, n)] or [float("inf")])
        if min_d < 2 * robot_radius + safety_margin:
            warnings.append("some desired distances are smaller than robot diameter plus safety margin")
    edges = formation_edges_from_matrix(spec, max_edges=200)
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "ids": ids,
        "centroid": {"x": c[0], "y": c[1]},
        "span_y_m": span_y,
        "matrix": matrix,
        "geom_matrix": geom_matrix,
        "edges": edges,
    }


def formation_edges_from_matrix(spec: FormationSpec, max_edges: int = 120) -> List[Dict[str, object]]:
    ids = [n.id for n in spec.nodes]
    if len(ids) <= 1:
        return []
    locked = []
    for edge in spec.locked_edges:
        if len(edge) >= 2 and edge[0] in ids and edge[1] in ids:
            locked.append({"source": edge[0], "target": edge[1], "kind": "locked"})
    if locked:
        return locked[:max_edges]
    matrix = desired_distance_matrix(spec)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((matrix[i][j], ids[i], ids[j]))
    # Keep the local graph readable: nearest neighbors plus enough edges for diagnostics.
    pairs.sort(key=lambda x: x[0])
    degree = {rid: 0 for rid in ids}
    edges = []
    for d, a, b in pairs:
        if len(edges) >= max_edges:
            break
        if degree[a] < 4 or degree[b] < 4 or len(ids) <= 8:
            edges.append({"source": a, "target": b, "desired_m": d, "kind": "metric"})
            degree[a] += 1
            degree[b] += 1
    return edges


def formation_matrices(ids: Sequence[str], positions: Dict[str, Vec], desired_matrix: List[List[float]], tolerance: float) -> Dict[str, object]:
    pts = [positions[rid] for rid in ids]
    current = pairwise_matrix(pts)
    n = len(ids)
    delta = [[current[i][j] - desired_matrix[i][j] for j in range(n)] for i in range(n)]
    bad_pairs = []
    vals = []
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(delta[i][j])
            if abs(delta[i][j]) > tolerance:
                bad_pairs.append({
                    "a": ids[i],
                    "b": ids[j],
                    "desired_m": desired_matrix[i][j],
                    "current_m": current[i][j],
                    "delta_m": delta[i][j],
                    "severity": "red" if abs(delta[i][j]) > 1.7 * tolerance else "yellow",
                })
    return {
        "ids": list(ids),
        "initial": desired_matrix,
        "current": current,
        "delta": delta,
        "rms_m": rms(vals),
        "max_abs_delta_m": max([abs(v) for v in vals] or [0.0]),
        "bad_pairs": bad_pairs,
    }


def align_formation_to_preset(spec: FormationSpec, shape: str, spacing: float, recompute_matrix: bool = True, preserve_ids: bool = True) -> FormationSpec:
    """Move existing formation nodes onto a selected preset grid.

    This is used by the editor: a user can start from triangle/square/hex/circle/line,
    add/remove agents, then snap the embedding back to that grid while optionally
    keeping the formal matrix or recomputing it from the snapped embedding.
    """
    n = len(spec.nodes)
    preset = make_formation_spec(shape, n, spacing)
    out_nodes: List[FormationNode] = []
    for i, pnode in enumerate(preset.nodes):
        old = spec.nodes[i]
        if preserve_ids:
            out_nodes.append(FormationNode(id=old.id, name=old.name or old.id, x=pnode.x, y=pnode.y, powered=old.powered))
        else:
            out_nodes.append(pnode)
    matrix = pairwise_matrix([(node.x, node.y) for node in out_nodes]) if recompute_matrix else spec.distance_matrix
    return FormationSpec(
        name=f"custom_from_{shape}_{n}",
        nodes=out_nodes,
        distance_matrix=matrix,
        tolerance_m=spec.tolerance_m,
        edge_tolerance_m=spec.edge_tolerance_m,
        locked_edges=spec.locked_edges,
        assignment_mode=spec.assignment_mode,
    )
