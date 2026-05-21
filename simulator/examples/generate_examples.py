from __future__ import annotations

import csv
import json
from pathlib import Path

from app.formation import make_formation_spec
from app.models import ObstacleSpec, SimConfig, SingleOptimalRequest, Vec2Model
from app.sim import SwarmSim, compute_single_optimal

BASE = Path(__file__).resolve().parent


def dump_json(name: str, data):
    (BASE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_and_log(config_data: dict, stem: str):
    sim = SwarmSim(SimConfig(**config_data))
    glog_lines = []
    glog_lines.extend(sim.status_lines_g())
    while not sim.done and sim.step_n < sim.config.max_steps:
        sim.step()
        if sim.step_n % 1 == 0:
            glog_lines.extend(sim.status_lines_g())
    (BASE / f"{stem}.csv").write_text(sim.csv_text(), encoding="utf-8")
    (BASE / f"{stem}.g.log").write_text("\n".join(glog_lines) + "\n", encoding="utf-8")
    return sim


def base_config(name: str, n: int, shape: str, spacing: float, width: float = 2.4, alg: str = "smooth_adaptive_clusters"):
    formation = make_formation_spec(shape, n, spacing)
    return {
        "name": name,
        "mission_length_m": 12.0,
        "corridor_width_m": width,
        "dt_s": 0.2,
        "max_steps": 950,
        "algorithm": alg,
        "grid_resolution_m": 0.12,
        "seed": 42,
        "formation": formation.model_dump(),
        "n_agents": n,
        "formation_shape": shape,
        "formation_spacing_m": spacing,
        "random_obstacles": False,
        "random_obstacle_count": 0,
        "perception": {"mode": "local", "shared_memory": True, "obstacle_memory_s": 18.0, "comms_range_m": 4.0, "broadcast_obstacles": True, "broadcast_positions": True},
        "objective": {"completion_policy": "finish_first", "collective_goal_weight": 1.0, "agent_survival_weight": 1.0, "final_place_weight": 1.0, "energy_weight": 0.2, "time_weight": 0.25, "centroid_length_weight": 0.25, "allow_reassignment": False, "require_all_powered_agents": True, "success_requires_formation": True, "success_requires_no_collision": False},
        "limits": {"safety_margin_m": 0.055, "robot_robot_margin_m": 0.04, "obstacle_margin_m": 0.05, "replan_period_steps": 3, "max_reconfig_speed_mps": 0.30, "smooth_reconfiguration": True, "setpoint_smoothing_alpha": 0.70, "setpoint_max_rate_mps": 0.55, "min_route_lookahead_m": 0.30, "route_preview_points": 28, "agent_trail_points": 140, "cluster_hysteresis_steps": 2, "avoidance_horizon_s": 1.2, "enable_safety_filter": True, "allow_reverse": True, "stop_if_no_path": False},
        "obstacles": [],
    }


def rect(id, x, y, w, h, appear=0.0, disappear=None, label=None):
    return {"id": id, "label": label or id, "kind": "rectangle", "x": x, "y": y, "width_m": w, "height_m": h, "radius_m": 0.25, "rotation_deg": 0.0, "appear_time_s": appear, "disappear_time_s": disappear, "vx_mps": 0.0, "vy_mps": 0.0, "active_by_default": True, "opacity": 0.35}


def circle(id, x, y, r, appear=0.0, disappear=None):
    return {"id": id, "label": id, "kind": "circle", "x": x, "y": y, "width_m": 0.5, "height_m": 0.5, "radius_m": r, "rotation_deg": 0.0, "appear_time_s": appear, "disappear_time_s": disappear, "vx_mps": 0.0, "vy_mps": 0.0, "active_by_default": True, "opacity": 0.35}


def main():
    examples = []

    c0 = base_config("formation_7_smooth_bypass", 7, "triangle", 0.38, width=2.55, alg="smooth_adaptive_clusters")
    c0["obstacles"] = [rect("smooth-block-center", 4.2, 0.0, 0.68, 0.70), rect("smooth-upper", 6.15, 0.43, 0.72, 0.52), rect("smooth-lower", 7.75, -0.42, 0.58, 0.56)]
    dump_json("formation_7_smooth_bypass.json", c0)
    sim0 = run_and_log(c0, "control_formation_7_smooth_bypass")
    examples.append({"title": "7 агентов, плавное обтекание split/rejoin v4", "config": "formation_7_smooth_bypass.json", "csv": "control_formation_7_smooth_bypass.csv", "glog": "control_formation_7_smooth_bypass.g.log", "expected": {"finish_reason": sim0.finish_reason, "steps": sim0.step_n}})

    c1 = base_config("single_clear_12m", 1, "line", 0.38, width=2.0)
    c1["perception"]["mode"] = "global_debug"
    dump_json("single_clear_12m.json", c1)
    sim1 = run_and_log(c1, "control_single_clear_12m")
    examples.append({"title": "1 машинка, чистая полоса 12 м", "config": "single_clear_12m.json", "csv": "control_single_clear_12m.csv", "glog": "control_single_clear_12m.g.log", "expected": {"finish_reason": sim1.finish_reason, "steps": sim1.step_n}})

    c2 = base_config("single_rectangles_physical", 1, "line", 0.38, width=2.2)
    c2["perception"]["mode"] = "global_debug"
    c2["obstacles"] = [rect("rA", 3.2, 0.0, 0.55, 0.55, label="center-rect"), rect("rB", 5.7, 0.45, 0.70, 0.45, label="upper-rect"), rect("rC", 8.1, -0.42, 0.65, 0.45, label="lower-rect")]
    dump_json("single_rectangles_physical.json", c2)
    sim2 = run_and_log(c2, "control_single_rectangles_physical")
    req = SingleOptimalRequest(mission_length_m=12, corridor_width_m=2.2, start=Vec2Model(x=0, y=0), goal=Vec2Model(x=12, y=0), obstacles=[ObstacleSpec(**o) for o in c2["obstacles"]])
    opt2 = compute_single_optimal(req)
    with (BASE / "optimal_single_rectangles_control.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(opt2["control_rows"][0].keys()) if opt2["control_rows"] else ["empty"])
        writer.writeheader(); writer.writerows(opt2["control_rows"])
    dump_json("optimal_single_rectangles_path.json", opt2)
    examples.append({"title": "1 машинка, прямоугольники, оптимальный маршрут", "config": "single_rectangles_physical.json", "csv": "control_single_rectangles_physical.csv", "glog": "control_single_rectangles_physical.g.log", "optimal_path": "optimal_single_rectangles_path.json", "optimal_control_csv": "optimal_single_rectangles_control.csv", "expected": {"finish_reason": sim2.finish_reason, "steps": sim2.step_n, "optimal_length_m": opt2.get("length_m")}})

    c3 = base_config("formation_7_adaptive_split", 7, "triangle", 0.38, width=2.6, alg="adaptive_clusters")
    c3["obstacles"] = [rect("block1", 4.2, 0.0, 0.65, 0.65), rect("block2", 6.2, 0.45, 0.70, 0.55), rect("block3", 7.8, -0.45, 0.55, 0.55)]
    dump_json("formation_7_adaptive_split.json", c3)
    sim3 = run_and_log(c3, "control_formation_7_adaptive_split")
    examples.append({"title": "7 агентов, adaptive split/rejoin вокруг прямоугольников", "config": "formation_7_adaptive_split.json", "csv": "control_formation_7_adaptive_split.csv", "glog": "control_formation_7_adaptive_split.g.log", "expected": {"finish_reason": sim3.finish_reason, "steps": sim3.step_n}})

    c4 = base_config("formation_6_dynamic_popup", 6, "hex", 0.36, width=2.5, alg="completion_astar")
    c4["obstacles"] = [rect("static-left", 3.8, -0.45, 0.55, 0.5), rect("popup", 6.0, 0.0, 0.65, 0.55, appear=6.0, disappear=18.0, label="pop-up"), circle("round", 8.0, 0.42, 0.28)]
    dump_json("formation_6_dynamic_popup.json", c4)
    sim4 = run_and_log(c4, "control_formation_6_dynamic_popup")
    examples.append({"title": "6 агентов, препятствие появляется и исчезает", "config": "formation_6_dynamic_popup.json", "csv": "control_formation_6_dynamic_popup.csv", "glog": "control_formation_6_dynamic_popup.g.log", "expected": {"finish_reason": sim4.finish_reason, "steps": sim4.step_n}})

    c5 = base_config("formation_4_gate_rejoin", 4, "square", 0.42, width=2.1, alg="completion_astar")
    c5["obstacles"] = [rect("gate-left", 5.2, -0.58, 0.85, 0.50), rect("gate-right", 5.2, 0.58, 0.85, 0.50), rect("after-gate", 7.4, 0.0, 0.45, 0.45)]
    dump_json("formation_4_gate_rejoin.json", c5)
    sim5 = run_and_log(c5, "control_formation_4_gate_rejoin")
    examples.append({"title": "4 агента, проход через ворота и сборка строя", "config": "formation_4_gate_rejoin.json", "csv": "control_formation_4_gate_rejoin.csv", "glog": "control_formation_4_gate_rejoin.g.log", "expected": {"finish_reason": sim5.finish_reason, "steps": sim5.step_n}})

    dump_json("index.json", {"examples": examples})
    print(json.dumps({"examples": examples}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
