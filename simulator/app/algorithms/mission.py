from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .base import Planner
from ..domain import PlannerCommand, RobotState
from ..geometry import obstacle_active, obstacle_center, obstacle_approx_radius, point_clearance_to_obstacle, segment_clearance_to_obstacle
from ..path_planning import astar_grid, pick_next_waypoint
from ..utils import Vec, add, clamp, clamp_vec, dist, mul, norm, sub, unit


def _planner_safety_margin(sim, robot: RobotState) -> float:
    """Margin used by graph planners, kept below hard physics margin.

    v6.2 was over-inflating obstacles for narrow physical tests: A* found no
    path, then the fallback selected a straight staging point and the simulator
    stopped at the obstacle.  The hard integrator still enforces the physical
    collision margin; this smaller planning margin keeps the route search from
    declaring valid side passages impossible.
    """
    requested = sim.config.limits.safety_margin_m + sim.config.limits.obstacle_margin_m
    return max(0.025, min(requested, 0.105))


def completion_guard_velocity(sim, robot: RobotState, obstacles, target: Vec, current_v: Vec, speed_scale: float, reason: str, force: bool = False) -> Tuple[Vec, str]:
    """Progress fallback for local baselines under completion-first objectives.

    APF/DWA/RVO are intentionally local and may stop in a local minimum.  The
    mission setting can still require completion; in that case this small guard
    asks the same observed/local map for a per-agent A* waypoint only when the
    local command has essentially no progress.
    """
    if sim.config.objective.completion_policy not in ("finish_first", "min_time_all_powered"):
        return current_v, reason
    to_goal = unit(sub(target, robot.pos), fallback=(1.0, 0.0))
    progress = current_v[0] * to_goal[0] + current_v[1] * to_goal[1]
    if (not force) and norm(current_v) > 0.035 and progress > 0.020:
        return current_v, reason
    try:
        result = astar_grid(robot.pos, target, sim.config.mission_length_m, sim.config.corridor_width_m, obstacles, sim.time_s, robot.radius_m, _planner_safety_margin(sim, robot), sim.config.grid_resolution_m)
        if result.status == "ok" and len(result.path) >= 2:
            waypoint = pick_next_waypoint(result.path, robot.pos, max(sim.config.limits.min_route_lookahead_m, robot.spec.max_speed_mps * 1.0))
            guarded = clamp_vec(mul(unit(sub(waypoint, robot.pos), fallback=to_goal), robot.spec.max_speed_mps * max(0.72, speed_scale)), robot.spec.max_speed_mps)
            if norm(guarded) > 0.01:
                return guarded, (reason + "; completion_guard_astar").strip("; ")
    except Exception:
        pass
    # Last-resort completion staging point for local baselines.  This is not a
    # new planner; it is an escape waypoint around the nearest known blocker so
    # DWA/RVO do not keep selecting a locally safe but non-completing command.
    try:
        nearest = None
        best_dx = float("inf")
        for obs in obstacles:
            if not obstacle_active(obs, sim.time_s):
                continue
            c = obstacle_center(obs, sim.time_s)
            dx = c[0] - robot.x
            if -0.25 <= dx < best_dx:
                best_dx = dx
                nearest = obs
        if nearest is not None:
            c = obstacle_center(nearest, sim.time_s)
            kind = getattr(nearest, "kind", "rectangle")
            cross_half = (float(getattr(nearest, "height_m", 0.0) or 0.0) / 2.0) if kind == "rectangle" else obstacle_approx_radius(nearest)
            lane_half = sim.config.corridor_width_m / 2.0
            margin = robot.radius_m + sim.config.limits.safety_margin_m + sim.config.limits.obstacle_margin_m
            upper_room = lane_half - (c[1] + cross_half + margin)
            lower_room = (c[1] - cross_half - margin) - (-lane_half)
            side = 1.0 if upper_room >= lower_room else -1.0
            y = c[1] + side * (cross_half + margin + 0.10)
            y = clamp(y, -lane_half + robot.radius_m + sim.config.limits.safety_margin_m, lane_half - robot.radius_m - sim.config.limits.safety_margin_m)
            x = min(target[0], max(robot.x + 0.28, c[0] + obstacle_approx_radius(nearest) + 0.25))
            waypoint = (x, y)
            guarded = clamp_vec(mul(unit(sub(waypoint, robot.pos), fallback=to_goal), robot.spec.max_speed_mps * max(0.72, speed_scale)), robot.spec.max_speed_mps)
            if norm(guarded) > 0.01:
                return guarded, (reason + "; completion_guard_staging").strip("; ")
    except Exception:
        pass
    return current_v, reason


class CompletionAStarPlanner(Planner):
    key = "completion_astar"
    label = "Completion-first A* per agent"
    description = "Each agent plans to its final assigned place using only observed/shared obstacles; completion is prioritized over centroid optimality."

    def __init__(self, speed_scale: float = 0.82, policy_name: str = "finish_first"):
        self.speed_scale = speed_scale
        self.policy_name = policy_name

    def plan(self, sim) -> Dict[str, PlannerCommand]:
        sim.begin_planning_cycle()
        sim.refresh_shared_memory()
        obstacles = sim.planning_obstacles()
        final_targets = sim.final_targets()
        commands: Dict[str, PlannerCommand] = {}
        clusters: Dict[str, List[str]] = {"upper": [], "lower": [], "center": [], "blocked": [], "rejoined": []}
        blockers = sim.front_blockers(obstacles)
        adaptive_keys = ("smooth_adaptive_clusters", "adaptive_clusters", "cluster_adaptive")
        active_blockers = (blockers or self._known_route_blockers_v61(sim, obstacles)) if self.key in adaptive_keys else blockers
        bypass_assignments = self._bypass_assignments(sim, active_blockers) if (self.key in adaptive_keys and active_blockers) else {}
        path_debug = []

        for robot in sim.robots:
            if not robot.powered or not robot.alive:
                commands[robot.id] = PlannerCommand(vx=0.0, vy=0.0, target=robot.pos, cluster_id="unpowered", role="hold", reason="no_power")
                continue

            final_target = final_targets.get(robot.id, (sim.config.mission_length_m, 0.0))
            bypass_info = bypass_assignments.get(robot.id)
            # During split/bypass, plan to a temporary slot in a free transverse interval.
            # After the blocker is passed, the same algorithm returns to the final formation slot.
            target = bypass_info["target"] if bypass_info else final_target
            remaining = dist(robot.pos, final_target)

            arrival_tol = sim.config.finish_agent_tolerance_m * 0.55
            if len(sim.robots) > 1 and sim.config.formation is not None:
                arrival_tol = min(arrival_tol, sim.config.formation.edge_tolerance_m * 0.45, 0.055)
            if remaining < arrival_tol:
                control_point = sim.smooth_control_point(robot, final_target, key="control_point")
                cluster_id = sim.stabilize_cluster_id(robot, "rejoined")
                commands[robot.id] = PlannerCommand(vx=0.0, vy=0.0, target=control_point, cluster_id=cluster_id, role="arrived", reason="target_reached", led="-G-", debug={"final_target": final_target, "control_point": control_point})
                clusters["rejoined"].append(robot.id)
                sim.remember_planned_route(robot, [robot.pos, final_target], final_target, control_point, final_target, cluster_id, "ok", "target_reached", 0.0)
                continue

            result = self._path_for_robot(sim, robot, target, obstacles)
            if result.status != "ok":
                waypoint, reason = self._fallback_waypoint(sim, robot, target, obstacles)
                raw_cluster_id = str(bypass_info.get("cluster_id", "blocked")) if bypass_info else "blocked"
                path_for_waypoint = [robot.pos, waypoint, target]
                robot.blocked_steps += 1
                route_length = float("inf")
            else:
                path_for_waypoint = self._subpath_from_robot(robot, result.path, target)
                lookahead = max(sim.config.limits.min_route_lookahead_m, min(robot.spec.max_speed_mps * 0.60, 0.50))
                waypoint = pick_next_waypoint(path_for_waypoint, robot.pos, lookahead_m=lookahead)
                if bypass_info:
                    reason = "adaptive_split_to_bypass_target" if result.reason != "cached" else "cached_adaptive_split"
                    raw_cluster_id = str(bypass_info.get("cluster_id", "center"))
                else:
                    reason = "path_to_final_target" if result.reason != "cached" else "cached_path_to_final_target"
                    raw_cluster_id = self._cluster_from_path(robot, path_for_waypoint)
                robot.blocked_steps = 0
                route_length = result.length_m

            cluster_id = sim.stabilize_cluster_id(robot, raw_cluster_id)
            clusters.setdefault(cluster_id, []).append(robot.id)
            control_point = sim.smooth_control_point(robot, waypoint, key="control_point")
            path_debug.append({
                "robot": robot.id,
                "status": result.status,
                "length_m": round(route_length, 4) if math.isfinite(route_length) else None,
                "cached": getattr(result, "reason", "") == "cached",
                "raw_cluster": raw_cluster_id,
                "cluster": cluster_id,
                "waypoint": waypoint,
                "control_point": control_point,
                "temporary_target": target,
                "final_target": final_target,
                "path": path_for_waypoint[:12],
            })
            sim.remember_planned_route(robot, path_for_waypoint, waypoint, control_point, final_target, cluster_id, result.status, reason, route_length)

            direction = sub(control_point, robot.pos)
            d = norm(direction)
            if d < 1e-6:
                desired = (0.0, 0.0)
            else:
                params = sim.algorithm_params_for(self.key)
                speed_scale = float(params.get("speed_scale", self.speed_scale))
                max_speed = robot.spec.max_speed_mps * speed_scale
                objective = getattr(sim.config.objective, "objective_function", "finish_first")
                if objective == "min_makespan":
                    max_speed = robot.spec.max_speed_mps * float(params.get("speed_scale", max(1.0, self.speed_scale)))
                elif objective == "min_total_energy":
                    max_speed = robot.spec.max_speed_mps * float(params.get("energy_speed_scale", min(0.60, self.speed_scale)))
                # Completion-first: do not stop in front of a detour. Slow down only near the final slot.
                speed = min(max_speed, max(0.07, 1.30 * max(remaining, 0.20)))
                desired = mul(unit(direction), speed)
                # Near finish: add soft correction to restore exact formation embedding.
                if sim.current_centroid()[0] > sim.config.mission_length_m - 1.5:
                    desired = add(desired, clamp_vec(mul(sub(final_target, robot.pos), 0.45), sim.config.limits.max_reconfig_speed_mps))
            commands[robot.id] = PlannerCommand(
                vx=desired[0],
                vy=desired[1],
                target=control_point,
                cluster_id=cluster_id,
                role=str(bypass_info.get("role", "bypass")) if bypass_info else ("blocked" if cluster_id == "blocked" else "member"),
                reason=reason,
                debug={"waypoint": waypoint, "control_point": control_point, "temporary_target": target, "final_target": final_target, "objective": getattr(sim.config.objective, "objective_function", self.policy_name)},
            )

        sim.last_cluster_plan = {
            "mode": "split" if bypass_assignments else ("rejoin" if active_blockers else "formation_cruise"),
            "clusters": self._cluster_payload(clusters, bypass_assignments),
            "blockers": [b.id or "obs" for b in active_blockers],
            "free_intervals": getattr(self, "_last_free_intervals", []),
            "visual_model": "v2_style_visible_bypass" if self.key in adaptive_keys else "per_agent_astar",
            "path_debug": path_debug[:20],
            "route_count": len(path_debug),
            "smooth_reconfiguration": bool(sim.config.limits.smooth_reconfiguration),
            "setpoint_max_rate_mps": sim.config.limits.setpoint_max_rate_mps,
            "decision_basis": getattr(sim.config.objective, "objective_function", sim.config.objective.completion_policy),
            "perception_mode": sim.config.perception.mode,
            "known_obstacle_count": len(obstacles),
            "sonar_dependency": "global_debug uses full active map; local uses sonar-visible + optional shared memory observations",
            "commands": [
                {
                    "robot": rid,
                    "vx": round(cmd.vx, 4),
                    "vy": round(cmd.vy, 4),
                    "target": {"x": round(cmd.target[0], 4), "y": round(cmd.target[1], 4)} if cmd.target else None,
                    "cluster_id": cmd.cluster_id,
                    "reason": cmd.reason,
                }
                for rid, cmd in list(commands.items())[:40]
            ],
        }
        return commands

    def _known_route_blockers_v61(self, sim, obstacles) -> List[object]:
        """Known obstacles ahead of the formation, inside sonar/shared-memory range.

        This restores the early-version visible bypass behavior: adaptive planners
        begin split before `front_blockers()` becomes true, but still only from
        obstacles already in the local/shared planning set.
        """
        c = sim.current_centroid()
        lane_half = sim.config.corridor_width_m / 2.0
        span = sim.formation_y_span()
        max_r = max([r.radius_m for r in sim.robots] or [0.10])
        y_reach = min(lane_half, span / 2.0 + max_r + 0.45)
        sonic = max([r.spec.sonic_range_m for r in sim.robots if r.powered and r.alive] or [4.0])
        x_max = min(sim.config.start_x_m + sim.config.mission_length_m, c[0] + min(5.0, sonic + 0.9))
        out = []
        for obs in obstacles:
            if not obstacle_active(obs, sim.time_s):
                continue
            ox, oy = obstacle_center(obs, sim.time_s)
            approx = obstacle_approx_radius(obs)
            if ox + approx < c[0] - 0.30 or ox - approx > x_max:
                continue
            if abs(oy) - approx <= y_reach:
                out.append(obs)
        out.sort(key=lambda o: obstacle_center(o, sim.time_s)[0])
        return out

    def _bypass_assignments(self, sim, blockers) -> Dict[str, Dict[str, object]]:
        powered = [r for r in sim.robots if r.powered and r.alive]
        if not powered:
            self._last_free_intervals = []
            return {}
        max_radius = max([r.radius_m for r in powered] or [0.1])
        margin = max_radius + sim.config.limits.safety_margin_m + sim.config.limits.obstacle_margin_m + 0.04
        intervals = self._free_y_intervals(sim, blockers, margin)
        self._last_free_intervals = [{"lo": round(lo, 3), "hi": round(hi, 3), "width": round(hi - lo, 3)} for lo, hi in intervals]
        if not intervals:
            return {}
        centroid = sim.current_centroid()
        right_edges = [obstacle_center(o, sim.time_s)[0] + obstacle_approx_radius(o) for o in blockers]
        x_ahead = min(sim.config.mission_length_m, max(centroid[0] + 0.70, max(right_edges or [centroid[0]]) + 0.65))
        # Keep ordering by formation offset to avoid agents crossing each other during split.
        ordered = sorted(powered, key=lambda r: (sim.formation_offsets.get(r.id, (0.0, 0.0))[1], r.y, r.id))
        y_slots = self._distributed_y_positions(intervals, len(ordered))
        assignments: Dict[str, Dict[str, object]] = {}
        for idx, (robot, y_slot) in enumerate(zip(ordered, y_slots)):
            cluster_id = "upper" if y_slot > 0.12 else ("lower" if y_slot < -0.12 else "center")
            # Stagger temporary targets along x.  Without this, several agents try to occupy
            # the same bypass cross-section and the safety filter can legitimately freeze them.
            x_slot = min(sim.config.mission_length_m, x_ahead + 0.16 * idx)
            assignments[robot.id] = {
                "target": (x_slot, y_slot),
                "cluster_id": cluster_id,
                "role": "bypass" if cluster_id in ("upper", "lower") else "passage",
                "reason": "free_interval_bypass",
            }
        return assignments

    def _free_y_intervals(self, sim, blockers, margin: float) -> List[Tuple[float, float]]:
        lane_half = sim.config.corridor_width_m / 2.0
        lo0 = -lane_half + margin
        hi0 = lane_half - margin
        if lo0 >= hi0:
            return []
        intervals: List[Tuple[float, float]] = [(lo0, hi0)]
        for obs in blockers:
            if not obstacle_active(obs, sim.time_s):
                continue
            c = obstacle_center(obs, sim.time_s)
            kind = getattr(obs, "kind", "rectangle")
            if kind == "rectangle" and abs(getattr(obs, "rotation_deg", 0.0) or 0.0) < 1e-6:
                cross_half = float(getattr(obs, "height_m", 0.0) or 0.0) / 2.0
            elif kind == "circle":
                cross_half = float(getattr(obs, "radius_m", obstacle_approx_radius(obs)) or obstacle_approx_radius(obs))
            else:
                cross_half = obstacle_approx_radius(obs)
            extent = cross_half + margin
            block_lo = c[1] - extent
            block_hi = c[1] + extent
            new_intervals: List[Tuple[float, float]] = []
            for lo, hi in intervals:
                if block_hi <= lo or block_lo >= hi:
                    new_intervals.append((lo, hi))
                    continue
                if block_lo - lo > 0.08:
                    new_intervals.append((lo, min(hi, block_lo)))
                if hi - block_hi > 0.08:
                    new_intervals.append((max(lo, block_hi), hi))
            intervals = new_intervals
        return sorted([(lo, hi) for lo, hi in intervals if hi - lo > 0.08])

    def _distributed_y_positions(self, intervals: List[Tuple[float, float]], n: int) -> List[float]:
        if n <= 0 or not intervals:
            return []
        widths = [max(0.0, hi - lo) for lo, hi in intervals]
        total = sum(widths)
        if total <= 1e-9:
            lo, hi = intervals[0]
            return [(lo + hi) * 0.5 for _ in range(n)]
        # Allocate agents approximately proportional to free width, then fix rounding.
        raw = [n * w / total for w in widths]
        counts = [int(math.floor(v)) for v in raw]
        for i, w in enumerate(widths):
            if w > 0 and counts[i] == 0 and sum(counts) < n:
                counts[i] = 1
        while sum(counts) < n:
            best = max(range(len(raw)), key=lambda i: raw[i] - counts[i])
            counts[best] += 1
        while sum(counts) > n:
            candidates = [i for i, c in enumerate(counts) if c > 0]
            worst = min(candidates, key=lambda i: raw[i] - counts[i])
            counts[worst] -= 1
        slots: List[float] = []
        for (lo, hi), count in zip(intervals, counts):
            if count <= 0:
                continue
            if count == 1:
                slots.append((lo + hi) * 0.5)
            else:
                span = hi - lo
                for j in range(count):
                    slots.append(lo + span * (j + 1) / (count + 1))
        slots.sort()
        # Rare rounding edge case.
        while len(slots) < n:
            slots.append((intervals[-1][0] + intervals[-1][1]) * 0.5)
        return slots[:n]

    def _cluster_payload(self, clusters: Dict[str, List[str]], assignments: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        for cluster_id, members in clusters.items():
            if not members:
                continue
            targets = []
            for rid in members:
                info = assignments.get(rid)
                if info and isinstance(info.get("target"), tuple):
                    t = info["target"]
                    targets.append({"robot_id": rid, "x": t[0], "y": t[1]})
            item: Dict[str, object] = {"id": cluster_id, "members": members}
            if targets:
                item["temporary_targets"] = targets
            payload.append(item)
        return payload

    def _path_for_robot(self, sim, robot: RobotState, target: Vec, obstacles):
        fingerprint = sim.planning_fingerprint(obstacles)
        cache = sim.path_cache.get(robot.id)
        period = max(1, sim.config.limits.replan_period_steps)
        should_replan = (sim.step_n % period == 0) or cache is None or robot.blocked_steps > 2
        if cache is not None:
            old_target = cache.get("target", target)
            if dist(old_target, target) > 0.08 or cache.get("fingerprint") != fingerprint:
                should_replan = True
            if sim.step_n - int(cache.get("step", -999999)) > period * 6:
                should_replan = True
            # If the robot drifted far from the cached route, replan.
            path = cache.get("path", [])
            if path and min(dist(robot.pos, p) for p in path) > 0.60:
                should_replan = True
        if not should_replan and cache is not None:
            from ..path_planning import PathResult, path_length
            path = cache.get("path", [])
            return PathResult(path, path_length(path), "ok", expanded=0, reason="cached")
        result = astar_grid(
            start=robot.pos,
            goal=target,
            mission_length_m=sim.config.mission_length_m,
            corridor_width_m=sim.config.corridor_width_m,
            obstacles=obstacles,
            time_s=sim.time_s,
            agent_radius_m=robot.radius_m,
            safety_margin_m=_planner_safety_margin(sim, robot),
            resolution_m=sim.config.grid_resolution_m,
            x_pad_m=0.9,
        )
        if result.status == "ok":
            sim.path_cache[robot.id] = {"path": result.path, "target": target, "fingerprint": fingerprint, "step": sim.step_n}
        return result

    def _subpath_from_robot(self, robot: RobotState, path: List[Vec], target: Vec) -> List[Vec]:
        if not path:
            return [robot.pos, target]
        nearest = min(range(len(path)), key=lambda i: dist(robot.pos, path[i]))
        # Drop already passed points, keep final target.
        tail = path[min(nearest + 1, len(path) - 1):]
        if not tail:
            tail = [target]
        return [robot.pos] + tail

    def _cluster_from_path(self, robot: RobotState, path: List[Vec]) -> str:
        if len(path) < 3:
            return "center"
        # First meters of the route determine the bypass cluster.
        ys = [p[1] for p in path[: min(7, len(path))]]
        avg_y = sum(ys) / len(ys)
        if avg_y - robot.y > 0.16 or max(ys) > 0.25:
            return "upper"
        if avg_y - robot.y < -0.16 or min(ys) < -0.25:
            return "lower"
        return "center"

    def _fallback_waypoint(self, sim, robot: RobotState, target: Vec, obstacles) -> Tuple[Vec, str]:
        # Completion remains more important than local optimality: if A* fails, try a corridor-side staging point ahead.
        lane_half = sim.config.corridor_width_m / 2.0
        margin = robot.radius_m + sim.config.limits.safety_margin_m + sim.config.limits.obstacle_margin_m
        candidates: List[Vec] = []
        x = min(sim.config.mission_length_m, max(robot.x + 0.45, robot.x + robot.spec.max_speed_mps * 1.5))
        for y in [0.0, lane_half - margin - 0.04, -lane_half + margin + 0.04, 0.55 * lane_half, -0.55 * lane_half]:
            y = clamp(y, -lane_half + margin, lane_half - margin)
            candidates.append((x, y))
        candidates.append(target)
        best = robot.pos
        best_score = -1e18
        for cand in candidates:
            if abs(cand[1]) > lane_half - margin:
                continue
            endpoint_clearance = min([point_clearance_to_obstacle(cand, o, sim.time_s) - margin for o in obstacles if obstacle_active(o, sim.time_s)] or [10.0])
            swept_clearance = min([segment_clearance_to_obstacle(robot.pos, cand, o, sim.time_s, samples_per_m=24.0) - margin for o in obstacles if obstacle_active(o, sim.time_s)] or [10.0])
            clearance = min(endpoint_clearance, swept_clearance)
            progress = cand[0] - robot.x
            target_gain = -0.25 * dist(cand, target)
            # Do not choose a straight staging point if the segment crosses an obstacle.
            score = progress + target_gain + 2.2 * min(0.5, clearance)
            if clearance > -0.005 and score > best_score:
                best = cand
                best_score = score
        return best, "fallback_staging_point"


class SmoothAdaptiveClustersPlanner(CompletionAStarPlanner):
    key = "smooth_adaptive_clusters"
    label = "Smooth adaptive clusters"
    description = "Default mission algorithm: per-agent completion A* with smoothed split/bypass/rejoin setpoints and route previews."

    def __init__(self):
        super().__init__(speed_scale=0.82, policy_name="finish_first")


class AdaptiveClustersPlanner(CompletionAStarPlanner):
    key = "adaptive_clusters"
    label = "Adaptive clusters"
    description = "Completion-first individual A* with explicit split/rejoin labels and per-agent final targets."

    def __init__(self):
        super().__init__(speed_scale=0.86, policy_name="finish_first")


class TimePriorityPlanner(CompletionAStarPlanner):
    key = "time_priority"
    label = "Minimum time for powered agents"
    description = "Same onboard algorithm but with aggressive speed and lower mid-course formation pressure."

    def __init__(self):
        super().__init__(speed_scale=1.0, policy_name="min_time_all_powered")


class EnergySaverPlanner(CompletionAStarPlanner):
    key = "energy_saver"
    label = "Energy saver"
    description = "Same route logic with lower speed and smoother corrections."

    def __init__(self):
        super().__init__(speed_scale=0.55, policy_name="min_energy")


class VirtualStructurePlanner(Planner):
    key = "virtual_structure"
    label = "Virtual structure baseline"
    description = "Plans the centroid path and drags the whole formation; useful as a baseline that can fail at edges."

    def plan(self, sim) -> Dict[str, PlannerCommand]:
        sim.begin_planning_cycle()
        sim.refresh_shared_memory()
        obstacles = sim.planning_obstacles()
        c = sim.current_centroid()
        goal = (sim.config.mission_length_m, 0.0)
        # Inflate by group half-width, so this is conservative. If no route, it falls back to direct.
        radius = max([r.radius_m for r in sim.robots] or [0.1]) + sim.formation_y_span() * 0.5
        result = astar_grid(c, goal, sim.config.mission_length_m, sim.config.corridor_width_m, obstacles, sim.time_s, radius, sim.config.limits.safety_margin_m, sim.config.grid_resolution_m)
        waypoint = pick_next_waypoint(result.path, c, 0.40) if result.status == "ok" else goal
        base_v = clamp_vec(mul(unit(sub(waypoint, c)), sim.config.default_agent.max_speed_mps * 0.72), sim.config.default_agent.max_speed_mps * 0.72)
        targets = {rid: add(waypoint, off) for rid, off in sim.formation_offsets.items()}
        commands: Dict[str, PlannerCommand] = {}
        for robot in sim.robots:
            target = targets.get(robot.id, add(goal, sim.formation_offsets.get(robot.id, (0.0, 0.0))))
            raw_control = pick_next_waypoint([robot.pos, target], robot.pos, max(sim.config.limits.min_route_lookahead_m, robot.spec.max_speed_mps * 1.3))
            control = sim.smooth_control_point(robot, raw_control, key="control_point")
            corr = clamp_vec(mul(sub(control, robot.pos), 1.0), sim.config.limits.max_reconfig_speed_mps)
            v = clamp_vec(add(base_v, corr), robot.spec.max_speed_mps * 0.82)
            sim.remember_planned_route(robot, [robot.pos, target], raw_control, control, target, "main", "ok", "centroid_virtual_structure", dist(robot.pos, target))
            commands[robot.id] = PlannerCommand(vx=v[0], vy=v[1], target=control, cluster_id="main", role="member", reason="centroid_virtual_structure")
        sim.last_cluster_plan = {"mode": "virtual_structure", "clusters": [{"id": "main", "members": [r.id for r in sim.robots]}], "blockers": [], "route_count": len(sim.planned_routes), "smooth_reconfiguration": bool(sim.config.limits.smooth_reconfiguration), "decision_basis": "shortest_centroid"}
        return commands


class APFReactivePlanner(Planner):
    key = "apf_reactive"
    label = "Reactive APF baseline"
    description = "Attraction to final targets plus repulsion from observed obstacles and other agents."

    def plan(self, sim) -> Dict[str, PlannerCommand]:
        sim.begin_planning_cycle()
        sim.refresh_shared_memory()
        obstacles = sim.planning_obstacles()
        targets = sim.final_targets()
        lane_half = sim.config.corridor_width_m / 2.0
        commands: Dict[str, PlannerCommand] = {}
        for robot in sim.robots:
            target = targets.get(robot.id, (sim.config.mission_length_m, 0.0))
            v = clamp_vec(mul(sub(target, robot.pos), 0.42), robot.spec.max_speed_mps * 0.72)
            for obs in obstacles:
                if not obstacle_active(obs, sim.time_s):
                    continue
                dclear = point_clearance_to_obstacle(robot.pos, obs, sim.time_s) - robot.radius_m
                if dclear < 0.70:
                    c = obstacle_center(obs, sim.time_s)
                    away = unit(sub(robot.pos, c), fallback=(0.0, 1.0))
                    tangent = (-away[1], away[0])
                    # Tangent helps APF not to get permanently stuck in front of the obstacle.
                    sign = 1.0 if robot.y <= c[1] else -1.0
                    gain = min(0.35, 0.18 / max(0.05, dclear + 0.12))
                    v = add(v, add(mul(away, gain), mul(tangent, sign * gain * 0.55)))
            for other in sim.robots:
                if other.id == robot.id:
                    continue
                d = dist(robot.pos, other.pos)
                safe = robot.radius_m + other.radius_m + sim.config.limits.robot_robot_margin_m
                if d < safe + 0.20:
                    v = add(v, mul(unit(sub(robot.pos, other.pos), fallback=(0.0, 1.0)), 0.13 * (safe + 0.20 - d) / (safe + 0.20)))
            margin = robot.radius_m + sim.config.limits.safety_margin_m
            if robot.y > lane_half - margin:
                v = add(v, (0.0, -0.30))
            if robot.y < -lane_half + margin:
                v = add(v, (0.0, 0.30))
            v = clamp_vec(v, robot.spec.max_speed_mps * 0.72)
            v, apf_reason = completion_guard_velocity(sim, robot, obstacles, target, v, 0.72, "apf_reactive")
            raw_control = add(robot.pos, v)
            control = sim.smooth_control_point(robot, raw_control, key="control_point")
            sim.remember_planned_route(robot, [robot.pos, control, target], control, control, target, "apf", "ok", apf_reason, dist(robot.pos, target))
            commands[robot.id] = PlannerCommand(vx=v[0], vy=v[1], target=control, cluster_id="apf", role="member", reason=apf_reason)
        sim.last_cluster_plan = {"mode": "apf", "clusters": [{"id": "apf", "members": [r.id for r in sim.robots]}], "blockers": [o.id for o in obstacles], "route_count": len(sim.planned_routes), "smooth_reconfiguration": bool(sim.config.limits.smooth_reconfiguration), "decision_basis": "reactive"}
        return commands


class AStarVirtualStructurePlanner(VirtualStructurePlanner):
    key = "astar_vs"
    label = "A* + virtual structure"
    description = "Global grid A* for the centroid with a virtual rigid formation around it; strong baseline for shortest centroid path, weak when edge agents meet obstacles."


class DWAPlanner(Planner):
    key = "dwa"
    label = "Dynamic Window Approach baseline"
    description = "Local velocity-space rollout. It samples admissible velocity directions and chooses the best safe short-horizon command."

    def plan(self, sim) -> Dict[str, PlannerCommand]:
        sim.begin_planning_cycle()
        sim.refresh_shared_memory()
        obstacles = sim.planning_obstacles()
        targets = sim.final_targets()
        commands: Dict[str, PlannerCommand] = {}
        lane_half = sim.config.corridor_width_m / 2.0
        params = sim.algorithm_params_for(self.key)
        horizon = float(params.get("horizon_s", 1.4))
        speed_scale = float(params.get("speed_scale", 0.78))
        heading_samples = int(params.get("heading_samples", 13))
        heading_span = float(params.get("heading_span_rad", 1.35))
        for robot in sim.robots:
            if not robot.powered or not robot.alive:
                commands[robot.id] = PlannerCommand(target=robot.pos, cluster_id="unpowered", role="hold", reason="no_power")
                continue
            target = targets.get(robot.id, (sim.config.mission_length_m, 0.0))
            to_goal = unit(sub(target, robot.pos), fallback=(1.0, 0.0))
            base_ang = math.atan2(to_goal[1], to_goal[0])
            candidates: List[Vec] = [(0.0, 0.0)]
            for si, sscale in enumerate([1.0, 0.75, 0.50, 0.30]):
                for hi in range(max(3, heading_samples)):
                    frac = 0.0 if heading_samples <= 1 else (hi / (heading_samples - 1) - 0.5)
                    ang = base_ang + frac * 2.0 * heading_span
                    v = robot.spec.max_speed_mps * speed_scale * sscale
                    candidates.append((v * math.cos(ang), v * math.sin(ang)))
            best_v = (0.0, 0.0)
            best_score = -1e18
            best_reason = "dwa_hold"
            for cand in candidates:
                cand = clamp_vec(cand, robot.spec.max_speed_mps * speed_scale)
                pred = (robot.x + cand[0] * horizon, robot.y + cand[1] * horizon)
                if abs(pred[1]) > lane_half - robot.radius_m - sim.config.limits.safety_margin_m:
                    continue
                obs_clear = min([point_clearance_to_obstacle(pred, o, sim.time_s) - robot.radius_m - sim.config.limits.obstacle_margin_m for o in obstacles] or [10.0])
                rob_clear = 10.0
                for other in sim.robots:
                    if other.id == robot.id:
                        continue
                    op = (other.x + other.last_cmd_vx * horizon, other.y + other.last_cmd_vy * horizon)
                    rob_clear = min(rob_clear, dist(pred, op) - robot.radius_m - other.radius_m - sim.config.limits.robot_robot_margin_m)
                clear = min(obs_clear, rob_clear)
                progress = (pred[0] - robot.x) * to_goal[0] + (pred[1] - robot.y) * to_goal[1]
                goal_score = -0.18 * dist(pred, target)
                clear_score = 2.6 * min(clear, 0.60) + (18.0 * clear if clear < 0 else 0.0)
                smooth_score = -0.10 * dist(cand, (robot.last_cmd_vx, robot.last_cmd_vy))
                score = progress + goal_score + clear_score + smooth_score
                if score > best_score:
                    best_score = score
                    best_v = cand
                    best_reason = "dwa_velocity_rollout"
            best_v, best_reason = completion_guard_velocity(sim, robot, obstacles, target, best_v, speed_scale, best_reason, force=('physical_collision_guard' in (robot.last_reason or '')))
            control = sim.smooth_control_point(robot, add(robot.pos, mul(best_v, max(sim.config.dt_s, 0.25))), key="control_point")
            cluster = "upper" if best_v[1] > 0.07 else "lower" if best_v[1] < -0.07 else "center"
            sim.remember_planned_route(robot, [robot.pos, control, target], control, control, target, cluster, "ok", best_reason, dist(robot.pos, target))
            commands[robot.id] = PlannerCommand(vx=best_v[0], vy=best_v[1], target=control, cluster_id=cluster, role="member", reason=best_reason)
        sim.last_cluster_plan = {"mode": "dwa", "clusters": [], "blockers": [o.id for o in obstacles], "route_count": len(sim.planned_routes), "decision_basis": "velocity-space local rollout"}
        return commands


class RVOLitePlanner(Planner):
    key = "rvo_lite"
    label = "RVO-lite / ORCA-inspired"
    description = "Reciprocal-avoidance-inspired local planner. It starts with each agent's preferred velocity and subtracts local collision-avoidance corrections."

    def plan(self, sim) -> Dict[str, PlannerCommand]:
        sim.begin_planning_cycle()
        sim.refresh_shared_memory()
        obstacles = sim.planning_obstacles()
        targets = sim.final_targets()
        commands: Dict[str, PlannerCommand] = {}
        lane_half = sim.config.corridor_width_m / 2.0
        params = sim.algorithm_params_for(self.key)
        neighbor_gain = float(params.get("neighbor_gain", 0.34))
        obstacle_gain = float(params.get("obstacle_gain", 0.28))
        speed_scale = float(params.get("speed_scale", 0.74))
        for robot in sim.robots:
            if not robot.powered or not robot.alive:
                commands[robot.id] = PlannerCommand(target=robot.pos, cluster_id="unpowered", role="hold", reason="no_power")
                continue
            target = targets.get(robot.id, (sim.config.mission_length_m, 0.0))
            desired = clamp_vec(mul(sub(target, robot.pos), 0.55), robot.spec.max_speed_mps * speed_scale)
            v = desired
            for other in sim.robots:
                if other.id == robot.id:
                    continue
                rel = sub(robot.pos, other.pos)
                d = norm(rel)
                safe = robot.radius_m + other.radius_m + sim.config.limits.robot_robot_margin_m + 0.12
                if d < safe:
                    away = unit(rel, fallback=(0.0, 1.0))
                    # Reciprocal idea: assume both agents share responsibility, hence the 0.5 factor.
                    v = add(v, mul(away, 0.5 * neighbor_gain * (safe - d) / max(0.03, safe)))
            for obs in obstacles:
                dclear = point_clearance_to_obstacle(robot.pos, obs, sim.time_s) - robot.radius_m - sim.config.limits.obstacle_margin_m
                if dclear < 0.65:
                    c = obstacle_center(obs, sim.time_s)
                    away = unit(sub(robot.pos, c), fallback=(0.0, 1.0))
                    tangent = (-away[1], away[0])
                    sign = 1.0 if robot.y <= c[1] else -1.0
                    v = add(v, add(mul(away, obstacle_gain * (0.65 - dclear) / 0.65), mul(tangent, sign * obstacle_gain * 0.45)))
            margin = robot.radius_m + sim.config.limits.safety_margin_m
            if robot.y > lane_half - margin:
                v = add(v, (0.0, -0.26))
            if robot.y < -lane_half + margin:
                v = add(v, (0.0, 0.26))
            v = clamp_vec(v, robot.spec.max_speed_mps * speed_scale)
            v, rvo_reason = completion_guard_velocity(sim, robot, obstacles, target, v, speed_scale, "rvo_lite_local_avoidance", force=('physical_collision_guard' in (robot.last_reason or '')))
            control = sim.smooth_control_point(robot, add(robot.pos, mul(v, max(sim.config.dt_s, 0.25))), key="control_point")
            cluster = "upper" if v[1] > 0.07 else "lower" if v[1] < -0.07 else "center"
            sim.remember_planned_route(robot, [robot.pos, control, target], control, control, target, cluster, "ok", rvo_reason, dist(robot.pos, target))
            commands[robot.id] = PlannerCommand(vx=v[0], vy=v[1], target=control, cluster_id=cluster, role="member", reason=rvo_reason)
        sim.last_cluster_plan = {"mode": "rvo_lite", "clusters": [], "blockers": [o.id for o in obstacles], "route_count": len(sim.planned_routes), "decision_basis": "reciprocal local avoidance"}
        return commands


class APFClassicPlanner(APFReactivePlanner):
    key = "apf"
    label = "Artificial Potential Fields"
    description = "Classic reactive potential-field baseline: goal attraction with obstacle and neighbor repulsion."


class APFPlanner(APFClassicPlanner):
    key = "apf"
    label = "APF / Artificial Potential Field"
    description = "User-facing alias for the classic artificial potential field baseline."


class APFBaselinePlanner(APFReactivePlanner):
    key = "apf"
    label = "APF / Artificial Potential Fields baseline"
    description = "Alias kept from v1: artificial potential fields with target attraction and obstacle/agent repulsion."
