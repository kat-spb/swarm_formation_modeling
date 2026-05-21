from __future__ import annotations

import copy
import csv
import io
import math
import random
import re
import uuid
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .algorithms.registry import make_planner
from .domain import PlannerCommand, RobotState, SharedObstacle
from .formation import desired_distance_matrix, formation_edges_from_matrix, formation_matrices, formation_points, make_formation_spec, validate_formation
from .geometry import (
    obstacle_active,
    obstacle_approx_radius,
    obstacle_center,
    obstacle_id,
    obstacle_to_public,
    point_clearance_to_any_obstacle,
    point_clearance_to_obstacle,
    segment_clearance_to_obstacle,
    validate_obstacle_geometry,
)
from .models import (
    AddObstacleRequest,
    AgentSpec,
    ManualPatch,
    ObstacleSpec,
    SimConfig,
    SingleOptimalRequest,
)
from .path_planning import chaikin_smooth, resample_polyline, route_to_commands, visibility_shortest_path
from .utils import Vec, add, angle_wrap, clamp, clamp_vec, centroid, dist, fmt_float, model_dump_compat, mul, norm, pairwise_matrix, rms, sub, unit


CSV_COLUMNS = [
    # identification
    "mission_id",
    "mission_name",
    "step",
    "time_s",
    "ts_ms",
    "algorithm",
    "completion_policy",
    "robot_id",
    "robot_name",
    "powered",
    "cluster_id",
    "role",
    "perception_mode",
    "visible_obstacle_count",
    "known_obstacle_count",
    # primary measured physical state
    "primary_x_m",
    "primary_y_m",
    "primary_theta_rad",
    "primary_v_mps",
    "primary_omega_radps",
    "primary_pwr_left",
    "primary_pwr_right",
    "primary_sonic_angle_deg",
    "primary_sonic_distance_cm",
    "primary_sonic_obstacle_id",
    "primary_sonic_hit_x_m",
    "primary_sonic_hit_y_m",
    "primary_sonic_visible_count",
    "primary_camera_pan_deg",
    "primary_camera_tilt_deg",
    "primary_tracking_bits",
    "primary_led",
    "primary_buz",
    # command/decision
    "control_cmd_vx_mps",
    "control_cmd_vy_mps",
    "control_cmd_v_mps",
    "control_cmd_omega_radps",
    "control_target_x_m",
    "control_target_y_m",
    "control_reason",
    "tank_key_command",
    "tank_protocol_command",
    "tank_pwm_left",
    "tank_pwm_right",
    "tank_command_note",
    # per-agent metrics
    "agent_path_length_m",
    "agent_energy_proxy",
    "agent_collision_events",
    "agent_near_miss_events",
    "agent_lane_violations",
    "agent_min_obstacle_clearance_m",
    "agent_min_robot_clearance_m",
    "agent_goal_distance_m",
    # group metrics/control metrics
    "group_centroid_x_m",
    "group_centroid_y_m",
    "group_centroid_to_goal_m",
    "group_centroid_path_m",
    "group_effective_centroid_length_m",
    "group_formation_rms_m",
    "group_formation_bad_pairs",
    "group_min_clearance_m",
    "group_collision_steps",
    "group_lane_violations",
    "group_total_energy_proxy",
    "group_sum_agent_path_m",
    "group_mean_instant_speed_mps",
    "group_average_speed_mps",
    "group_active_cluster_count",
    "done",
    "finish_reason",
]


class SwarmSim:
    def __init__(self, config: SimConfig):
        self.config = self._normalize_config(copy.deepcopy(config))
        self.id = uuid.uuid4().hex[:10]
        self.rng = random.Random(self.config.seed)
        self.step_n = 0
        self.time_s = 0.0
        self.done = False
        self.finish_reason = "running"
        self.planner_key = self.config.algorithm
        self.planner = make_planner(self.planner_key)
        self.formation_offsets = formation_points(self.config.formation)  # type: ignore[arg-type]
        self.formation_ids = [n.id for n in self.config.formation.nodes]  # type: ignore[union-attr]
        self.desired_matrix = desired_distance_matrix(self.config.formation)  # type: ignore[arg-type]
        self.formation_edges = formation_edges_from_matrix(self.config.formation)  # type: ignore[arg-type]
        self.robots: List[RobotState] = []
        self.shared_obstacles: Dict[str, SharedObstacle] = {}
        self.history: List[Vec] = []
        self.centroid_path_length_m = 0.0
        self._last_centroid: Optional[Vec] = None
        self.collision_steps = 0
        self.near_miss_steps = 0
        self.lane_violations_group = 0
        self.total_energy_proxy = 0.0
        self.events: List[dict] = []
        self.csv_rows: List[dict] = []
        self.last_cluster_plan: Dict[str, object] = {"mode": "init", "clusters": [], "blockers": [], "decision_basis": self.config.objective.completion_policy}
        self.path_cache: Dict[str, Dict[str, object]] = {}
        self.plan_memory: Dict[str, Dict[str, object]] = {}
        self.planned_routes: List[Dict[str, object]] = []
        self.agent_trails: Dict[str, List[Vec]] = {}
        self._init_robots()
        self.agent_trails = {r.id: [r.pos] for r in self.robots}
        self._init_obstacles()
        self._last_centroid = self.current_centroid()
        self.history.append(self._last_centroid)
        self._measure_sensors()
        self._record_csv_rows()

    # ----- setup and normalization -------------------------------------------------
    def _normalize_config(self, cfg: SimConfig) -> SimConfig:
        if cfg.formation is None or not cfg.formation.nodes:
            shape = "triangle" if cfg.formation_shape == "custom" else cfg.formation_shape
            cfg.formation = make_formation_spec(shape, cfg.n_agents, cfg.formation_spacing_m)
        else:
            # If matrix is missing, compute it from the editor coordinates.
            if not cfg.formation.distance_matrix:
                cfg.formation.distance_matrix = pairwise_matrix([(n.x, n.y) for n in cfg.formation.nodes])
        cfg.n_agents = len(cfg.formation.nodes)
        # Ensure default agent has non-default id replaced only when cloned.
        agents_by_id: Dict[str, AgentSpec] = {}
        for a in cfg.agents or []:
            agents_by_id[a.id] = a
        out_agents: List[AgentSpec] = []
        for node in cfg.formation.nodes:
            if node.id in agents_by_id:
                agent = agents_by_id[node.id]
            else:
                agent = copy.deepcopy(cfg.default_agent)
                agent.id = node.id
                agent.name = node.name or node.id
                agent.powered = node.powered
            if not agent.name or agent.name == "default":
                agent.name = node.name or node.id
            agent.powered = node.powered and agent.powered
            out_agents.append(agent)
        cfg.agents = out_agents
        # IDs for obstacles and default activation.
        for i, obs in enumerate(cfg.obstacles):
            if not obs.id:
                obs.id = f"o{i + 1:02d}"
            if obs.label is None:
                obs.label = obs.id
        return cfg

    def _init_robots(self) -> None:
        agents = {a.id: a for a in self.config.agents or []}
        for node in self.config.formation.nodes:  # type: ignore[union-attr]
            spec = copy.deepcopy(agents[node.id])
            off = self.formation_offsets[node.id]
            self.robots.append(
                RobotState(
                    id=node.id,
                    name=spec.name or node.name or node.id,
                    spec=spec,
                    x=self.config.start_x_m + off[0],
                    y=self.config.start_y_m + off[1],
                    theta=0.0,
                    camera_pan_deg=spec.camera_pan_deg,
                    camera_tilt_deg=spec.camera_tilt_deg,
                )
            )

    def _init_obstacles(self) -> None:
        if self.config.random_obstacles and self.config.random_obstacle_count > 0:
            start = len(self.config.obstacles)
            for i in range(self.config.random_obstacle_count):
                w = self.rng.uniform(0.25, 0.70)
                h = self.rng.uniform(0.20, 0.55)
                x = self.rng.uniform(1.6, max(1.8, self.config.mission_length_m - 1.2))
                max_y = max(0.05, self.config.corridor_width_m / 2 - h / 2 - 0.12)
                y = self.rng.uniform(-max_y, max_y)
                appear = 0.0 if self.rng.random() < 0.75 else self.rng.uniform(0.8, 10.0)
                disappear = None if self.rng.random() < 0.72 else appear + self.rng.uniform(4.0, 16.0)
                self.config.obstacles.append(ObstacleSpec(id=f"o{start + i + 1:02d}", label=f"rnd-{i + 1}", kind="rectangle", x=x, y=y, width_m=w, height_m=h, appear_time_s=appear, disappear_time_s=disappear))

    # ----- observation and planning ------------------------------------------------
    def current_centroid(self) -> Vec:
        powered_or_all = [r.pos for r in self.robots if r.powered and r.alive] or [r.pos for r in self.robots]
        return centroid(powered_or_all)

    def formation_y_span(self) -> float:
        ys = [p[1] for p in self.formation_offsets.values()]
        return max(ys) - min(ys) if ys else 0.0

    def active_obstacles(self) -> List[ObstacleSpec]:
        return [o for o in self.config.obstacles if obstacle_active(o, self.time_s)]

    def sonar_observations_for_robot(self, robot: RobotState) -> List[Dict[str, object]]:
        """Return obstacles currently visible to the robot ultrasonic head.

        This is the only obstacle source in `perception.mode=local`: the
        algorithm receives objects only after they are inside range and inside
        the forward sonar cone.  The returned dict also gives UI-friendly beam
        endpoints so the operator can see exactly what the agent sees.
        """
        observations: List[Dict[str, object]] = []
        fov = math.radians(max(1.0, min(360.0, float(robot.spec.sonic_fov_deg))))
        for obs in self.active_obstacles():
            approx = obstacle_approx_radius(obs)
            c = obstacle_center(obs, self.time_s)
            center_dist = dist(robot.pos, c)
            if center_dist > robot.spec.sonic_range_m + approx + robot.radius_m:
                continue
            angle = angle_wrap(math.atan2(c[1] - robot.y, c[0] - robot.x) - robot.theta)
            # A finite-width obstacle can be partially visible even when its
            # center is a little outside the cone; approximate that by angular
            # half-size.
            half_size = math.asin(min(0.95, approx / max(center_dist, approx + 1e-6))) if center_dist > 1e-6 else math.pi
            if fov < 2.0 * math.pi - 1e-6 and abs(angle) > fov / 2.0 + half_size:
                continue
            surface = max(point_clearance_to_obstacle(robot.pos, obs, self.time_s) - robot.radius_m, 0.0)
            if surface > robot.spec.sonic_range_m:
                continue
            hit_x = robot.x + surface * math.cos(robot.theta + angle)
            hit_y = robot.y + surface * math.sin(robot.theta + angle)
            observations.append({
                "id": obs.id or "obs",
                "label": obs.label or obs.id or "obs",
                "kind": obs.kind,
                "angle_rad": angle,
                "angle_deg": math.degrees(angle),
                "distance_m": surface,
                "distance_cm": int(round(surface * 100.0)),
                "center_x": c[0],
                "center_y": c[1],
                "hit_x": hit_x,
                "hit_y": hit_y,
            })
        observations.sort(key=lambda item: float(item["distance_m"]))
        return observations

    def visible_obstacles_for_robot(self, robot: RobotState) -> List[ObstacleSpec]:
        ids = {str(item["id"]) for item in self.sonar_observations_for_robot(robot)}
        return [obs for obs in self.active_obstacles() if (obs.id or "obs") in ids]

    def _freeze_obstacle_observation(self, obs: ObstacleSpec) -> ObstacleSpec:
        # The onboard algorithm receives the observed current shape, not the future schedule.
        o = copy.deepcopy(obs)
        c = obstacle_center(obs, self.time_s)
        o.x, o.y = c
        o.vx_mps = 0.0
        o.vy_mps = 0.0
        o.appear_time_s = 0.0
        o.disappear_time_s = None
        o.active_by_default = True
        return o

    def refresh_shared_memory(self) -> None:
        if self.config.perception.mode == "global_debug":
            return
        now = self.time_s
        for robot in self.robots:
            if not robot.powered or not robot.alive:
                continue
            for obs in self.visible_obstacles_for_robot(robot):
                oid = obs.id or "obs"
                if not self.config.perception.shared_memory:
                    # no shared storage, but keep robot-local info available only for this step via an event;
                    # algorithms use planning_obstacles(), so no memory means only current visible union.
                    continue
                frozen = self._freeze_obstacle_observation(obs)
                entry = self.shared_obstacles.get(oid)
                if entry is None:
                    self.shared_obstacles[oid] = SharedObstacle(frozen, first_seen_s=now, last_seen_s=now, seen_by={robot.id})
                else:
                    entry.spec = frozen
                    entry.last_seen_s = now
                    entry.seen_by.add(robot.id)
        # Drop stale observations.
        ttl = self.config.perception.obstacle_memory_s
        if ttl >= 0:
            stale = [oid for oid, e in self.shared_obstacles.items() if now - e.last_seen_s > ttl]
            for oid in stale:
                del self.shared_obstacles[oid]

    def planning_obstacles(self) -> List[ObstacleSpec]:
        if self.config.perception.mode == "global_debug":
            return [copy.deepcopy(o) for o in self.active_obstacles()]
        if self.config.perception.shared_memory:
            return [copy.deepcopy(e.spec) for e in self.shared_obstacles.values()]
        # Union of what is currently visible without persistence.
        out: Dict[str, ObstacleSpec] = {}
        for robot in self.robots:
            for obs in self.visible_obstacles_for_robot(robot):
                out[obs.id or "obs"] = self._freeze_obstacle_observation(obs)
        return list(out.values())

    def planning_fingerprint(self, obstacles: Sequence[ObstacleSpec]) -> Tuple:
        items = []
        for o in obstacles:
            c = obstacle_center(o, self.time_s)
            items.append((o.id or "obs", o.kind, round(c[0], 2), round(c[1], 2), round(o.width_m, 2), round(o.height_m, 2), round(o.radius_m, 2)))
        return tuple(sorted(items))

    def algorithm_params_for(self, key: Optional[str] = None) -> Dict[str, object]:
        """Return editable algorithm parameters for the selected planner.

        The onboard algorithm is still the same for every agent; these parameters
        are global mission constants that the user edits through the algorithm modal.
        """
        k = key or self.planner_key
        params = getattr(self.config, "algorithm_params", {}) or {}
        item = params.get(k, {}) if isinstance(params, dict) else {}
        return item if isinstance(item, dict) else {}


    def begin_planning_cycle(self) -> None:
        """Clear visualization-only route diagnostics for the next planner call."""
        self.planned_routes = []

    def stabilize_cluster_id(self, robot: RobotState, raw_cluster_id: str) -> str:
        """Apply small hysteresis so agents do not flicker between bypass lanes."""
        if not self.config.limits.smooth_reconfiguration:
            return raw_cluster_id
        steps_required = int(self.config.limits.cluster_hysteresis_steps)
        if steps_required <= 0:
            return raw_cluster_id
        mem = self.plan_memory.setdefault(robot.id, {})
        stable = str(mem.get("stable_cluster_id", robot.cluster_id or raw_cluster_id))
        if raw_cluster_id == stable:
            mem["pending_cluster_id"] = raw_cluster_id
            mem["pending_cluster_count"] = 0
            mem["stable_cluster_id"] = stable
            return stable
        pending = str(mem.get("pending_cluster_id", ""))
        count = int(mem.get("pending_cluster_count", 0))
        if pending == raw_cluster_id:
            count += 1
        else:
            pending = raw_cluster_id
            count = 1
        if count >= steps_required:
            stable = raw_cluster_id
            count = 0
        mem["pending_cluster_id"] = pending
        mem["pending_cluster_count"] = count
        mem["stable_cluster_id"] = stable
        return stable

    def smooth_control_point(self, robot: RobotState, raw_point: Vec, key: str = "control_point") -> Vec:
        """Rate-limit and low-pass the temporary point the robot follows.

        This makes split/rejoin transitions visible and physically plausible:
        the local setpoint moves continuously instead of jumping from the
        formation slot to an obstacle-bypass waypoint and back.
        """
        if not self.config.limits.smooth_reconfiguration:
            return raw_point
        mem = self.plan_memory.setdefault(robot.id, {})
        prev = mem.get(key)
        if not isinstance(prev, tuple):
            prev = robot.pos
        # Keep the setpoint near the robot after manual drag or after a route reset.
        if dist(prev, robot.pos) > max(0.75, robot.spec.max_speed_mps * 3.0):
            prev = robot.pos
        rate = max(self.config.limits.setpoint_max_rate_mps, robot.spec.max_speed_mps * 1.25, self.config.limits.max_reconfig_speed_mps)
        max_step = max(0.025, rate * self.config.dt_s)
        delta = sub(raw_point, prev)
        limited = add(prev, clamp_vec(delta, max_step))
        alpha = self.config.limits.setpoint_smoothing_alpha
        smooth = (prev[0] * (1.0 - alpha) + limited[0] * alpha, prev[1] * (1.0 - alpha) + limited[1] * alpha)
        mem[key] = smooth
        return smooth

    def remember_planned_route(
        self,
        robot: RobotState,
        raw_path: Sequence[Vec],
        waypoint: Vec,
        control_point: Vec,
        final_target: Vec,
        cluster_id: str,
        status: str,
        reason: str,
        length_m: float,
    ) -> None:
        max_points = int(self.config.limits.route_preview_points)
        compact = resample_polyline(list(raw_path), max_points=max_points)
        smooth = chaikin_smooth(compact, iterations=2, max_points=max_points)
        self.planned_routes.append(
            {
                "robot_id": robot.id,
                "robot_name": robot.name,
                "cluster_id": cluster_id,
                "status": status,
                "reason": reason,
                "length_m": round(length_m, 4) if math.isfinite(length_m) else None,
                "waypoint": {"x": waypoint[0], "y": waypoint[1]},
                "control_point": {"x": control_point[0], "y": control_point[1]},
                "final_target": {"x": final_target[0], "y": final_target[1]},
                "path": [{"x": x, "y": y} for x, y in compact],
                "smooth_path": [{"x": x, "y": y} for x, y in smooth],
            }
        )

    def final_targets(self) -> Dict[str, Vec]:
        goal_center = (self.config.start_x_m + self.config.mission_length_m, self.config.start_y_m)
        return {rid: add(goal_center, off) for rid, off in self.formation_offsets.items()}

    def front_blockers(self, obstacles: Sequence[ObstacleSpec]) -> List[ObstacleSpec]:
        c = self.current_centroid()
        span = self.formation_y_span()
        lane_half = self.config.corridor_width_m / 2.0
        max_r = max([r.radius_m for r in self.robots] or [0.1])
        y_reach = min(lane_half, span / 2.0 + max_r + 0.45)
        sonic = max([r.spec.sonic_range_m for r in self.robots if r.powered and r.alive] or [4.0])
        lookahead = max(2.7, min(4.4, sonic + 0.35))
        x_min = c[0] - 0.35
        x_max = min(self.config.start_x_m + self.config.mission_length_m + 0.4, c[0] + lookahead)
        out = []
        for obs in obstacles:
            if not obstacle_active(obs, self.time_s):
                continue
            center = obstacle_center(obs, self.time_s)
            approx = obstacle_approx_radius(obs)
            left = center[0] - approx
            right = center[0] + approx
            if right >= x_min and left <= x_max and abs(center[1] - self.config.start_y_m) - approx <= y_reach:
                out.append(obs)
        out.sort(key=lambda o: obstacle_center(o, self.time_s)[0])
        return out

    # ----- simulation step ----------------------------------------------------------
    def step(self, algorithm: Optional[str] = None) -> None:
        if self.done:
            return
        if algorithm and algorithm != self.planner_key:
            self.planner_key = algorithm
            self.config.algorithm = algorithm
            self.planner = make_planner(algorithm)
        commands = self.planner.plan(self)
        if self.config.limits.enable_safety_filter:
            commands = self._safety_filter(commands)
        for robot in self.robots:
            cmd = commands.get(robot.id, PlannerCommand(reason="missing_command"))
            self._apply_command(robot, cmd)
        self.step_n += 1
        self.time_s = self.step_n * self.config.dt_s
        self._measure_sensors()
        self._update_metrics()
        self._update_done()
        self._record_csv_rows()

    def _safety_filter(self, commands: Dict[str, PlannerCommand]) -> Dict[str, PlannerCommand]:
        filtered: Dict[str, PlannerCommand] = {}
        active = self.active_obstacles()
        lane_half = self.config.corridor_width_m / 2.0
        dt = self.config.dt_s
        horizon = self.config.limits.avoidance_horizon_s
        for robot in self.robots:
            cmd = commands.get(robot.id, PlannerCommand())
            desired = clamp_vec((cmd.vx, cmd.vy), robot.spec.max_speed_mps)
            target = cmd.target or robot.pos
            to_target = unit(sub(target, robot.pos), fallback=(1.0, 0.0))
            candidates: List[Vec] = [desired, mul(desired, 0.65), mul(desired, 0.35), (0.0, 0.0)]

            # Add tangent candidates around the nearest obstacle to avoid a hard stop in front of it.
            nearest = None
            nearest_clearance = float("inf")
            for obs in active:
                clearance = point_clearance_to_obstacle(robot.pos, obs, self.time_s) - robot.radius_m - self.config.limits.obstacle_margin_m
                if clearance < nearest_clearance:
                    nearest_clearance = clearance
                    nearest = obs
            if nearest is not None and nearest_clearance < 0.75:
                c = obstacle_center(nearest, self.time_s)
                away = unit(sub(robot.pos, c), fallback=(0.0, 1.0))
                # Choose side that still makes progress to the final target; include both for safety scoring.
                for sign in (1.0, -1.0):
                    tangent = (-away[1] * sign, away[0] * sign)
                    cand = add(mul(tangent, robot.spec.max_speed_mps * 0.75), mul(away, robot.spec.max_speed_mps * 0.22))
                    candidates.append(cand)
                    candidates.append(add(mul(desired, 0.45), cand))

            # Wall escape candidates.
            wall_margin = robot.radius_m + self.config.limits.safety_margin_m
            if robot.y > lane_half - wall_margin - 0.08:
                candidates.append((max(desired[0], 0.03), -robot.spec.max_speed_mps * 0.55))
            if robot.y < -lane_half + wall_margin + 0.08:
                candidates.append((max(desired[0], 0.03), robot.spec.max_speed_mps * 0.55))

            current_obs_clear = min([point_clearance_to_obstacle(robot.pos, obs, self.time_s) - robot.radius_m - self.config.limits.obstacle_margin_m for obs in active] or [10.0])
            current_robot_clear = 10.0
            for other in self.robots:
                if other.id == robot.id:
                    continue
                current_robot_clear = min(current_robot_clear, dist(robot.pos, other.pos) - robot.radius_m - other.radius_m - self.config.limits.robot_robot_margin_m)
            current_min_clear = min(current_obs_clear, current_robot_clear)

            best_v = (0.0, 0.0)
            best_score = -1e18
            for cand0 in candidates:
                cand = clamp_vec(cand0, robot.spec.max_speed_mps)
                pred = (robot.x + cand[0] * horizon, robot.y + cand[1] * horizon)
                step_pred = (robot.x + cand[0] * dt, robot.y + cand[1] * dt)
                if abs(pred[1]) > lane_half - robot.radius_m or abs(step_pred[1]) > lane_half - robot.radius_m:
                    continue
                obs_endpoint_clear = min([point_clearance_to_obstacle(pred, obs, self.time_s) - robot.radius_m - self.config.limits.obstacle_margin_m for obs in active] or [10.0])
                obs_step_clear = min([point_clearance_to_obstacle(step_pred, obs, self.time_s) - robot.radius_m - self.config.limits.obstacle_margin_m for obs in active] or [10.0])
                obs_swept_clear = min([segment_clearance_to_obstacle(robot.pos, step_pred, obs, self.time_s, samples_per_m=24.0) - robot.radius_m - self.config.limits.obstacle_margin_m for obs in active] or [10.0])
                obs_clear = min(obs_endpoint_clear, obs_step_clear, obs_swept_clear)
                robot_clear = 10.0
                for other in self.robots:
                    if other.id == robot.id:
                        continue
                    other_pred = (other.x + other.last_cmd_vx * horizon, other.y + other.last_cmd_vy * horizon)
                    robot_clear = min(robot_clear, dist(pred, other_pred) - robot.radius_m - other.radius_m - self.config.limits.robot_robot_margin_m)
                min_clear = min(obs_clear, robot_clear)
                # Hard barrier: prefer any non-colliding candidate. When all collide, select least-bad.
                progress = cand[0] * to_target[0] + cand[1] * to_target[1]
                lateral_penalty = 0.03 * abs(cand[1])
                # Clearance is a hard safety term when negative, but only a weak
                # preference when already safe.  The previous high positive clearance
                # reward made a full stop beat safe forward progress near rejoin.
                clearance_score = 0.35 * min(0.25, max(0.0, min_clear))
                if min_clear < 0:
                    clearance_score += 25.0 * min_clear
                # Deadlock breaker: if the robot is already in a tight pack, allow a
                # candidate that improves progress without materially worsening clearance.
                # This avoids the group freezing near a bypass/rejoin cross-section.
                if current_min_clear < 0.08 and min_clear >= current_min_clear - 0.035:
                    clearance_score += 3.5 * (min_clear - current_min_clear) + 0.55 * max(0.0, progress)
                score = progress + clearance_score - lateral_penalty
                if score > best_score:
                    best_score = score
                    best_v = cand
            reason = cmd.reason
            completion_first = self.config.objective.completion_policy in ("finish_first", "min_time_all_powered")
            if completion_first and norm(best_v) < 0.018 and norm(desired) > 0.02:
                # Completion-first deadlock breaker.  If all conservative
                # candidates collapse to stop, try a small progress-biased
                # move that does not materially reduce the current clearance.
                escape_speed = max(float(getattr(self.config.limits, "completion_unstuck_speed_mps", 0.055) or 0.055), min(robot.spec.max_speed_mps * 0.22, norm(desired)))
                escape = clamp_vec(add(mul(to_target, escape_speed), (max(0.0, desired[0]) * 0.20, desired[1] * 0.25)), robot.spec.max_speed_mps * 0.45)
                short_h = min(horizon, 0.55)
                pred = (robot.x + escape[0] * short_h, robot.y + escape[1] * short_h)
                step_pred = (robot.x + escape[0] * dt, robot.y + escape[1] * dt)
                obs_clear = min([
                    min(
                        point_clearance_to_obstacle(pred, obs, self.time_s),
                        point_clearance_to_obstacle(step_pred, obs, self.time_s),
                        segment_clearance_to_obstacle(robot.pos, step_pred, obs, self.time_s, samples_per_m=24.0),
                    ) - robot.radius_m - self.config.limits.obstacle_margin_m
                    for obs in active
                ] or [10.0])
                robot_clear = 10.0
                for other in self.robots:
                    if other.id == robot.id:
                        continue
                    other_pred = (other.x + other.last_cmd_vx * min(horizon, 0.55), other.y + other.last_cmd_vy * min(horizon, 0.55))
                    robot_clear = min(robot_clear, dist(pred, other_pred) - robot.radius_m - other.radius_m - self.config.limits.robot_robot_margin_m)
                if abs(pred[1]) <= lane_half - robot.radius_m and min(obs_clear, robot_clear) >= current_min_clear - 0.05:
                    best_v = escape
                    reason = (reason + "; completion_unstuck").strip("; ")
            if best_score < -5.0 and self.config.limits.stop_if_no_path:
                best_v = (0.0, 0.0)
                reason = (reason + "; safety_stop").strip("; ")
            filtered[robot.id] = PlannerCommand(
                vx=best_v[0],
                vy=best_v[1],
                led=cmd.led,
                buz=cmd.buz,
                target=cmd.target,
                cluster_id=cmd.cluster_id,
                role=cmd.role,
                reason=reason,
                debug=cmd.debug,
            )
        return filtered

    def _position_safe_for_robot(self, robot: RobotState, pos: Vec, extra_margin: float = 0.0, check_robots: bool = True) -> bool:
        lane_half = self.config.corridor_width_m / 2.0
        wall_margin = robot.radius_m + self.config.limits.safety_margin_m + extra_margin
        if abs(pos[1]) > lane_half - wall_margin:
            return False
        inflated = robot.radius_m + self.config.limits.obstacle_margin_m + extra_margin
        for obs in self.active_obstacles():
            if point_clearance_to_obstacle(pos, obs, self.time_s) <= inflated:
                return False
        if check_robots:
            # v6.3 only guarded obstacles; agents could still overlap each other during
            # split/rejoin.  This is a physical-body guard, not a new planner: a bad
            # command is clipped instead of allowing one tank to drive through another.
            rr_extra = max(0.0, self.config.limits.robot_robot_margin_m + extra_margin)
            for other in self.robots:
                if other.id == robot.id or (not other.alive):
                    continue
                if dist(pos, other.pos) <= robot.radius_m + other.radius_m + rr_extra:
                    return False
        return True

    def _robot_clearance_at(self, robot: RobotState, pos: Vec) -> float:
        vals = [dist(pos, other.pos) - robot.radius_m - other.radius_m - self.config.limits.robot_robot_margin_m
                for other in self.robots if other.id != robot.id and other.alive]
        return min(vals or [10.0])

    def _project_safe_motion(self, robot: RobotState, old_pos: Vec, desired_pos: Vec) -> Tuple[Vec, bool]:
        """Hard physical guard: do not let the simulated body cross obstacles.

        The planner and safety filter should already avoid collisions.  This
        guard is a final integration-level constraint, so even a bad algorithm
        cannot visually drive through a rectangle/circle and produce unusable
        logs.
        """
        if self._position_safe_for_robot(robot, desired_pos):
            # Also check the swept segment; otherwise a large dt could jump across
            # a thin obstacle and only check the endpoint.
            clearance = min([segment_clearance_to_obstacle(old_pos, desired_pos, obs, self.time_s, samples_per_m=24.0) - robot.radius_m - self.config.limits.obstacle_margin_m for obs in self.active_obstacles()] or [10.0])
            if clearance >= 0.0:
                return desired_pos, False
        relaxed = -min(0.02, self.config.limits.obstacle_margin_m * 0.5)
        old_safe_relaxed = self._position_safe_for_robot(robot, old_pos, extra_margin=relaxed)
        old_robot_clearance = self._robot_clearance_at(robot, old_pos)
        lo, hi = 0.0, 1.0
        best = old_pos
        best_clearance = old_robot_clearance
        for _ in range(18):
            mid = (lo + hi) * 0.5
            cand = (old_pos[0] + (desired_pos[0] - old_pos[0]) * mid, old_pos[1] + (desired_pos[1] - old_pos[1]) * mid)
            clearance = min([segment_clearance_to_obstacle(old_pos, cand, obs, self.time_s, samples_per_m=24.0) - robot.radius_m - self.config.limits.obstacle_margin_m for obs in self.active_obstacles()] or [10.0])
            pos_safe = self._position_safe_for_robot(robot, cand) and clearance >= 0.0
            # If the robot is already in an overlap from an older run/dynamic spawn,
            # allow only movements that improve robot-robot clearance while staying
            # obstacle-safe; otherwise it could never escape the overlap.
            if (not old_safe_relaxed) and clearance >= 0.0:
                rc = self._robot_clearance_at(robot, cand)
                pos_safe = rc > best_clearance + 1e-5
            if pos_safe:
                best = cand
                best_clearance = max(best_clearance, self._robot_clearance_at(robot, cand))
                lo = mid
            else:
                hi = mid
        return best, True

    def _apply_command(self, robot: RobotState, cmd: PlannerCommand) -> None:
        dt = self.config.dt_s
        if not robot.powered or not robot.alive:
            robot.pwr_left = robot.pwr_right = 0
            robot.last_cmd_v = robot.last_cmd_omega = robot.last_cmd_vx = robot.last_cmd_vy = 0.0
            return
        physics = getattr(self.config, "scene_physics", None)
        speed_scale = float(getattr(physics, "speed_scale", 1.0) or 1.0)
        accel_scale = float(getattr(physics, "accel_scale", 1.0) or 1.0)
        friction = float(getattr(physics, "friction_coefficient", 0.85) or 0.85)
        friction_accel_scale = clamp(friction / 0.85, 0.25, 1.75)
        effective_max_speed = max(0.02, robot.spec.max_speed_mps * speed_scale)
        effective_reverse = max(0.0, robot.spec.max_reverse_mps * speed_scale)
        desired_vec = clamp_vec((cmd.vx, cmd.vy), effective_max_speed)
        desired_speed = norm(desired_vec)
        desired_heading = robot.theta if desired_speed < 1e-9 else math.atan2(desired_vec[1], desired_vec[0])
        heading_error = angle_wrap(desired_heading - robot.theta)
        desired_v = desired_speed * max(0.0, math.cos(heading_error))
        desired_omega = clamp(2.8 * heading_error, -robot.spec.max_omega_radps, robot.spec.max_omega_radps)
        if self.config.limits.allow_reverse and abs(heading_error) > math.pi * 0.72:
            desired_v = -min(effective_reverse, desired_speed)
            desired_omega = clamp(2.8 * angle_wrap(desired_heading - (robot.theta + math.pi)), -robot.spec.max_omega_radps, robot.spec.max_omega_radps)
        max_dv = robot.spec.max_accel_mps2 * accel_scale * friction_accel_scale * dt
        max_do = robot.spec.max_omega_accel_radps2 * accel_scale * friction_accel_scale * dt
        old_v = robot.v
        old_omega = robot.omega
        v = clamp(desired_v, old_v - max_dv, old_v + max_dv)
        omega = clamp(desired_omega, old_omega - max_do, old_omega + max_do)
        v = clamp(v, -effective_reverse, effective_max_speed)
        omega = clamp(omega, -robot.spec.max_omega_radps, robot.spec.max_omega_radps)
        old_pos = robot.pos
        applied_reason = cmd.reason
        new_theta = angle_wrap(robot.theta + omega * dt)
        desired_pos = (robot.x + v * math.cos(new_theta) * dt, robot.y + v * math.sin(new_theta) * dt)
        guarded_pos, motion_guarded = self._project_safe_motion(robot, old_pos, desired_pos)
        robot.theta = new_theta
        if motion_guarded:
            # If the guard clipped the step, recompute the achieved linear speed
            # from the actual displacement and keep the reason visible in logs.
            achieved = dist(old_pos, guarded_pos) / max(dt, 1e-9)
            if achieved < abs(v) * 0.15:
                v = 0.0
                omega = 0.0
            else:
                v = math.copysign(achieved, v if abs(v) > 1e-9 else 1.0)
            applied_reason = (cmd.reason + "; physical_collision_guard").strip('; ')
        robot.x, robot.y = guarded_pos
        lane_half = self.config.corridor_width_m / 2.0
        if abs(robot.y) > lane_half - robot.radius_m:
            self.lane_violations_group += 1
            robot.lane_violations += 1
            robot.y = clamp(robot.y, -lane_half + robot.radius_m, lane_half - robot.radius_m)
        robot.v = v
        robot.omega = omega
        left = v - omega * robot.spec.wheel_base_m / 2.0
        right = v + omega * robot.spec.wheel_base_m / 2.0
        max_speed = max(effective_max_speed, 1e-6)
        robot.pwr_left = int(round(robot.spec.max_power * clamp(left / max_speed, -1.0, 1.0)))
        robot.pwr_right = int(round(robot.spec.max_power * clamp(right / max_speed, -1.0, 1.0)))
        step_dist = dist(old_pos, robot.pos)
        robot.traveled_m += step_dist
        rolling = float(getattr(physics, "rolling_resistance", 0.02) or 0.0)
        energy_scale = float(getattr(physics, "energy_scale", 1.0) or 1.0)
        energy_inc = energy_scale * robot.spec.energy_coeff * ((abs(robot.pwr_left) + abs(robot.pwr_right)) / (2.0 * robot.spec.max_power) * dt + 0.08 * abs(v - old_v) + 0.010 * abs(omega - old_omega) + rolling * step_dist)
        robot.energy += energy_inc
        self.total_energy_proxy += energy_inc
        # Store requested vector too, not just physically achievable chassis vector.
        robot.last_cmd_vx = cmd.vx
        robot.last_cmd_vy = cmd.vy
        robot.last_cmd_v = v
        robot.last_cmd_omega = omega
        if cmd.target is not None:
            robot.target_x, robot.target_y = cmd.target
        if cmd.cluster_id:
            if robot.cluster_id != cmd.cluster_id:
                robot.cluster_switches += 1
            robot.cluster_id = cmd.cluster_id
        if cmd.role:
            robot.role = cmd.role
        if cmd.led is not None:
            robot.led = cmd.led
        if cmd.buz is not None:
            robot.buz = bool(cmd.buz)
        robot.last_reason = applied_reason

    # ----- sensors and metrics ------------------------------------------------------
    def _measure_sensors(self) -> None:
        for robot in self.robots:
            observations = self.sonar_observations_for_robot(robot)
            robot.sonic_visible_count = len(observations)
            if not observations:
                robot.sonic_angle_deg = 0
                robot.sonic_distance_cm = -1
                robot.sonic_obstacle_id = ""
                robot.sonic_hit_x = None
                robot.sonic_hit_y = None
            else:
                best = observations[0]
                robot.sonic_angle_deg = int(round(float(best["angle_deg"])))
                robot.sonic_distance_cm = int(best["distance_cm"])
                robot.sonic_obstacle_id = str(best["id"])
                robot.sonic_hit_x = float(best["hit_x"])
                robot.sonic_hit_y = float(best["hit_y"])
            robot.tracking_bits = self._tracking_bits(robot)

    def _tracking_bits(self, robot: RobotState) -> str:
        offsets = [-0.12, -0.04, 0.04, 0.12]
        bits = []
        for off in offsets:
            sensor_y = robot.y + off * math.cos(robot.theta)
            bits.append("1" if abs(sensor_y) < 0.035 else "0")
        return "".join(bits)

    def _update_metrics(self) -> None:
        c = self.current_centroid()
        if self._last_centroid is not None:
            self.centroid_path_length_m += dist(self._last_centroid, c)
        self._last_centroid = c
        self.history.append(c)
        # By default the experiment trail is part of the evidence and must not
        # be erased.  A finite UI cap can still be enabled by setting
        # limits.keep_full_trails=false.
        keep_full = bool(getattr(self.config.limits, "keep_full_trails", True))
        if not keep_full and len(self.history) > 2000:
            self.history = self.history[-2000:]
        trail_limit = int(self.config.limits.agent_trail_points)
        for r in self.robots:
            trail = self.agent_trails.setdefault(r.id, [])
            if not trail or dist(trail[-1], r.pos) > 0.01:
                trail.append(r.pos)
                if (not keep_full) and trail_limit > 0 and len(trail) > trail_limit:
                    self.agent_trails[r.id] = trail[-trail_limit:]
        active = self.active_obstacles()
        collision = False
        near = False
        collided = set()
        nears = set()
        for i, a in enumerate(self.robots):
            for b in self.robots[i + 1 :]:
                clearance = dist(a.pos, b.pos) - a.radius_m - b.radius_m
                a.min_robot_clearance_m = min(a.min_robot_clearance_m, clearance)
                b.min_robot_clearance_m = min(b.min_robot_clearance_m, clearance)
                if clearance < 0:
                    collision = True
                    collided.update([a.id, b.id])
                elif clearance < self.config.limits.safety_margin_m:
                    near = True
                    nears.update([a.id, b.id])
            for obs in active:
                clearance = point_clearance_to_obstacle(a.pos, obs, self.time_s) - a.radius_m
                a.min_obstacle_clearance_m = min(a.min_obstacle_clearance_m, clearance)
                if clearance < 0:
                    collision = True
                    collided.add(a.id)
                elif clearance < self.config.limits.safety_margin_m:
                    near = True
                    nears.add(a.id)
        if collision:
            self.collision_steps += 1
            for rid in collided:
                self.robot_by_id(rid).collision_events += 1
        if near:
            self.near_miss_steps += 1
            for rid in nears:
                self.robot_by_id(rid).near_miss_events += 1

    def _update_done(self) -> None:
        if self.step_n >= self.config.max_steps:
            self.done = True
            self.finish_reason = "timeout"
            return
        targets = self.final_targets()
        powered = [r for r in self.robots if r.powered and r.alive]
        if self.config.objective.require_all_powered_agents and powered:
            agent_ok = all(dist(r.pos, targets.get(r.id, r.pos)) <= self.config.finish_agent_tolerance_m for r in powered)
        else:
            agent_ok = True
        c = self.current_centroid()
        centroid_ok = dist(c, (self.config.mission_length_m, 0.0)) <= self.config.finish_centroid_tolerance_m
        if self.config.formation:
            mat = self.matrices()
            formation_ok = (float(mat.get("rms_m", 0.0)) <= self.config.formation.tolerance_m) and (len(mat.get("bad_pairs", [])) == 0)
        else:
            formation_ok = True
        collision_ok = self.collision_steps == 0 if self.config.objective.success_requires_no_collision else True
        if agent_ok and centroid_ok and (formation_ok or not self.config.objective.success_requires_formation) and collision_ok:
            self.done = True
            self.finish_reason = "success"

    def robot_by_id(self, rid: str) -> RobotState:
        for r in self.robots:
            if r.id == rid:
                return r
        raise KeyError(rid)

    def formation_matrix_error(self) -> float:
        ids = self.formation_ids
        pos = {r.id: r.pos for r in self.robots}
        matrices = formation_matrices(ids, pos, self.desired_matrix, self.config.formation.edge_tolerance_m)  # type: ignore[union-attr]
        return float(matrices["rms_m"])

    # ----- public state and logs ----------------------------------------------------
    def matrices(self) -> Dict[str, object]:
        ids = self.formation_ids
        positions = {r.id: r.pos for r in self.robots}
        m = formation_matrices(ids, positions, self.desired_matrix, self.config.formation.edge_tolerance_m)  # type: ignore[union-attr]
        finals = self.final_targets()
        final_current = pairwise_matrix([finals[rid] for rid in ids])
        m["final_target"] = final_current
        return m

    def edge_diagnostics(self) -> List[Dict[str, object]]:
        m = self.matrices()
        ids = list(m["ids"])
        id_index = {rid: i for i, rid in enumerate(ids)}
        current = m["current"]
        desired = m["initial"]
        edges = []
        for e in self.formation_edges:
            a, b = e["source"], e["target"]
            i, j = id_index[a], id_index[b]
            delta = current[i][j] - desired[i][j]
            severity = "ok"
            tol = self.config.formation.edge_tolerance_m  # type: ignore[union-attr]
            if abs(delta) > tol * 1.7:
                severity = "red"
            elif abs(delta) > tol:
                severity = "yellow"
            out = dict(e)
            out.update({"current_m": current[i][j], "delta_m": delta, "severity": severity})
            edges.append(out)
        return edges

    def group_metrics(self) -> Dict[str, object]:
        c = self.current_centroid()
        goal = (self.config.start_x_m + self.config.mission_length_m, self.config.start_y_m)
        matrices = self.matrices()
        clearances = []
        for r in self.robots:
            clearances.append(r.min_obstacle_clearance_m if not math.isinf(r.min_obstacle_clearance_m) else 999.0)
            clearances.append(r.min_robot_clearance_m if not math.isinf(r.min_robot_clearance_m) else 999.0)
        active_clusters = len({r.cluster_id for r in self.robots if r.powered and r.alive})
        powered = [r for r in self.robots if r.powered and r.alive]
        mean_instant_speed = sum(abs(r.v) for r in powered) / max(1, len(powered))
        average_swarm_speed = self.config.mission_length_m / self.time_s if self.done and self.time_s > 1e-9 else (max(0.0, c[0] - self.config.start_x_m) / self.time_s if self.time_s > 1e-9 else 0.0)
        known_count = len(self.planning_obstacles()) if self.config.perception.mode == "local" else len(self.active_obstacles())
        visible_count = sum(len(self.visible_obstacles_for_robot(r)) for r in powered)
        return {
            "algorithm": self.planner_key,
            "perception_mode": self.config.perception.mode,
            "known_obstacle_count": known_count,
            "visible_obstacle_count_sum": visible_count,
            "objective_function": self.config.objective.objective_function,
            "centroid_x_m": fmt_float(c[0]),
            "centroid_y_m": fmt_float(c[1]),
            "centroid_to_goal_m": fmt_float(dist(c, goal)),
            "centroid_path_length_m": fmt_float(self.centroid_path_length_m),
            "effective_centroid_length_m": fmt_float(max(0.0, c[0] - self.config.start_x_m)),
            "reference_length_m": self.config.mission_length_m,
            "excess_over_reference_m": fmt_float(max(0.0, self.centroid_path_length_m - self.config.mission_length_m)),
            "formation_rms_m": fmt_float(float(matrices["rms_m"])),
            "formation_max_abs_delta_m": fmt_float(float(matrices["max_abs_delta_m"])),
            "formation_bad_pair_count": len(matrices["bad_pairs"]),
            "min_clearance_m": fmt_float(min(clearances) if clearances else 0.0),
            "collision_steps": self.collision_steps,
            "near_miss_steps": self.near_miss_steps,
            "lane_violations_group": self.lane_violations_group,
            "total_energy_proxy": fmt_float(self.total_energy_proxy),
            "sum_agent_path_length_m": fmt_float(sum(r.traveled_m for r in self.robots)),
            "mean_agent_path_length_m": fmt_float(sum(r.traveled_m for r in self.robots) / max(1, len(self.robots))),
            "max_agent_path_length_m": fmt_float(max([r.traveled_m for r in self.robots] or [0.0])),
            "mean_instant_speed_mps": fmt_float(mean_instant_speed),
            "average_swarm_speed_mps": fmt_float(average_swarm_speed),
            "active_cluster_count": active_clusters,
            "step": self.step_n,
            "time_s": fmt_float(self.time_s),
            "done": self.done,
            "finish_reason": self.finish_reason,
            "decision_basis": self.config.objective.completion_policy,
        }

    def agent_metrics(self) -> List[Dict[str, object]]:
        targets = self.final_targets()
        out = []
        for r in self.robots:
            out.append(
                {
                    "id": r.id,
                    "name": r.name,
                    "powered": r.powered,
                    "path_length_m": fmt_float(r.traveled_m),
                    "energy_proxy": fmt_float(r.energy),
                    "goal_distance_m": fmt_float(dist(r.pos, targets.get(r.id, r.pos))),
                    "current_obstacle_clearance_m": fmt_float(point_clearance_to_any_obstacle(r.pos, self.active_obstacles(), self.time_s) - r.radius_m if self.active_obstacles() else 999.0),
                    "min_obstacle_clearance_m": fmt_float(r.min_obstacle_clearance_m if not math.isinf(r.min_obstacle_clearance_m) else 999.0),
                    "min_robot_clearance_m": fmt_float(r.min_robot_clearance_m if not math.isinf(r.min_robot_clearance_m) else 999.0),
                    "collision_events": r.collision_events,
                    "near_miss_events": r.near_miss_events,
                    "lane_violations": r.lane_violations,
                    "cluster_switches": r.cluster_switches,
                    "cluster_id": r.cluster_id,
                    "role": r.role,
                    "last_reason": r.last_reason,
                }
            )
        return out

    def to_public_state(self) -> Dict[str, object]:
        matrices = self.matrices()
        targets = self.final_targets()
        return {
            "id": self.id,
            "config": self.export_config(),
            "step": self.step_n,
            "time_s": self.time_s,
            "done": self.done,
            "finish_reason": self.finish_reason,
            "algorithm": self.planner_key,
            "initial_summary": self.initial_summary(),
            "robots": [
                {
                    "id": r.id,
                    "name": r.name,
                    "x": r.x,
                    "y": r.y,
                    "theta": r.theta,
                    "v": r.v,
                    "omega": r.omega,
                    "radius_m": r.radius_m,
                    "length_m": r.spec.length_m,
                    "width_m": r.spec.width_m,
                    "powered": r.powered,
                    "pwr_left": r.pwr_left,
                    "pwr_right": r.pwr_right,
                    "sonic_angle_deg": r.sonic_angle_deg,
                    "sonic_distance_cm": r.sonic_distance_cm,
                    "sonic_obstacle_id": r.sonic_obstacle_id,
                    "sonic_hit": {"x": r.sonic_hit_x, "y": r.sonic_hit_y} if r.sonic_hit_x is not None and r.sonic_hit_y is not None else None,
                    "sonic_visible_count": r.sonic_visible_count,
                    "sonic_observations": self.sonar_observations_for_robot(r)[:8],
                    "tracking_bits": r.tracking_bits,
                    "led": r.led,
                    "buz": r.buz,
                    "target": {"x": r.target_x if r.target_x is not None else targets.get(r.id, r.pos)[0], "y": r.target_y if r.target_y is not None else targets.get(r.id, r.pos)[1]},
                    "final_target": {"x": targets.get(r.id, r.pos)[0], "y": targets.get(r.id, r.pos)[1]},
                    "cluster_id": r.cluster_id,
                    "role": r.role,
                    "last_reason": r.last_reason,
                    "color": r.spec.color,
                    "spec": model_dump_compat(r.spec),
                    "tank_command": self.tank_command_for_robot(r),
                    "visible_obstacle_count": len(self.visible_obstacles_for_robot(r)),
                }
                for r in self.robots
            ],
            "obstacles": [obstacle_to_public(o, i, self.time_s) for i, o in enumerate(self.config.obstacles)],
            "active_obstacles": [obstacle_to_public(o, i, self.time_s) for i, o in enumerate(self.config.obstacles) if obstacle_active(o, self.time_s)],
            "shared_obstacles": [
                {"id": oid, "last_seen_s": e.last_seen_s, "seen_by": sorted(e.seen_by), "spec": model_dump_compat(e.spec)} for oid, e in self.shared_obstacles.items()
            ],
            "formation_edges": self.edge_diagnostics(),
            "matrices": matrices,
            "metrics": self.group_metrics(),
            "agent_metrics": self.agent_metrics(),
            "history": [{"x": x, "y": y} for x, y in (self.history if getattr(self.config.limits, "keep_full_trails", True) else self.history[-800:])],
            "agent_trails": {rid: [{"x": x, "y": y} for x, y in pts] for rid, pts in self.agent_trails.items()},
            "planned_routes": self.planned_routes,
            "cluster_plan": self.last_cluster_plan,
            "logs": {
                "csv_rows": len(self.csv_rows),
                "g_lines": len(self.status_lines_g()),
                "csv_url": f"/api/sim/{self.id}/log.csv",
                "g_url": f"/api/sim/{self.id}/log/g.txt",
                "tank_commands_url": f"/api/sim/{self.id}/log/tank_commands.csv",
                "bundle_url": f"/api/sim/{self.id}/logs.zip",
            },
            "events": self.events[-100:],
        }

    def _record_csv_rows(self) -> None:
        c = self.current_centroid()
        targets = self.final_targets()
        matrices = self.matrices()
        metrics = self.group_metrics()
        for r in self.robots:
            target = targets.get(r.id, r.pos)
            row = {
                "mission_id": self.id,
                "mission_name": self.config.name,
                "step": self.step_n,
                "time_s": round(self.time_s, 4),
                "ts_ms": 1784462000 + int(round(self.time_s * 1000)),
                "algorithm": self.planner_key,
                "completion_policy": self.config.objective.completion_policy,
                "robot_id": r.id,
                "robot_name": r.name,
                "powered": r.powered,
                "cluster_id": r.cluster_id,
                "role": r.role,
                "perception_mode": self.config.perception.mode,
                "visible_obstacle_count": len(self.visible_obstacles_for_robot(r)),
                "known_obstacle_count": len(self.planning_obstacles()),
                "primary_x_m": round(r.x, 5),
                "primary_y_m": round(r.y, 5),
                "primary_theta_rad": round(r.theta, 5),
                "primary_v_mps": round(r.v, 5),
                "primary_omega_radps": round(r.omega, 5),
                "primary_pwr_left": r.pwr_left,
                "primary_pwr_right": r.pwr_right,
                "primary_sonic_angle_deg": r.sonic_angle_deg,
                "primary_sonic_distance_cm": r.sonic_distance_cm,
                "primary_sonic_obstacle_id": r.sonic_obstacle_id,
                "primary_sonic_hit_x_m": round(r.sonic_hit_x, 5) if r.sonic_hit_x is not None else "",
                "primary_sonic_hit_y_m": round(r.sonic_hit_y, 5) if r.sonic_hit_y is not None else "",
                "primary_sonic_visible_count": r.sonic_visible_count,
                "primary_camera_pan_deg": r.camera_pan_deg,
                "primary_camera_tilt_deg": r.camera_tilt_deg,
                "primary_tracking_bits": r.tracking_bits,
                "primary_led": r.led,
                "primary_buz": r.buz,
                "control_cmd_vx_mps": round(r.last_cmd_vx, 5),
                "control_cmd_vy_mps": round(r.last_cmd_vy, 5),
                "control_cmd_v_mps": round(r.last_cmd_v, 5),
                "control_cmd_omega_radps": round(r.last_cmd_omega, 5),
                "control_target_x_m": round(r.target_x if r.target_x is not None else target[0], 5),
                "control_target_y_m": round(r.target_y if r.target_y is not None else target[1], 5),
                "control_reason": r.last_reason,
                "tank_key_command": self.tank_command_for_robot(r)["key"],
                "tank_protocol_command": self.tank_command_for_robot(r)["protocol"],
                "tank_pwm_left": self.tank_command_for_robot(r)["pwm_left"],
                "tank_pwm_right": self.tank_command_for_robot(r)["pwm_right"],
                "tank_command_note": self.tank_command_for_robot(r)["note"],
                "agent_path_length_m": round(r.traveled_m, 5),
                "agent_energy_proxy": round(r.energy, 5),
                "agent_collision_events": r.collision_events,
                "agent_near_miss_events": r.near_miss_events,
                "agent_lane_violations": r.lane_violations,
                "agent_min_obstacle_clearance_m": round(r.min_obstacle_clearance_m if not math.isinf(r.min_obstacle_clearance_m) else 999.0, 5),
                "agent_min_robot_clearance_m": round(r.min_robot_clearance_m if not math.isinf(r.min_robot_clearance_m) else 999.0, 5),
                "agent_goal_distance_m": round(dist(r.pos, target), 5),
                "group_centroid_x_m": round(c[0], 5),
                "group_centroid_y_m": round(c[1], 5),
                "group_centroid_to_goal_m": metrics["centroid_to_goal_m"],
                "group_centroid_path_m": metrics["centroid_path_length_m"],
                "group_effective_centroid_length_m": metrics["effective_centroid_length_m"],
                "group_formation_rms_m": round(float(matrices["rms_m"]), 5),
                "group_formation_bad_pairs": len(matrices["bad_pairs"]),
                "group_min_clearance_m": metrics["min_clearance_m"],
                "group_collision_steps": self.collision_steps,
                "group_lane_violations": self.lane_violations_group,
                "group_total_energy_proxy": round(self.total_energy_proxy, 5),
                "group_sum_agent_path_m": round(sum(a.traveled_m for a in self.robots), 5),
                "group_mean_instant_speed_mps": metrics.get("mean_instant_speed_mps", 0.0),
                "group_average_speed_mps": metrics.get("average_swarm_speed_mps", 0.0),
                "group_active_cluster_count": metrics["active_cluster_count"],
                "done": self.done,
                "finish_reason": self.finish_reason,
            }
            self.csv_rows.append(row)

    def tank_command_for_robot(self, robot: RobotState) -> Dict[str, object]:
        """Approximate mapping from simulated track power to tank.c commands.

        tank.c has two command layers: console keys (w/a/s/d/e) and the
        Yahboom-style numeric protocol.  The simulator stores direct left/right
        powers; this adapter writes a reproducible approximation for physical
        replay logs.
        """
        max_power = max(1, int(robot.spec.max_power))
        pwm_period = int(getattr(robot.spec, "tank_pwm_period", 20000) or 20000)
        pwm_min = int(getattr(robot.spec, "tank_pwm_min", 4000) or 4000)
        pwm_step = int(getattr(robot.spec, "tank_pwm_step", 1000) or 1000)
        def scale(v: int) -> int:
            if abs(v) < 1:
                return 0
            raw = int(round(abs(v) / max_power * pwm_period))
            raw = max(pwm_min, min(pwm_period, raw))
            # Quantize to tank.c duty-cycle levels.
            q = pwm_min + round((raw - pwm_min) / pwm_step) * pwm_step
            q = max(pwm_min, min(pwm_period, q))
            return q if v > 0 else -q
        left = scale(robot.pwr_left)
        right = scale(robot.pwr_right)
        eps = max(100, pwm_step // 2)
        key = "e"
        move = rotate = 0
        if abs(left) < eps and abs(right) < eps:
            key, move, rotate = "e", 0, 0
        elif left > 0 and right > 0 and abs(left - right) <= pwm_step:
            key, move, rotate = "w", 1, 0
        elif left < 0 and right < 0 and abs(left - right) <= pwm_step:
            key, move, rotate = "s", 2, 0
        elif left < 0 and right > 0:
            key, move, rotate = "a", 0, 1
        elif left > 0 and right < 0:
            key, move, rotate = "d", 0, 2
        elif right > left:
            key, move, rotate = "a", 3, 0
        else:
            key, move, rotate = "d", 4, 0
        protocol = f"${move},{rotate},0,0,0,0,0,0,0#"
        note = "Set tank.c app_speed/duty level to the logged pwm before sending the move key/protocol command; e=stop."
        return {"key": key, "protocol": protocol, "pwm_left": left, "pwm_right": right, "note": note}

    def tank_command_rows(self) -> List[Dict[str, object]]:
        rows = []
        for row in self.csv_rows:
            rows.append({
                "time_s": row.get("time_s"),
                "robot_id": row.get("robot_id"),
                "robot_name": row.get("robot_name"),
                "key": row.get("tank_key_command"),
                "protocol": row.get("tank_protocol_command"),
                "pwm_left": row.get("tank_pwm_left"),
                "pwm_right": row.get("tank_pwm_right"),
                "sonic_angle_deg": row.get("primary_sonic_angle_deg"),
                "sonic_distance_cm": row.get("primary_sonic_distance_cm"),
                "reason": row.get("control_reason"),
            })
        return rows

    def tank_command_csv_text(self) -> str:
        buf = io.StringIO()
        fields = ["time_s", "robot_id", "robot_name", "key", "protocol", "pwm_left", "pwm_right", "sonic_angle_deg", "sonic_distance_cm", "reason"]
        writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in self.tank_command_rows():
            writer.writerow(row)
        return buf.getvalue()

    def tank_command_text(self) -> str:
        lines = ["# time_s robot key protocol pwm_left pwm_right reason"]
        for r in self.tank_command_rows():
            lines.append(f"{r['time_s']} {r['robot_id']} {r['key']} {r['protocol']} {r['pwm_left']} {r['pwm_right']} # {r.get('reason','')}")
        return "\n".join(lines) + "\n"

    def initial_summary(self) -> Dict[str, object]:
        f = self.config.formation
        return {
            "mission_name": self.config.name,
            "mission_length_m": self.config.mission_length_m,
            "corridor_width_m": self.config.corridor_width_m,
            "start": {"x": self.config.start_x_m, "y": self.config.start_y_m},
            "algorithm": self.planner_key,
            "objective_function": self.config.objective.objective_function,
            "completion_policy": self.config.objective.completion_policy,
            "perception_mode": self.config.perception.mode,
            "shared_memory": self.config.perception.shared_memory,
            "surface": model_dump_compat(self.config.scene_physics),
            "formation": {"name": f.name if f else None, "n": len(f.nodes) if f else 0, "tolerance_m": f.tolerance_m if f else None},
            "agents": [{"id": r.id, "name": r.name, "length_m": r.spec.length_m, "width_m": r.spec.width_m, "color": r.spec.color, "max_speed_mps": r.spec.max_speed_mps, "sonic_range_m": r.spec.sonic_range_m} for r in self.robots],
            "obstacle_count": len(self.config.obstacles),
            "obstacles": [{"id": o.id or f"o{i+1:02d}", "kind": o.kind, "x": o.x, "y": o.y, "width_m": o.width_m, "height_m": o.height_m, "radius_m": o.radius_m, "appear_time_s": o.appear_time_s, "disappear_time_s": o.disappear_time_s} for i, o in enumerate(self.config.obstacles)],
        }

    def agent_online_status(self, agent_id: str) -> Dict[str, object]:
        r = self.robot_by_id(agent_id)
        targets = self.final_targets()
        visible = self.visible_obstacles_for_robot(r)
        return {
            "sim_id": self.id,
            "step": self.step_n,
            "time_s": self.time_s,
            "algorithm": self.planner_key,
            "perception_mode": self.config.perception.mode,
            "agent": {
                "id": r.id,
                "name": r.name,
                "powered": r.powered,
                "x": r.x,
                "y": r.y,
                "theta": r.theta,
                "v": r.v,
                "omega": r.omega,
                "pwr_left": r.pwr_left,
                "pwr_right": r.pwr_right,
                "sonic_angle_deg": r.sonic_angle_deg,
                "sonic_distance_cm": r.sonic_distance_cm,
                "sonic_obstacle_id": r.sonic_obstacle_id,
                "sonic_hit": {"x": r.sonic_hit_x, "y": r.sonic_hit_y} if r.sonic_hit_x is not None and r.sonic_hit_y is not None else None,
                "sonic_visible_count": r.sonic_visible_count,
                "sonic_observations": self.sonar_observations_for_robot(r)[:8],
                "tracking_bits": r.tracking_bits,
                "led": r.led,
                "buz": r.buz,
                "cluster_id": r.cluster_id,
                "role": r.role,
                "last_reason": r.last_reason,
                "target": {"x": r.target_x, "y": r.target_y},
                "final_target": {"x": targets.get(r.id, r.pos)[0], "y": targets.get(r.id, r.pos)[1]},
                "tank_command": self.tank_command_for_robot(r),
            },
            "visible_obstacles": [obstacle_to_public(o, i, self.time_s) for i, o in enumerate(visible)],
            "known_obstacle_count": len(self.planning_obstacles()),
            "metrics": next((m for m in self.agent_metrics() if m["id"] == r.id), {}),
        }

    def csv_text(self) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in self.csv_rows:
            writer.writerow(row)
        return buf.getvalue()

    def status_lines_g(self) -> List[str]:
        # Full g-like mission log, not just the last frame.  This is the
        # reproducible control/status log used for physical comparison.
        if self.csv_rows:
            lines = []
            for row in self.csv_rows:
                sonic_distance = int(row.get("primary_sonic_distance_cm", -1) or -1)
                sonic_angle = int(row.get("primary_sonic_angle_deg", 0) or 0)
                sonic = f"{sonic_angle:+04d}, {sonic_distance:+04d}cm" if sonic_distance >= 0 else f"{sonic_angle:+04d}, -001cm"
                lines.append(
                    f"{row.get('robot_id')}: ts=[{row.get('ts_ms')}], "
                    f"tracks=[{int(row.get('primary_pwr_left', 0)):+06d}, {int(row.get('primary_pwr_right', 0)):+06d}], "
                    f"sonic=[{sonic}], camera=[{int(row.get('primary_camera_pan_deg', 0)):+04d}, {int(row.get('primary_camera_tilt_deg', 0)):+04d}], "
                    f"led=[{row.get('primary_led', '---')}], tracking=[{row.get('primary_tracking_bits', '0000')}], "
                    f"tank_cmd=[{row.get('tank_key_command', 'e')}], protocol=[{row.get('tank_protocol_command', '$0,0,0,0,0,0,0,0,0#')}], reason=[{row.get('control_reason', '')}]"
                )
            return lines
        lines = []
        for r in self.robots:
            sonic = f"{r.sonic_angle_deg:+04d}, {r.sonic_distance_cm:+04d}cm" if r.sonic_distance_cm >= 0 else f"{r.sonic_angle_deg:+04d}, -001cm"
            lines.append(
                f"{r.id}: ts=[{1784462000 + int(round(self.time_s * 1000))}], "
                f"tracks=[{r.pwr_left:+06d}, {r.pwr_right:+06d}], "
                f"sonic=[{sonic}], camera=[{r.camera_pan_deg:+04d}, {r.camera_tilt_deg:+04d}], "
                f"led=[{r.led}], tracking=[{r.tracking_bits}]"
            )
        return lines

    def export_config(self) -> Dict[str, object]:
        # Export current config plus edited agent properties.
        cfg = copy.deepcopy(self.config)
        agents = []
        for r in self.robots:
            spec = copy.deepcopy(r.spec)
            spec.id = r.id
            spec.name = r.name
            agents.append(spec)
        cfg.agents = agents
        return model_dump_compat(cfg)

    # ----- manual changes -----------------------------------------------------------
    def apply_manual_patch(self, patch: ManualPatch) -> None:
        for pr in patch.robots:
            try:
                r = self.robot_by_id(pr.id)
            except KeyError:
                continue
            if pr.x is not None:
                r.x = pr.x
            if pr.y is not None:
                r.y = pr.y
            if pr.theta is not None:
                r.theta = pr.theta
            if pr.name is not None:
                r.name = pr.name
                r.spec.name = pr.name
            if pr.spec is not None:
                # Preserve current id, but let the editor rename through name.
                spec = copy.deepcopy(pr.spec)
                spec.id = r.id
                r.spec = spec
                r.name = spec.name
            if pr.stop:
                r.v = r.omega = 0.0
                r.pwr_left = r.pwr_right = 0
        for po in patch.obstacles:
            if po.delete:
                self.config.obstacles = [o for o in self.config.obstacles if (o.id or "") != po.id]
                continue
            for obs in self.config.obstacles:
                if (obs.id or "") == po.id:
                    if po.spec is not None:
                        new_obs = copy.deepcopy(po.spec)
                        new_obs.id = po.id
                        idx = self.config.obstacles.index(obs)
                        self.config.obstacles[idx] = new_obs
                        obs = new_obs
                    if po.x is not None:
                        obs.x = po.x
                    if po.y is not None:
                        obs.y = po.y
                    if po.active_now is not None:
                        if po.active_now:
                            obs.appear_time_s = min(obs.appear_time_s, self.time_s)
                            obs.disappear_time_s = None
                        else:
                            obs.disappear_time_s = self.time_s
                    break
        if patch.note:
            self.events.append({"step": self.step_n, "time_s": self.time_s, "kind": "manual_patch", "note": patch.note})
        self.path_cache.clear()
        self.plan_memory.clear()
        self.planned_routes = []
        self._measure_sensors()

    def update_agent(self, agent: AgentSpec) -> Dict[str, object]:
        r = self.robot_by_id(agent.id)
        spec = copy.deepcopy(agent)
        r.spec = spec
        r.name = spec.name
        return self.to_public_state()

    def add_obstacle(self, req: AddObstacleRequest) -> None:
        obs = copy.deepcopy(req.obstacle)
        if not obs.id:
            obs.id = f"o{len(self.config.obstacles) + 1:02d}"
        if obs.label is None:
            obs.label = obs.id
        self.config.obstacles.append(obs)
        self.path_cache.clear()
        self.plan_memory.clear()
        self.planned_routes = []
        self.events.append({"step": self.step_n, "time_s": self.time_s, "kind": "add_obstacle", "id": obs.id})


# ----- single-agent physical replay / optimal route --------------------------------

def compute_single_optimal(req: SingleOptimalRequest) -> Dict[str, object]:
    result = visibility_shortest_path(
        start=(req.start.x, req.start.y),
        goal=(req.goal.x, req.goal.y),
        mission_length_m=req.mission_length_m,
        corridor_width_m=req.corridor_width_m,
        obstacles=req.obstacles,
        time_s=0.0,
        agent_radius_m=req.agent.radius_m or 0.1,
        safety_margin_m=req.safety_margin_m,
    )
    commands = route_to_commands(result.path, speed_mps=min(0.30, req.agent.max_speed_mps * 0.80), dt_s=0.20, max_power=req.agent.max_power) if result.status == "ok" else []
    return {
        "status": result.status,
        "reason": result.reason,
        "path": [{"x": x, "y": y} for x, y in result.path],
        "length_m": result.length_m,
        "expanded": result.expanded,
        "control_rows": commands,
    }


def parse_g_log(log_text: str) -> List[Dict[str, object]]:
    rows = []
    pattern = re.compile(r"ts=\[(?P<ts>\d+)\].*?tracks=\[(?P<left>[+-]?\d+),\s*(?P<right>[+-]?\d+)\].*?sonic=\[(?P<ang>[+-]?\d+),\s*(?P<dist>[+-]?\d+)cm\].*?tracking=\[(?P<tracking>[01]{4})\]")
    for line in log_text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        rows.append({
            "ts": int(m.group("ts")),
            "pwr_left": int(m.group("left")),
            "pwr_right": int(m.group("right")),
            "sonic_angle_deg": int(m.group("ang")),
            "sonic_distance_cm": int(m.group("dist")),
            "tracking_bits": m.group("tracking"),
        })
    return rows


def replay_single_log(req) -> Dict[str, object]:
    rows = parse_g_log(req.log_text)
    if not rows and req.csv_text.strip():
        buf = io.StringIO(req.csv_text)
        reader = csv.DictReader(buf)
        for row in reader:
            try:
                rows.append({"ts": int(float(row.get("ts", row.get("ts_ms", 0)))), "pwr_left": int(float(row.get("pwr_left", row.get("primary_pwr_left", 0)))), "pwr_right": int(float(row.get("pwr_right", row.get("primary_pwr_right", 0))))})
            except Exception:
                continue
    # Simple differential-drive dead reckoning.
    spec = req.mission.agent
    x, y = req.mission.start.x, req.mission.start.y
    theta = 0.0
    path = [(x, y)]
    last_ts = rows[0]["ts"] if rows else 0
    for row in rows:
        ts = row.get("ts", last_ts)
        dt_s = req.dt_s
        if last_ts and ts and ts != last_ts:
            dt_s = max(0.01, min(1.5, (int(ts) - int(last_ts)) / 1000.0))
        last_ts = ts
        left = int(row.get("pwr_left", 0)) / max(1, spec.max_power) * spec.max_speed_mps
        right = int(row.get("pwr_right", 0)) / max(1, spec.max_power) * spec.max_speed_mps
        v = (left + right) / 2.0
        omega = (right - left) / max(1e-6, spec.wheel_base_m)
        theta = angle_wrap(theta + omega * dt_s)
        x += v * math.cos(theta) * dt_s
        y += v * math.sin(theta) * dt_s
        path.append((x, y))
    optimal = compute_single_optimal(req.mission)
    traveled = sum(dist(path[i], path[i + 1]) for i in range(len(path) - 1)) if len(path) > 1 else 0.0
    return {
        "rows_parsed": len(rows),
        "replayed_path": [{"x": p[0], "y": p[1]} for p in path],
        "replayed_length_m": traveled,
        "final": {"x": x, "y": y, "theta": theta},
        "optimal": optimal,
        "excess_over_optimal_m": traveled - float(optimal.get("length_m", 0.0)) if optimal.get("status") == "ok" else None,
    }


# ----- benchmark ------------------------------------------------------------------

def run_benchmark(config: SimConfig, algorithms: Sequence[str], seeds: Sequence[int], max_steps: Optional[int] = None) -> Dict[str, object]:
    rows = []
    for alg in algorithms:
        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg.seed = seed
            cfg.algorithm = alg
            if max_steps is not None:
                cfg.max_steps = max_steps
            sim = SwarmSim(cfg)
            while not sim.done and sim.step_n < sim.config.max_steps:
                sim.step()
            metrics = sim.group_metrics()
            rows.append({
                "algorithm": alg,
                "seed": seed,
                "finish_reason": sim.finish_reason,
                "done": sim.done,
                "steps": sim.step_n,
                "time_s": sim.time_s,
                "centroid_path_length_m": metrics["centroid_path_length_m"],
                "excess_over_reference_m": metrics["excess_over_reference_m"],
                "formation_rms_m": metrics["formation_rms_m"],
                "formation_bad_pair_count": metrics["formation_bad_pair_count"],
                "collision_steps": metrics["collision_steps"],
                "lane_violations_group": metrics["lane_violations_group"],
                "total_energy_proxy": metrics["total_energy_proxy"],
                "sum_agent_path_length_m": metrics["sum_agent_path_length_m"],
            })
    summary = []
    for alg in algorithms:
        vals = [r for r in rows if r["algorithm"] == alg]
        success = [r for r in vals if r["finish_reason"] == "success"]
        summary.append({
            "algorithm": alg,
            "success_rate": len(success) / max(1, len(vals)),
            "mean_time_s": sum(float(r["time_s"]) for r in success) / max(1, len(success)),
            "mean_energy": sum(float(r["total_energy_proxy"]) for r in success) / max(1, len(success)),
            "mean_centroid_excess_m": sum(float(r["excess_over_reference_m"]) for r in success) / max(1, len(success)),
            "runs": len(vals),
        })
    return {"rows": rows, "summary": summary}
