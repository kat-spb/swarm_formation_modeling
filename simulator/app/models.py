from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Vec2Model(BaseModel):
    x: float = 0.0
    y: float = 0.0


class FormationNode(BaseModel):
    id: str
    name: Optional[str] = None
    x: float
    y: float
    powered: bool = True


class FormationSpec(BaseModel):
    """Formal mission formation.

    Coordinates are an embedding used for rendering and target placement.
    distance_matrix is the formal definition of the desired final formation;
    if it is missing, it is computed from coordinates.
    """

    name: str = "custom"
    nodes: List[FormationNode] = Field(default_factory=list)
    distance_matrix: Optional[List[List[float]]] = None
    tolerance_m: float = Field(0.10, ge=0.0)
    edge_tolerance_m: float = Field(0.09, ge=0.0)
    locked_edges: List[List[str]] = Field(default_factory=list)
    assignment_mode: Literal["identity", "free"] = "identity"


class FormationPresetRequest(BaseModel):
    n_agents: int = Field(7, ge=1, le=80)
    formation_shape: Literal["triangle", "square", "hex", "circle", "line", "custom"] = "triangle"
    formation_spacing: float = Field(0.38, gt=0.05, le=3.0)
    preserve_ids: bool = False


class FormationAlignRequest(BaseModel):
    formation: FormationSpec
    formation_shape: Literal["triangle", "square", "hex", "circle", "line", "custom"] = "triangle"
    formation_spacing: float = Field(0.38, gt=0.05, le=3.0)
    preserve_ids: bool = True
    recompute_matrix: bool = True


class FormationValidationRequest(BaseModel):
    formation: FormationSpec
    corridor_width: float = Field(2.4, gt=0.1)
    default_robot_radius: float = Field(0.09, gt=0.005)
    safety_margin: float = Field(0.05, ge=0.0)


class AgentSpec(BaseModel):
    id: str
    name: str = "agent"
    powered: bool = True
    length_m: float = Field(0.18, gt=0.01)
    width_m: float = Field(0.13, gt=0.01)
    radius_m: Optional[float] = Field(None, gt=0.005)
    mass_kg: float = Field(0.35, gt=0.01)
    wheel_base_m: float = Field(0.16, gt=0.01)
    max_speed_mps: float = Field(0.38, gt=0.01)
    max_reverse_mps: float = Field(0.12, ge=0.0)
    max_omega_radps: float = Field(3.2, gt=0.01)
    max_accel_mps2: float = Field(0.55, gt=0.01)
    max_omega_accel_radps2: float = Field(5.5, gt=0.01)
    max_power: int = Field(1000, ge=1)
    power_to_speed: float = Field(0.00038, gt=0.0)
    # Physical tank command calibration. Defaults match tank.c:
    # min duty 4000, step 1000, PWM period 20000.  They are optional for
    # simulation dynamics, but are written to logs for real reproduction.
    tank_pwm_min: int = Field(4000, ge=0)
    tank_pwm_step: int = Field(1000, ge=1)
    tank_pwm_period: int = Field(20000, ge=1)
    pwm_correction_left: int = 0
    pwm_correction_right: int = 0
    energy_coeff: float = Field(1.0, gt=0.0)
    sonic_range_m: float = Field(4.0, gt=0.05)
    sonic_fov_deg: float = Field(80.0, gt=1.0, le=360.0)
    camera_pan_deg: int = 0
    camera_tilt_deg: int = 0
    color: str = "#4fb3ff"

    @model_validator(mode="after")
    def derive_radius(self):
        import math

        if self.radius_m is None:
            self.radius_m = 0.5 * math.sqrt(self.length_m * self.length_m + self.width_m * self.width_m)
        return self


class ObstacleSpec(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    kind: Literal["rectangle", "circle", "polygon"] = "rectangle"
    x: float = 4.0
    y: float = 0.0
    width_m: float = Field(0.50, gt=0.001)
    height_m: float = Field(0.35, gt=0.001)
    rotation_deg: float = 0.0
    radius_m: float = Field(0.25, gt=0.001)
    vertices: Optional[List[Vec2Model]] = None
    side_lengths_m: Optional[List[float]] = None
    interior_angles_deg: Optional[List[float]] = None
    appear_time_s: float = Field(0.0, ge=0.0)
    disappear_time_s: Optional[float] = Field(None, ge=0.0)
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    active_by_default: bool = True
    opacity: float = Field(0.35, ge=0.05, le=1.0)

    @model_validator(mode="after")
    def validate_times(self):
        if self.disappear_time_s is not None and self.disappear_time_s < self.appear_time_s:
            raise ValueError("disappear_time_s must be >= appear_time_s")
        return self


class ObstacleValidationRequest(BaseModel):
    obstacles: List[ObstacleSpec] = Field(default_factory=list)
    corridor_width: float = Field(2.4, gt=0.1)
    mission_length: float = Field(12.0, gt=0.1)
    default_agent_radius_m: float = Field(0.09, gt=0.005)
    safety_margin_m: float = Field(0.05, ge=0.0)


class ObstacleRandomRequest(BaseModel):
    count: int = Field(5, ge=0, le=100)
    mission_length_m: float = Field(12.0, gt=0.1)
    corridor_width_m: float = Field(2.4, gt=0.1)
    seed: int = 42
    kind: Literal["rectangle", "circle", "mixed"] = "rectangle"
    # UI preset for obstacle sizes.  The backend still accepts explicit custom ranges.
    size_preset: Literal["small", "medium", "large", "custom"] = "medium"
    min_x_m: float = 1.0
    max_x_m: Optional[float] = None
    min_gap_m: float = Field(0.55, ge=0.0)
    appear_default_s: float = Field(0.0, ge=0.0)
    # Accept 0..1 fraction, 1..100 percent, or absolute count <= count.
    dynamic_fraction: float = Field(0.0, ge=0.0)
    dynamic_count: Optional[int] = Field(None, ge=0, le=100)
    width_min_m: float = Field(0.25, gt=0.01)
    width_max_m: float = Field(0.75, gt=0.01)
    height_min_m: float = Field(0.20, gt=0.01)
    height_max_m: float = Field(0.65, gt=0.01)

    @model_validator(mode="before")
    @classmethod
    def normalize_random_obstacle_ui(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        preset = d.get("size_preset") or d.get("type_size_preset") or d.get("obstacle_size")
        if preset in {"small", "medium", "large"}:
            sizes = {
                "small": (0.25, 0.45, 0.20, 0.38, 0.42),
                "medium": (0.40, 0.75, 0.30, 0.65, 0.55),
                "large": (0.70, 1.20, 0.50, 0.95, 0.72),
            }[preset]
            d["size_preset"] = preset
            # Preset is a typology; if custom ranges are absent, fill them.
            d.setdefault("width_min_m", sizes[0])
            d.setdefault("width_max_m", sizes[1])
            d.setdefault("height_min_m", sizes[2])
            d.setdefault("height_max_m", sizes[3])
            d.setdefault("min_gap_m", sizes[4])
        elif preset == "custom":
            d["size_preset"] = "custom"
        # Accept dynamic_count as a more obvious UI field.
        try:
            count = int(d.get("count", 0) or 0)
            dyn_count = d.get("dynamic_count")
            if dyn_count is not None:
                dc = max(0, min(count, int(float(dyn_count)))) if count > 0 else 0
                d["dynamic_fraction"] = (dc / count) if count > 0 else 0.0
        except Exception:
            pass
        return d

    @model_validator(mode="after")
    def normalize_random_fields(self):
        df = float(self.dynamic_fraction or 0.0)
        if df > 1.0:
            if self.count > 0 and df <= self.count and abs(df - round(df)) < 1e-9:
                df = df / float(self.count)
            elif df <= 100.0:
                df = df / 100.0
            else:
                df = 1.0
        self.dynamic_fraction = max(0.0, min(1.0, df))
        if self.width_min_m > self.width_max_m:
            self.width_min_m, self.width_max_m = self.width_max_m, self.width_min_m
        if self.height_min_m > self.height_max_m:
            self.height_min_m, self.height_max_m = self.height_max_m, self.height_min_m
        max_h = max(0.05, self.corridor_width_m - 0.18)
        self.height_min_m = min(self.height_min_m, max_h)
        self.height_max_m = min(self.height_max_m, max_h)
        if self.max_x_m is not None and self.max_x_m < self.min_x_m:
            self.min_x_m, self.max_x_m = self.max_x_m, self.min_x_m
        return self


RandomObstaclesRequest = ObstacleRandomRequest


class ObstacleMapRequest(BaseModel):
    key: str
    mission_length_m: float = Field(12.0, gt=0.1)
    corridor_width_m: float = Field(2.4, gt=0.1)


ObstacleMapPresetRequest = ObstacleMapRequest


class ScenePhysicsConfig(BaseModel):
    """Physical properties of the scene/track surface.

    This is intentionally a compact model: coefficients scale the agent limits
    and energy proxy so the same algorithm can be tested on different surfaces.
    """

    surface_type: Literal["smooth_floor", "mat", "carpet", "rough", "custom"] = "smooth_floor"
    friction_coefficient: float = Field(0.85, gt=0.05, le=3.0)
    rolling_resistance: float = Field(0.02, ge=0.0, le=1.0)
    speed_scale: float = Field(1.0, gt=0.05, le=3.0)
    accel_scale: float = Field(1.0, gt=0.05, le=3.0)
    energy_scale: float = Field(1.0, gt=0.05, le=10.0)
    note: str = ""


class PerceptionConfig(BaseModel):
    mode: Literal["local", "global_debug"] = "local"
    shared_memory: bool = True
    obstacle_memory_s: float = Field(12.0, ge=0.0)
    comms_range_m: float = Field(4.0, ge=0.0)
    broadcast_obstacles: bool = True
    broadcast_positions: bool = True


class ObjectiveConfig(BaseModel):
    # Objective is a mission/modeling parameter, not an algorithm key.
    objective_function: Literal[
        "finish_first",
        "shortest_centroid",
        "min_makespan",
        "min_total_energy",
        "balanced",
    ] = "finish_first"
    # Backward-compatible alias used by older saved configs and logs.
    completion_policy: Literal[
        "finish_first",
        "shortest_centroid",
        "min_time_all_powered",
        "min_energy",
        "balanced",
    ] = "finish_first"
    collective_goal_weight: float = Field(1.0, ge=0.0)
    agent_survival_weight: float = Field(1.0, ge=0.0)
    final_place_weight: float = Field(1.0, ge=0.0)
    energy_weight: float = Field(0.20, ge=0.0)
    time_weight: float = Field(0.25, ge=0.0)
    centroid_length_weight: float = Field(0.25, ge=0.0)
    allow_reassignment: bool = False
    require_all_powered_agents: bool = True
    success_requires_formation: bool = True
    success_requires_no_collision: bool = False

    @model_validator(mode="after")
    def sync_aliases(self):
        map_to_completion = {
            "min_makespan": "min_time_all_powered",
            "min_total_energy": "min_energy",
        }
        # Keep completion_policy as the text used in existing metrics and CSV.
        self.completion_policy = map_to_completion.get(self.objective_function, self.objective_function)  # type: ignore[assignment]
        return self


class MotionLimits(BaseModel):
    safety_margin_m: float = Field(0.055, ge=0.0)
    robot_robot_margin_m: float = Field(0.04, ge=0.0)
    obstacle_margin_m: float = Field(0.05, ge=0.0)
    replan_period_steps: int = Field(3, ge=1, le=100)
    max_reconfig_speed_mps: float = Field(0.30, gt=0.0)
    smooth_reconfiguration: bool = True
    setpoint_smoothing_alpha: float = Field(0.70, gt=0.0, le=1.0)
    setpoint_max_rate_mps: float = Field(0.55, gt=0.01)
    min_route_lookahead_m: float = Field(0.30, gt=0.02)
    route_preview_points: int = Field(28, ge=2, le=200)
    agent_trail_points: int = Field(140, ge=0, le=2000)
    keep_full_trails: bool = True
    completion_unstuck_speed_mps: float = Field(0.055, ge=0.0, le=0.5)
    cluster_hysteresis_steps: int = Field(2, ge=0, le=20)
    avoidance_horizon_s: float = Field(1.2, gt=0.05)
    enable_safety_filter: bool = True
    allow_reverse: bool = True
    stop_if_no_path: bool = False


class SimConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = "mission"
    mission_length_m: float = Field(12.0, gt=0.1)
    corridor_width_m: float = Field(2.4, gt=0.1)
    dt_s: float = Field(0.20, gt=0.01, le=2.0)
    max_steps: int = Field(900, ge=1, le=100000)
    seed: int = 42
    algorithm: str = "smooth_adaptive_clusters"
    algorithm_params: Dict[str, Any] = Field(default_factory=dict)
    grid_resolution_m: float = Field(0.12, gt=0.03, le=1.0)
    formation: Optional[FormationSpec] = None
    n_agents: int = Field(7, ge=1, le=80)
    formation_shape: Literal["triangle", "square", "hex", "circle", "line", "custom"] = "triangle"
    formation_spacing_m: float = Field(0.38, gt=0.05, le=3.0)
    agents: Optional[List[AgentSpec]] = None
    default_agent: AgentSpec = Field(default_factory=lambda: AgentSpec(id="default", name="default"))
    obstacles: List[ObstacleSpec] = Field(default_factory=list)
    random_obstacles: bool = False
    random_obstacle_count: int = Field(0, ge=0, le=100)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    scene_physics: ScenePhysicsConfig = Field(default_factory=ScenePhysicsConfig)
    limits: MotionLimits = Field(default_factory=MotionLimits)
    start_x_m: float = 0.0
    start_y_m: float = 0.0
    finish_centroid_tolerance_m: float = Field(0.18, ge=0.0)
    finish_agent_tolerance_m: float = Field(0.16, ge=0.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_common_ui_mixups(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        # v5 UI could accidentally send max_steps into dt_s.  Do not fail the
        # mission creation with 422; keep max_steps and restore a safe dt.
        try:
            dt = float(d.get("dt_s", 0.20))
        except Exception:
            dt = 0.20
        if dt > 2.0:
            d.setdefault("_ui_warning", f"dt_s={dt} looked like max_steps; reset to 0.20s")
            d["dt_s"] = 0.20
            if "max_steps" not in d and dt.is_integer():
                d["max_steps"] = int(dt)
        # Objective aliases from older configs.
        obj = d.get("objective")
        if isinstance(obj, dict) and "objective_function" not in obj and "completion_policy" in obj:
            rev = {"min_time_all_powered": "min_makespan", "min_energy": "min_total_energy"}
            obj = dict(obj)
            obj["objective_function"] = rev.get(obj.get("completion_policy"), obj.get("completion_policy"))
            d["objective"] = obj
        return d


class StepRequest(BaseModel):
    steps: int = Field(1, ge=1, le=5000)
    algorithm: Optional[str] = None
    algorithm_params: Optional[Dict[str, Any]] = None


class RobotPatch(BaseModel):
    id: str
    x: Optional[float] = None
    y: Optional[float] = None
    theta: Optional[float] = None
    stop: bool = False
    spec: Optional[AgentSpec] = None
    name: Optional[str] = None


class ObstaclePatch(BaseModel):
    id: str
    x: Optional[float] = None
    y: Optional[float] = None
    delete: bool = False
    spec: Optional[ObstacleSpec] = None
    active_now: Optional[bool] = None


class ManualPatch(BaseModel):
    robots: List[RobotPatch] = Field(default_factory=list)
    obstacles: List[ObstaclePatch] = Field(default_factory=list)
    note: str = ""


class AgentUpdateRequest(BaseModel):
    agent: AgentSpec


class AddObstacleRequest(BaseModel):
    obstacle: ObstacleSpec = Field(default_factory=ObstacleSpec)


class BenchmarkRequest(BaseModel):
    algorithms: List[str] = Field(
        default_factory=lambda: [
            "astar_vs",
            "apf",
            "dwa",
            "rvo_lite",
            "smooth_adaptive_clusters",
            "completion_astar",
            "adaptive_clusters",
            "virtual_structure",
            "apf_reactive",
        ]
    )
    seeds: List[int] = Field(default_factory=lambda: [1, 2, 3])
    max_steps: Optional[int] = Field(None, ge=1, le=100000)


class SingleOptimalRequest(BaseModel):
    mission_length_m: float = Field(12.0, gt=0.1)
    corridor_width_m: float = Field(2.4, gt=0.1)
    start: Vec2Model = Field(default_factory=lambda: Vec2Model(x=0.0, y=0.0))
    goal: Vec2Model = Field(default_factory=lambda: Vec2Model(x=12.0, y=0.0))
    agent: AgentSpec = Field(default_factory=lambda: AgentSpec(id="solo", name="solo"))
    obstacles: List[ObstacleSpec] = Field(default_factory=list)
    safety_margin_m: float = Field(0.05, ge=0.0)
    resolution_m: float = Field(0.04, gt=0.005)


class ReplayLogRequest(BaseModel):
    mission: SingleOptimalRequest
    log_text: str = ""
    csv_text: str = ""
    dt_s: float = Field(0.20, gt=0.01)
