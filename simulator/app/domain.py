from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from .models import AgentSpec, ObstacleSpec

Vec = Tuple[float, float]


@dataclass
class RobotState:
    id: str
    name: str
    spec: AgentSpec
    x: float
    y: float
    theta: float = 0.0
    v: float = 0.0
    omega: float = 0.0
    pwr_left: int = 0
    pwr_right: int = 0
    traveled_m: float = 0.0
    energy: float = 0.0
    collision_events: int = 0
    near_miss_events: int = 0
    lane_violations: int = 0
    cluster_switches: int = 0
    min_obstacle_clearance_m: float = float("inf")
    min_robot_clearance_m: float = float("inf")
    led: str = "---"
    buz: bool = False
    sonic_angle_deg: int = 0
    sonic_distance_cm: int = -1
    sonic_obstacle_id: str = ""
    sonic_hit_x: Optional[float] = None
    sonic_hit_y: Optional[float] = None
    sonic_visible_count: int = 0
    camera_pan_deg: int = 0
    camera_tilt_deg: int = 0
    tracking_bits: str = "0000"
    last_cmd_vx: float = 0.0
    last_cmd_vy: float = 0.0
    last_cmd_v: float = 0.0
    last_cmd_omega: float = 0.0
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    cluster_id: str = "main"
    role: str = "member"
    alive: bool = True
    blocked_steps: int = 0
    last_reason: str = "init"

    @property
    def pos(self) -> Vec:
        return (self.x, self.y)

    @property
    def radius_m(self) -> float:
        return float(self.spec.radius_m or 0.1)

    @property
    def powered(self) -> bool:
        return self.spec.powered


@dataclass
class PlannerCommand:
    vx: float = 0.0
    vy: float = 0.0
    led: Optional[str] = None
    buz: Optional[bool] = None
    target: Optional[Vec] = None
    cluster_id: Optional[str] = None
    role: Optional[str] = None
    reason: str = ""
    debug: Dict[str, object] = field(default_factory=dict)


@dataclass
class SharedObstacle:
    spec: ObstacleSpec
    first_seen_s: float
    last_seen_s: float
    seen_by: set[str] = field(default_factory=set)
