from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .algorithms.registry import available_planners, planner_catalog
from .formation import align_formation_to_preset, make_formation_spec, validate_formation
from .geometry import obstacle_approx_radius, validate_obstacle_geometry
from .models import (
    AddObstacleRequest,
    AgentUpdateRequest,
    BenchmarkRequest,
    FormationAlignRequest,
    FormationPresetRequest,
    FormationValidationRequest,
    ManualPatch,
    ObstacleMapRequest,
    ObstacleValidationRequest,
    RandomObstaclesRequest,
    ReplayLogRequest,
    SimConfig,
    SingleOptimalRequest,
    StepRequest,
)
from .obstacle_maps import available_obstacle_maps, map_obstacles, random_obstacles
from .sim import SwarmSim, compute_single_optimal, replay_single_log, run_benchmark

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
EXAMPLES_DIR = BASE_DIR / "examples"

app = FastAPI(title="Swarm Formation Mission Simulator", version="0.6.2")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/examples", StaticFiles(directory=str(EXAMPLES_DIR)), name="examples")

SIMS: Dict[str, SwarmSim] = {}


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/algorithms")
def algorithms():
    return available_planners()


@app.get("/api/algorithms/catalog")
def algorithms_catalog():
    return planner_catalog()

@app.get("/api/objectives")
def objectives():
    return {"objective_functions": planner_catalog().get("objective_functions", [])}



@app.get("/api/formation/presets")
def formation_presets():
    return {
        "presets": [
            {"key": "triangle", "title": "Треугольная решётка", "description": "Заполнение рядами: удобно для клина и компактного строя."},
            {"key": "square", "title": "Квадратная решётка", "description": "Равномерная сетка; проще физически разметить на полу."},
            {"key": "hex", "title": "Шестиугольная решётка", "description": "Плотная упаковка с близкими соседними расстояниями."},
            {"key": "circle", "title": "Круг", "description": "Кольцевая формация; полезна для обхода центральных препятствий."},
            {"key": "line", "title": "Линия", "description": "Колонна или шеренга; полезна для узких проходов и одиночных проверок."},
            {"key": "custom", "title": "Пользовательский", "description": "Текущий строй: можно двигать вершины и править матрицу расстояний."},
        ]
    }


@app.post("/api/formation/preset")
def formation_preset(req: FormationPresetRequest):
    if req.formation_shape == "custom":
        raise HTTPException(status_code=400, detail="custom formation is edited as FormationSpec")
    try:
        return make_formation_spec(req.formation_shape, req.n_agents, req.formation_spacing).model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/formation/align")
def formation_align(req: FormationAlignRequest):
    try:
        shape = "triangle" if req.formation_shape == "custom" else req.formation_shape
        spec = align_formation_to_preset(req.formation, shape, req.formation_spacing, req.recompute_matrix, req.preserve_ids)
        return spec.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/formation/validate")
def formation_validate(req: FormationValidationRequest):
    try:
        return validate_formation(req.formation, req.corridor_width, req.default_robot_radius, req.safety_margin)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/obstacle-maps")
def obstacle_maps_index():
    return {"maps": available_obstacle_maps()}


@app.post("/api/obstacle-maps")
def obstacle_map(req: ObstacleMapRequest):
    try:
        return {"key": req.key, "obstacles": [o.model_dump() for o in map_obstacles(req)]}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/obstacle-maps/preset")
def obstacle_map_preset(req: ObstacleMapRequest):
    return obstacle_map(req)


@app.post("/api/obstacle-maps/apply")
def obstacle_map_apply(req: ObstacleMapRequest):
    return obstacle_map(req)


@app.post("/api/obstacles/random")
def obstacles_random(req: RandomObstaclesRequest):
    try:
        return {"obstacles": [o.model_dump() for o in random_obstacles(req)]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/obstacles/validate")
def obstacles_validate(req: ObstacleValidationRequest):
    out = []
    errors = []
    warnings = []
    for i, obs in enumerate(req.obstacles):
        item = validate_obstacle_geometry(obs)
        radius = obstacle_approx_radius(obs)
        oid = obs.id or f"o{i + 1:02d}"
        if obs.x - radius < -0.5 or obs.x + radius > req.mission_length + 0.5:
            item["warnings"].append("obstacle is partly outside the mission x-range")
        if abs(obs.y) + radius > req.corridor_width / 2:
            item["warnings"].append("obstacle is partly outside the corridor/lane")
        item.update({"id": oid, "approx_radius_m": radius})
        out.append(item)
        errors.extend([f"{oid}: {e}" for e in item.get("errors", [])])
        warnings.extend([f"{oid}: {w}" for w in item.get("warnings", [])])
    return {"ok": not errors, "errors": errors, "warnings": warnings, "items": out}


@app.post("/api/sim")
def create_sim(config: SimConfig):
    try:
        sim = SwarmSim(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    SIMS[sim.id] = sim
    return sim.to_public_state()


@app.get("/api/sim/{sim_id}")
def get_sim(sim_id: str):
    return _get(sim_id).to_public_state()


@app.post("/api/sim/{sim_id}/step")
def step_sim(sim_id: str, req: StepRequest):
    sim = _get(sim_id)
    try:
        if req.algorithm_params is not None:
            sim.config.algorithm_params = req.algorithm_params
        for _ in range(req.steps):
            sim.step(req.algorithm)
            if sim.done:
                break
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return sim.to_public_state()


@app.post("/api/sim/{sim_id}/manual")
def manual_patch(sim_id: str, patch: ManualPatch):
    sim = _get(sim_id)
    sim.apply_manual_patch(patch)
    return sim.to_public_state()


@app.put("/api/sim/{sim_id}/agent/{agent_id}")
def update_agent(sim_id: str, agent_id: str, req: AgentUpdateRequest):
    sim = _get(sim_id)
    if req.agent.id != agent_id:
        req.agent.id = agent_id
    try:
        return sim.update_agent(req.agent)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")


@app.post("/api/sim/{sim_id}/obstacle")
def add_obstacle(sim_id: str, req: AddObstacleRequest):
    sim = _get(sim_id)
    sim.add_obstacle(req)
    return sim.to_public_state()


@app.post("/api/sim/{sim_id}/benchmark")
def benchmark(sim_id: str, req: BenchmarkRequest):
    sim = _get(sim_id)
    try:
        return run_benchmark(sim.config, req.algorithms, req.seeds, req.max_steps)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/sim/{sim_id}/metrics/agents")
def metrics_agents(sim_id: str):
    return {"agents": _get(sim_id).agent_metrics()}

@app.get("/api/sim/{sim_id}/initial-summary")
def initial_summary(sim_id: str):
    return _get(sim_id).initial_summary()


@app.get("/api/sim/{sim_id}/agent/{agent_id}/status")
def agent_status(sim_id: str, agent_id: str):
    try:
        return _get(sim_id).agent_online_status(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="agent not found")



@app.get("/api/sim/{sim_id}/log/g")
def log_g(sim_id: str):
    return {"lines": _get(sim_id).status_lines_g()}


@app.get("/api/sim/{sim_id}/log/g.txt")
def log_g_txt(sim_id: str):
    text = "\n".join(_get(sim_id).status_lines_g()) + "\n"
    return Response(text, media_type="text/plain; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="mission_{sim_id}_g.log"'})


@app.get("/api/sim/{sim_id}/log.csv")
def log_csv_dot(sim_id: str):
    sim = _get(sim_id)
    return Response(
        sim.csv_text(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="mission_{sim_id}.csv"'},
    )


@app.get("/api/sim/{sim_id}/log/csv")
def log_csv(sim_id: str):
    return log_csv_dot(sim_id)

@app.get("/api/sim/{sim_id}/log/tank_commands.csv")
def log_tank_commands_csv(sim_id: str):
    sim = _get(sim_id)
    return Response(
        sim.tank_command_csv_text(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="mission_{sim_id}_tank_commands.csv"'},
    )


@app.get("/api/sim/{sim_id}/log/tank_commands.txt")
def log_tank_commands_txt(sim_id: str):
    sim = _get(sim_id)
    return Response(
        sim.tank_command_text(),
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="mission_{sim_id}_tank_commands.txt"'},
    )


@app.get("/api/sim/{sim_id}/logs.zip")
def log_bundle_zip(sim_id: str):
    sim = _get(sim_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"mission_{sim_id}.csv", sim.csv_text())
        zf.writestr(f"mission_{sim_id}_g.log", "\n".join(sim.status_lines_g()) + "\n")
        zf.writestr(f"mission_{sim_id}_tank_commands.csv", sim.tank_command_csv_text())
        zf.writestr(f"mission_{sim_id}_tank_commands.txt", sim.tank_command_text())
        zf.writestr(f"mission_{sim_id}_config.json", json.dumps(sim.export_config(), ensure_ascii=False, indent=2))
    buf.seek(0)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="mission_{sim_id}_logs.zip"'},
    )



@app.get("/api/sim/{sim_id}/export.json")
def export_sim(sim_id: str):
    sim = _get(sim_id)
    text = json.dumps(sim.export_config(), ensure_ascii=False, indent=2)
    return Response(text, media_type="application/json; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="mission_{sim_id}.json"'})


@app.post("/api/single/optimal")
def single_optimal(req: SingleOptimalRequest):
    try:
        return compute_single_optimal(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/api/single/replay-log")
def single_replay(req: ReplayLogRequest):
    try:
        return replay_single_log(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/api/examples")
def examples_index():
    index_path = EXAMPLES_DIR / "index.json"
    if index_path.exists():
        return JSONResponse(json.loads(index_path.read_text(encoding="utf-8")))
    files = sorted(p.name for p in EXAMPLES_DIR.glob("*.json"))
    return {"examples": [{"name": f, "config": f} for f in files]}


def _get(sim_id: str) -> SwarmSim:
    sim = SIMS.get(sim_id)
    if sim is None:
        raise HTTPException(status_code=404, detail="simulation not found")
    return sim
