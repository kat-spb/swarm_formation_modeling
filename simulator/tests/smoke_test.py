from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app


def test_single_and_multi_smoke():
    c = TestClient(app)
    assert c.get('/api/algorithms').status_code == 200
    conf = {
        'name': 'smoke',
        'mission_length_m': 12,
        'corridor_width_m': 2.4,
        'dt_s': 0.2,
        'max_steps': 500,
        'algorithm': 'completion_astar',
        'n_agents': 1,
        'formation_shape': 'line',
        'formation_spacing_m': 0.38,
        'perception': {'mode': 'global_debug', 'shared_memory': True, 'obstacle_memory_s': 12, 'comms_range_m': 4, 'broadcast_obstacles': True, 'broadcast_positions': True},
        'obstacles': [{'id': 'r1', 'kind': 'rectangle', 'x': 4.0, 'y': 0.0, 'width_m': 0.55, 'height_m': 0.55, 'appear_time_s': 0}],
    }
    r = c.post('/api/sim', json=conf)
    assert r.status_code == 200, r.text
    sid = r.json()['id']
    state = None
    for _ in range(80):
        state = c.post(f'/api/sim/{sid}/step', json={'steps': 5}).json()
        if state['done']:
            break
    assert state['finish_reason'] == 'success'
    assert c.get(f'/api/sim/{sid}/log.csv').text.startswith('mission_id')


def test_dt_s_ui_mixup_is_recovered():
    c = TestClient(app)
    conf = {
        'name': 'dt_mixup',
        'mission_length_m': 12,
        'corridor_width_m': 2.4,
        'dt_s': 900,
        'max_steps': 500,
        'algorithm': 'smooth_adaptive_clusters',
        'n_agents': 1,
        'formation_shape': 'line',
        'formation_spacing_m': 0.38,
        'obstacles': [],
        'objective': {'objective_function': 'finish_first'},
    }
    r = c.post('/api/sim', json=conf)
    assert r.status_code == 200, r.text
    assert r.json()['config']['dt_s'] == 0.2


def test_algorithm_catalog_separates_objectives():
    c = TestClient(app)
    cat = c.get('/api/algorithms/catalog').json()
    keys = [a['key'] for a in cat['algorithms']]
    assert 'time_priority' not in keys
    assert 'energy_saver' not in keys
    assert any(o['key'] == 'min_makespan' for o in cat['objective_functions'])
