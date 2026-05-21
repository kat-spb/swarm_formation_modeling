# Swarm Formation Mission Simulator

Макет симулятора миссии: перенести рой из A в B на 12 м, обходить препятствия, завершать миссию и проверять строй через формальную матрицу расстояний.
Стек: FastAPI + алгоритмы (Python) + визуализатор (html, cytoscape.js)

## Запуск

```bash
cd swarm_formatioт_modeling
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Открыть:

```text
http://localhost:8000/
```

## Назначение программы

Создавать миссии выполнения задач группой (роем) и симулировать их выполнение с использованием системы команд физических устройств.

## Ключевые возможности

### Раздельные фазы моделирования

Интерфейс разбит на пять фаз:

1. Начальное состояние: метрические ограничения сцены, физические  ограничения (например, тип покрытия для едущей группы), формация, физические свойства агентов, препятствия. Импорт/экспорт описаний.
2. Параметры моделирования: выбор режима симуляции централизованное/агентное исполнение, целевая функция, алгоритмы, способы коммуникации.
3. Запуск симуляции и визуализация: выполнение по шагам или , pan/zoom, обтекание, метрики, логи выполнения миссии.
4. Бенчмарки: сравнение алгоритмов и исполнение физических логов прохождения в режиме симуляции.
5. Анализ результатов: каталог алгоритмов, объяснимость и итоговые данные.

### 1. Алгоритмы, доступные в системе

В миссии есть выпадающий список алгоритмов. Кнопка **«Описание / редактор алгоритмов»** открывает модалку со стандартными benchmark-baseline и mission-алгоритмами:

- `astar_vs` — A* + virtual structure;
- `apf` — Artificial Potential Fields;
- `dwa` — Dynamic Window Approach;
- `rvo_lite` — ORCA/RVO-inspired reciprocal avoidance;
- `smooth_adaptive_clusters` — основной completion-first split/bypass/rejoin;
- `completion_astar`, `adaptive_clusters`, `virtual_structure`, `apf_reactive` — дополнительные исполняемые варианты;
- дополнительные алгоритмы: `cbs`, `rrt_star`, `dstar_lite` доступны в каталоге описаний, но не выбираются как исполняемые.

В карточке алгоритма показаны назначение, ограничения, ссылка на описание метода и редактируемый JSON параметров. Эти параметры передаются в `StepRequest.algorithm_params` и сохраняются в `SimConfig.algorithm_params`. Целевая функция вынесена отдельно от алгоритма: алгоритм описывает способ планирования/управления, а objective только меняет критерий оценки и приоритеты.

API:

```text
GET /api/algorithms
GET /api/algorithms/catalog
```

### 2. Редактор строя теперь работает от предопределённого строя

Во вкладке **Строй** можно выбрать `triangle`, `square`, `hex`, `circle`, `line`, задать N и шаг, загрузить preset, затем открыть модалку графа и матрицы.

Поддержаны сценарии:

- выбрать известный строй и редактировать его;
- добавить агента в текущую сетку;
- вручную двигать вершины графа;
- вручную править матрицу расстояний `D[i,j]`;
- сделать строй `custom`;
- выровнять пользовательский строй обратно по выбранной сетке.

API:

```text
GET  /api/formation/presets
POST /api/formation/preset
POST /api/formation/align
POST /api/formation/validate
```

### 3. Препятствия: random, точные позиции, клик-редактор и карты для benchmark

Во вкладке **Препятствия** есть:

- предопределённые карты для benchmark;
- генерация случайных препятствий по count/seed/dynamic fraction;
- быстрое добавление прямоугольника по центру, ширине, высоте и времени появления;
- JSON-редактор `ObstacleSpec[]`;
- клик по препятствию на canvas открывает модалку свойств;
- в модалке можно править `label`, тип, размеры, центр, `appear_time_s`, `disappear_time_s`, opacity.

Поддерживаются `rectangle`, `circle`, `polygon`. Для физических контрольных прогонов основной тип — прямоугольник со сторонами, параллельными полосе.

API:

```text
GET  /api/obstacle-maps
POST /api/obstacle-maps
POST /api/obstacles/random
POST /api/obstacles/validate
```

Предопределённые карты:

```text
clear_lane
single_gate
two_gates
rect_gate
central_block
zigzag_rects
split_rejoin
narrow_slalom
dynamic_popup
mixed_circles_rects
```

### 4. Плавное перестроение и визуализация обтекания сохранены

Алгоритм по умолчанию `smooth_adaptive_clusters` показывает:

- цветовой кластер агента (`upper`, `lower`, `blocked`, `rejoined`);
- пунктирные индивидуальные маршруты;
- raw waypoint;
- сглаженный текущий control setpoint;
- финальные слоты;
- следы агентов;
- трек центра тяжести.

### 5. Контрольные примеры и логи

В `examples/` лежат контрольные конфигурации и логи:

- `single_clear_12m.json` + `control_single_clear_12m.csv` + `control_single_clear_12m.g.log`;
- `single_rectangles_physical.json` + `control_single_rectangles_physical.csv` + `control_single_rectangles_physical.g.log`;
- `optimal_single_rectangles_path.json` + `optimal_single_rectangles_control.csv`;
- `formation_7_smooth_bypass.json` + CSV/g.log;
- `formation_7_adaptive_split.json` + CSV/g.log;
- `formation_6_dynamic_popup.json` + CSV/g.log;
- `formation_4_gate_rejoin.json` + CSV/g.log.

Индекс примеров доступен через:

```text
GET /api/examples
```

## Основные API

```text
POST /api/sim
GET  /api/sim/{sim_id}
POST /api/sim/{sim_id}/step
POST /api/sim/{sim_id}/manual
PUT  /api/sim/{sim_id}/agent/{agent_id}
POST /api/sim/{sim_id}/obstacle
GET  /api/sim/{sim_id}/export.json
GET  /api/sim/{sim_id}/log.csv
GET  /api/sim/{sim_id}/log/g
GET  /api/sim/{sim_id}/log/g.txt
POST /api/single/optimal
POST /api/single/replay-log
POST /api/sim/{sim_id}/benchmark
```

## Примечания по модели

Это исследовательский макет, а не физически откалиброванный контроллер. Модель учитывает размеры агента, ограничения скорости/ускорения/поворота, коридор, физику покрытия через `scene_physics`, препятствия, collision/near-miss, personal path/energy, мгновенные/средние скорости и финальную матрицу строя. Для переноса на физический рой нужно калибровать `power_to_speed`, задержки команд, шум ultrasonic sensor, трение, динамику гусениц и протокол обмена.
