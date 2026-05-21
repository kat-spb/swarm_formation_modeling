from __future__ import annotations

from typing import Dict, Type

from .base import Planner
from .mission import (
    APFPlanner,
    APFReactivePlanner,
    AStarVirtualStructurePlanner,
    AdaptiveClustersPlanner,
    CompletionAStarPlanner,
    DWAPlanner,
    EnergySaverPlanner,
    RVOLitePlanner,
    SmoothAdaptiveClustersPlanner,
    TimePriorityPlanner,
    VirtualStructurePlanner,
)

# Compatibility keys are kept executable, but the UI does not present
# objective presets as algorithms.  The objective is selected in the mission
# modeling phase and can be combined with any actual algorithm.
_PLANNERS: Dict[str, Type[Planner]] = {
    AStarVirtualStructurePlanner.key: AStarVirtualStructurePlanner,
    APFPlanner.key: APFPlanner,
    DWAPlanner.key: DWAPlanner,
    RVOLitePlanner.key: RVOLitePlanner,
    SmoothAdaptiveClustersPlanner.key: SmoothAdaptiveClustersPlanner,
    "cluster_adaptive": AdaptiveClustersPlanner,  # v2-compatible key: explicit split/bypass/rejoin
    CompletionAStarPlanner.key: CompletionAStarPlanner,
    AdaptiveClustersPlanner.key: AdaptiveClustersPlanner,
    VirtualStructurePlanner.key: VirtualStructurePlanner,
    APFReactivePlanner.key: APFReactivePlanner,
    TimePriorityPlanner.key: TimePriorityPlanner,
    EnergySaverPlanner.key: EnergySaverPlanner,
}

_ALGORITHM_META: Dict[str, Dict[str, object]] = {
    "astar_vs": {
        "family": "standard benchmark algorithms",
        "bench_default": True,
        "label": "A* + Virtual Structure",
        "description": "Глобальный A* ищет путь для центра тяжести; вокруг найденной точки движется виртуальная структура строя.",
        "explainability": "Решение объясняется как последовательность клеток пути центра, затем как смещение каждого агента на фиксированный offset строя. Если крайний агент врезался, причина обычно в том, что путь центра был свободен, а край строя нет.",
        "steps": [
            "Дискретизировать полосу в клеточный граф с препятствиями, раздутыми на радиус агента/строя.",
            "A* минимизирует g(n)+h(n), где g — длина уже пройденного пути, h — эвристика до цели.",
            "Центр тяжести следует по пути; каждому агенту назначается слот formation_offset_i.",
            "Низкоуровневый контроллер переводит вектор к слоту в команды гусениц с ограничениями скорости/ускорения.",
        ],
        "inputs": ["карта препятствий/наблюдаемая карта", "начальная и конечная точка центра", "offset-координаты строя", "радиусы агентов"],
        "outputs": ["путь центра тяжести", "целевой слот каждого агента", "управляющая скорость каждого агента"],
        "limitations": "Хороший эталон длины центра, но слабый при широких формациях: свободный центр не означает свободные края.",
        "reference_label": "Hart, Nilsson, Raphael — A Formal Basis for the Heuristic Determination of Minimum Cost Paths, 1968",
        "reference_url": "https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf",
        "references": [
            {"title": "Lewis & Tan — High precision formation control of mobile robots using virtual structures", "url": "https://link.springer.com/article/10.1023/A:1008814708459"}
        ],
        "params_schema": {"speed_scale": {"default": 0.72}, "group_inflation_m": {"default": 0.0}, "use_centroid_path": {"default": True}},
    },
    "apf": {
        "family": "standard benchmark algorithms",
        "bench_default": True,
        "label": "APF / Artificial Potential Fields",
        "description": "Реактивное поле: цель притягивает агента, препятствия и соседи отталкивают; команда — результирующий вектор силы.",
        "explainability": "Каждую команду можно разложить на вклад цели, препятствия, соседей и стенки полосы. Если агент застрял, причина обычно локальный минимум или симметрия сил.",
        "steps": [
            "Посчитать притяжение к локальной цели/final slot.",
            "Добавить отталкивание от видимых препятствий и соседей в радиусе реакции.",
            "Добавить мягкое отталкивание от границ полосы.",
            "Ограничить результирующую скорость физическими лимитами агента.",
        ],
        "inputs": ["локальная цель", "видимые препятствия", "позиции соседей, если broadcast_positions включён", "коэффициенты поля"],
        "outputs": ["мгновенный вектор скорости", "причины отталкивания", "локальный setpoint"],
        "limitations": "Не гарантирует достижение цели: возможны локальные минимумы, осцилляции и затыки в воротах.",
        "reference_label": "Khatib — Real-Time Obstacle Avoidance for Manipulators and Mobile Robots, 1986",
        "reference_url": "https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf",
        "params_schema": {"goal_gain": {"default": 0.42}, "obstacle_gain": {"default": 0.18}, "neighbor_gain": {"default": 0.16}, "speed_scale": {"default": 0.74}},
    },
    "dwa": {
        "family": "standard benchmark algorithms",
        "bench_default": True,
        "label": "DWA / Dynamic Window Approach",
        "description": "Локальный поиск в пространстве допустимых скоростей: выбирается безопасная пара команд, лучшая по целевой функции rollout-а.",
        "explainability": "Для выбранной команды сохраняется локальный rollout и score: прогресс к цели, clearance, плавность и штраф за риск. Это удобно сопоставлять с физикой гусеничной машинки.",
        "steps": [
            "Построить окно скоростей, достижимых при текущих ограничениях скорости и ускорения.",
            "Сгенерировать короткие траектории-кандидаты.",
            "Отбросить кандидаты, пересекающие препятствия, соседей или границы полосы.",
            "Выбрать максимум score: progress + clearance + smoothness + objective weights.",
        ],
        "inputs": ["текущее v/omega", "лимиты скорости и ускорения", "локальная карта", "горизонт rollout"],
        "outputs": ["допустимая скорость", "локальный прогноз", "оценка безопасности"],
        "limitations": "Локальный метод; без глобального плана может выбрать тупиковую сторону обхода.",
        "reference_label": "Fox, Burgard, Thrun — The Dynamic Window Approach to Collision Avoidance, 1997",
        "reference_url": "https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf",
        "params_schema": {"horizon_s": {"default": 1.4}, "heading_samples": {"default": 13}, "heading_span_rad": {"default": 1.35}, "speed_scale": {"default": 0.78}},
    },
    "rvo_lite": {
        "family": "standard benchmark algorithms",
        "bench_default": True,
        "label": "RVO-lite / ORCA-inspired",
        "description": "Многоагентное взаимное избегание: агент начинает с preferred velocity к цели и корректирует её по соседям и препятствиям.",
        "explainability": "Команда объясняется preferred velocity плюс поправки: сосед слишком близко, препятствие слишком близко, стенка полосы. В отличие от полного ORCA здесь не решается линейная программа, поэтому это честно помечено как lite.",
        "steps": [
            "Посчитать preferred velocity к личному финальному слоту.",
            "Для каждого соседа добавить половину ответственности за расхождение траекторий.",
            "Для препятствий добавить локальную tangent/away коррекцию.",
            "Ограничить скорость и отправить команду агенту.",
        ],
        "inputs": ["позиции/скорости соседей", "локальная цель", "видимые препятствия", "радиусы агентов"],
        "outputs": ["скорректированная скорость", "кластер обхода", "локальная причина коррекции"],
        "limitations": "Это не полный ORCA solver; используется как понятный локальный baseline для роя.",
        "reference_label": "ORCA — Optimal Reciprocal Collision Avoidance",
        "reference_url": "https://gamma.cs.unc.edu/ORCA/",
        "params_schema": {"neighbor_gain": {"default": 0.34}, "obstacle_gain": {"default": 0.28}, "speed_scale": {"default": 0.74}},
    },
    "smooth_adaptive_clusters": {
        "family": "mission algorithms",
        "bench_default": True,
        "label": "Smooth Adaptive Clusters",
        "description": "Per-agent A* с плавным split/bypass/rejoin: каждый агент строит путь к своему финальному слоту, а setpoint меняется ограниченно по скорости.",
        "explainability": "Визуализация показывает raw waypoint, сглаженный setpoint, маршрут агента, финальный слот и cluster_id. Решение можно читать как: обнаружено препятствие → группа делится на upper/lower/center → каждый агент имеет индивидуальный путь → после обхода возвращается в слот.",
        "steps": [
            "Собрать локально известные препятствия из сенсоров и shared memory.",
            "Для каждого powered-агента построить A* путь к его финальному слоту.",
            "По первым точкам пути назначить кластер upper/lower/center/blocked.",
            "Сгладить setpoint и ограничить скорость перестроения.",
            "Safety filter проверяет прогноз на столкновения и границы полосы.",
        ],
        "inputs": ["финальные слоты строя", "локальная/shared карта", "целевая функция миссии", "лимиты перестроения"],
        "outputs": ["индивидуальные маршруты", "кластеры обхода", "сглаженные setpoint", "лог причин"],
        "limitations": "Эвристический mission-алгоритм; не доказывает оптимальность MAPF, но ставит завершение выше кратчайшего центра.",
        "reference_label": "A* как базовая компонента поиска пути",
        "reference_url": "https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf",
        "params_schema": {"speed_scale": {"default": 0.82}, "smooth_reconfiguration": {"default": True}, "replan_period_steps": {"default": 3}, "setpoint_smoothing_alpha": {"default": 0.70}},
    },
    "cluster_adaptive": {
        "family": "mission algorithms",
        "bench_default": True,
        "label": "cluster_adaptive / v2-style split-bypass-rejoin",
        "description": "Совместимый ключ из v2: адаптивное разделение на кластеры прохода с видимыми временными целями, затем rejoin в исходный строй.",
        "explainability": "Когда препятствие попадает в окно перед роем, алгоритм строит свободные поперечные интервалы полосы, распределяет агентов по upper/lower/center и ведёт их к временным bypass-target. После прохождения препятствия цель снова становится финальным слотом строя.",
        "steps": [
            "Получить локально известные препятствия из sonar/shared memory.",
            "Найти blockers перед центром тяжести роя.",
            "Вычесть раздутые препятствия из поперечного сечения полосы и получить free intervals.",
            "Распределить агентов по интервалам, сохраняя порядок по y, чтобы уменьшить пересечения.",
            "Построить per-agent A* к временной цели, сгладить setpoint и применить safety filter.",
            "Когда центр прошёл blocker, переключиться на rejoin к финальной матрице строя."
        ],
        "inputs": ["локальная/shared карта", "ширина полосы", "матрица/offset строя", "размеры агентов", "лимиты скорости"],
        "outputs": ["cluster_plan", "free_intervals", "temporary targets", "route previews", "g/CSV logs"],
        "limitations": "Эвристический completion-first алгоритм, не оптимальный CBS/MAPF; нужен как рабочий mission baseline и восстановление поведения v2.",
        "reference_label": "A* как базовая компонента поиска пути",
        "reference_url": "https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf",
        "params_schema": {"speed_scale": {"default": 0.82}, "smooth_reconfiguration": {"default": True}, "split_window_m": {"default": 2.5}}
    },
    "completion_astar": {
        "family": "mission algorithms",
        "bench_default": True,
        "label": "Completion-first per-agent A*",
        "description": "Каждый агент независимо планирует путь к своему финальному месту. Кратчайший центр не навязывается, чтобы не блокировать завершение миссии.",
        "explainability": "Если агент не идёт по кратчайшей линии, это видно в planned_routes: путь выбран для достижения личного слота с учётом известных препятствий, а не для минимизации центра тяжести.",
        "steps": ["Построить target slot для каждого агента.", "Запустить A* в конфигурационном пространстве агента.", "Выбрать lookahead waypoint.", "Применить safety filter."],
        "inputs": ["target slots", "локальная/shared карта", "радиус агента", "grid resolution"],
        "outputs": ["маршрут каждого агента", "команда скорости", "finish status"],
        "limitations": "Может создавать конфликты между агентами, потому что не решает полный совместный MAPF.",
        "reference_label": "Hart, Nilsson, Raphael — A*, 1968",
        "reference_url": "https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf",
        "params_schema": {"speed_scale": {"default": 0.82}},
    },
    "adaptive_clusters": {
        "family": "mission algorithms",
        "bench_default": True,
        "label": "Adaptive Clusters",
        "description": "Вариант per-agent A* с явной диагностикой split/rejoin, но без плавного setpoint v4/v6.",
        "explainability": "Показывает кластер каждого агента и причины перестроения, полезен для сравнения с smooth_adaptive_clusters.",
        "steps": ["Определить препятствие перед роем.", "Назначить кластеры по личным путям.", "Довести агентов до финальных слотов.", "Собрать строй."],
        "inputs": ["локальная/shared карта", "target slots", "formation matrix"],
        "outputs": ["cluster_plan", "routes", "commands"],
        "limitations": "Более резкие перестроения, чем у smooth_adaptive_clusters; не является CBS.",
        "reference_label": "CBS как ориентир для полного MAPF, не как текущая реализация",
        "reference_url": "https://www.sciencedirect.com/science/article/pii/S0004370214001386",
        "params_schema": {"speed_scale": {"default": 0.86}, "split_rejoin": {"default": True}},
    },
    "virtual_structure": {
        "family": "standard benchmark algorithms",
        "bench_default": False,
        "label": "Virtual Structure",
        "description": "Формация как жёсткое тело вокруг виртуального лидера/центра; совместимый baseline из ранних версий.",
        "explainability": "Команда агента = скорость виртуального центра + коррекция к фиксированному offset. Хорошо показывает, как формально задаётся строй, но плохо обтекает препятствия краями.",
        "steps": ["Создать виртуальный центр.", "Задать fixed offsets строя.", "Двигать центр к цели.", "Корректировать агентов к offset slots."],
        "inputs": ["formation offsets", "путь центра", "коэффициент коррекции"],
        "outputs": ["слоты строя", "команды агентов"],
        "limitations": "Не адаптирует формуцию к препятствию.",
        "reference_label": "Lewis & Tan — virtual structures, 1997",
        "reference_url": "https://link.springer.com/article/10.1023/A:1008814708459",
        "params_schema": {"speed_scale": {"default": 0.72}},
    },
    "apf_reactive": {
        "family": "compatibility algorithms",
        "bench_default": False,
        "label": "APF reactive compatibility",
        "description": "Совместимый ключ старой реализации APF; для обычного выбора используйте apf.",
        "explainability": "То же семейство, что APF: сумма притяжения и отталкиваний.",
        "steps": ["goal attraction", "obstacle repulsion", "neighbor repulsion"],
        "inputs": ["local target", "obstacles", "neighbors"],
        "outputs": ["velocity command"],
        "limitations": "Локальные минимумы.",
        "reference_label": "Khatib — APF, 1986",
        "reference_url": "https://khatib.stanford.edu/publications/pdfs/Khatib_1986_IJRR.pdf",
        "params_schema": {"goal_gain": {"default": 0.42}, "obstacle_gain": {"default": 0.18}},
    },
    "time_priority": {"hidden_in_selector": True, "family": "objective presets, compatibility only", "label": "Deprecated objective preset: time_priority", "description": "Сохранено только для старых конфигураций. Используйте objective_function=min_makespan + любой алгоритм.", "params_schema": {"speed_scale": {"default": 1.0}}},
    "energy_saver": {"hidden_in_selector": True, "family": "objective presets, compatibility only", "label": "Deprecated objective preset: energy_saver", "description": "Сохранено только для старых конфигураций. Используйте objective_function=min_total_energy + любой алгоритм.", "params_schema": {"speed_scale": {"default": 0.55}}},
}

_CANDIDATES = [
    {
        "key": "cbs_candidate",
        "label": "CBS / Conflict-Based Search",
        "family": "research candidates",
        "implemented": False,
        "bench_default": False,
        "description": "Двухуровневый оптимальный MAPF: верхний уровень ищет дерево конфликтов, нижний строит путь отдельного агента с ограничениями.",
        "use_for": "Будущий строгий benchmark для sum-of-costs/makespan и межагентных конфликтов.",
        "limitations": "Требует дискретизации времени и быстро дорожает при большом N.",
        "reference_label": "Sharon et al. — Conflict-Based Search for Optimal Multi-Agent Pathfinding",
        "reference_url": "https://www.sciencedirect.com/science/article/pii/S0004370214001386",
        "params_schema": {"grid_resolution_m": {"default": 0.10}, "time_step_s": {"default": 0.20}},
    },
    {
        "key": "rrt_star_candidate",
        "label": "RRT* / PRM*",
        "family": "research candidates",
        "implemented": False,
        "bench_default": False,
        "description": "Sampling-based оптимальное планирование в сложном непрерывном пространстве.",
        "use_for": "Физически проверяемые одиночные маршруты и лидер подроя в сложных полигонах.",
        "limitations": "Нужна отдельная обработка динамических препятствий и кинематики.",
        "reference_label": "Karaman & Frazzoli — Sampling-based Algorithms for Optimal Motion Planning",
        "reference_url": "https://arxiv.org/abs/1105.1186",
        "params_schema": {"samples": {"default": 1200}, "rewire_radius_m": {"default": 0.55}},
    },
    {
        "key": "dstar_lite_candidate",
        "label": "D* Lite",
        "family": "research candidates",
        "implemented": False,
        "bench_default": False,
        "description": "Инкрементальное перепланирование в неизвестной карте при обновлении стоимости рёбер.",
        "use_for": "Pop-up obstacles и сценарии, где агент знает карту только частично.",
        "limitations": "Требует явной карты стоимости и аккуратного обновления графа.",
        "reference_label": "Koenig & Likhachev — D* Lite",
        "reference_url": "https://idm-lab.org/bib/abstracts/Koen02e.html",
        "params_schema": {"grid_resolution_m": {"default": 0.10}, "unknown_cost": {"default": 1.0}},
    },
]

_OBJECTIVE_FUNCTIONS = [
    {"key": "finish_first", "label": "Сначала завершить", "description": "Основной критерий — success: все powered-агенты должны оказаться в финальных слотах и восстановить строй. Длина/энергия считаются вторично."},
    {"key": "min_makespan", "label": "Минимум времени завершения", "description": "Оптимизация по времени последнего powered-агента. Это целевая функция, а не отдельный алгоритм."},
    {"key": "shortest_centroid", "label": "Кратчайший трек центра", "description": "Сравнение с эталоном 12 м: минимизируется path length центра тяжести при сохранении success."},
    {"key": "min_total_energy", "label": "Минимум суммарной энергии", "description": "Минимизируется сумма energy_proxy агентов; полезно для сравнения с физическими логами управления."},
    {"key": "balanced", "label": "Баланс", "description": "Взвешенная сумма времени, энергии, длины центра, штрафов столкновения и ошибки строя."},
]


def make_planner(key: str) -> Planner:
    cls = _PLANNERS.get(key)
    if cls is None:
        raise KeyError(f"unknown algorithm: {key}")
    return cls()


def _decorate_meta(key: str, cls: Type[Planner], include_hidden: bool = False) -> Dict[str, object] | None:
    meta = dict(_ALGORITHM_META.get(key, {}))
    if meta.get("hidden_in_selector") and not include_hidden:
        return None
    refs = list(meta.get("references", []))
    if meta.get("reference_url"):
        refs.insert(0, {"title": str(meta.get("reference_label", "reference")), "url": str(meta["reference_url"])})
    params_schema = dict(meta.get("params_schema", {}))
    default_params = {k: (v.get("default") if isinstance(v, dict) and "default" in v else None) for k, v in params_schema.items()}
    meta.update(
        {
            "key": key,
            "label": meta.get("label") or getattr(cls, "label", key),
            "description": meta.get("description") or getattr(cls, "description", ""),
            "implemented": True,
            "family": meta.get("family", "other"),
            "group": meta.get("family", "other"),
            "standard": bool(meta.get("bench_default", False)),
            "short": meta.get("use_for", meta.get("description", "")),
            "references": refs,
            "default_params": default_params,
            "perception_dependency": meta.get("perception_dependency", "Алгоритм не получает истинную карту сам по себе: в global_debug ему передаются все активные препятствия; в local — только препятствия, видимые соником агентов и/или находящиеся в shared memory."),
            "tank_command_model": meta.get("tank_command_model", "Выход алгоритма — желаемая скорость/поворот агента. Симулятор переводит её в left/right track power, затем в приближённые команды tank.c: e/w/a/s/d и $move,rotate,...# для реального запуска."),
        }
    )
    return meta


def available_planners(include_hidden: bool = False):
    out = []
    for key, cls in _PLANNERS.items():
        item = _decorate_meta(key, cls, include_hidden=include_hidden)
        if item is not None:
            out.append(item)
    return out


def planner_catalog():
    return {
        "algorithms": available_planners(include_hidden=False),
        "all_algorithms": available_planners(include_hidden=True),
        "candidates": _CANDIDATES,
        "objective_functions": _OBJECTIVE_FUNCTIONS,
        "default": "smooth_adaptive_clusters",
    }
