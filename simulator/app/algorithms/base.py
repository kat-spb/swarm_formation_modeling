from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from ..domain import PlannerCommand


class Planner(ABC):
    key = "base"
    label = "Base planner"
    description = ""

    @abstractmethod
    def plan(self, sim) -> Dict[str, PlannerCommand]:
        raise NotImplementedError
