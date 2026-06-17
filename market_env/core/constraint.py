"""Constraint interfaces for physical, budget, policy, and safety limits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class ConstraintResult:
    name: str
    satisfied: bool
    violation: float = 0.0
    penalty: float = 0.0
    metadata: Dict[str, Any] | None = None


class Constraint(Protocol):
    name: str

    def evaluate(self, state: Dict[str, Any], action: Dict[str, Any]) -> ConstraintResult:
        ...
