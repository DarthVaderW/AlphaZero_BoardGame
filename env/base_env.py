# -*- coding: utf-8 -*-
"""
BaseEnv defines a minimal interface for board-game environments.
"""
from typing import Tuple, List, Any
import numpy as np

class BaseEnv:
    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        raise NotImplementedError

    def legal_actions(self, state: Any = None) -> List[int]:
        raise NotImplementedError

    def current_player(self, state: Any = None) -> int:
        raise NotImplementedError

    def is_terminal(self, state: Any = None) -> bool:
        raise NotImplementedError

    def result(self, state: Any = None) -> int:
        """Return +1/-1/0 at terminal, from the perspective of the last mover."""
        raise NotImplementedError

    def observe(self, state: Any = None) -> np.ndarray:
        raise NotImplementedError

    def render(self) -> None:
        pass