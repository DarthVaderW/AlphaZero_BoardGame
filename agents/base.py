# -*- coding: utf-8 -*-
"""
Base agent interface for consistency across agent implementations.
"""
from typing import Any

class BaseAgent:
    def reset(self) -> None:
        raise NotImplementedError

    def to_player(self) -> Any:
        """Return the underlying player object that interacts with Game."""
        raise NotImplementedError