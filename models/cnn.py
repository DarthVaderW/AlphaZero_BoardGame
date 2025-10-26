# -*- coding: utf-8 -*-
"""
CNN-based policy-value model wrapper.
Bridges to existing policy_value_net_pytorch.PolicyValueNet for compatibility.
"""
from typing import Sequence, Tuple
import numpy as np

# Reuse existing implementation
from policy_value_net_pytorch import PolicyValueNet as _PolicyValueNet

class PolicyValueNet(_PolicyValueNet):
    """Alias class to keep a clean models API while reusing the current impl."""
    pass

__all__ = ["PolicyValueNet"]