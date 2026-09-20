"""
===============================================================================
THE ZIA PIPELINE                                           [CORE - read zia.py]
===============================================================================

Zomato's conversational architecture, in five stages:

    function_classifier.py   [1] which data does this turn need?
    data_functions.py        [2] fetch only that, narrate it as prose
    zia.py                   [3] generate reply + decide action, in parallel
    action_decisioning.py    [3b] classify an action, then VERIFY it
    escalation_policy.py     [4] check escalation tags against real data

Start with zia.py - it shows the whole turn in one place.
===============================================================================
"""

from .action_decisioning import ActionDecision, decide_action, verify_action
from .escalation_policy import EscalationVerdict, verify_escalation
from .function_classifier import FunctionPlan, classify_functions
from .zia import TurnResult, Zia

__all__ = [
    "Zia", "TurnResult",
    "ActionDecision", "decide_action", "verify_action",
    "EscalationVerdict", "verify_escalation",
    "FunctionPlan", "classify_functions",
]
