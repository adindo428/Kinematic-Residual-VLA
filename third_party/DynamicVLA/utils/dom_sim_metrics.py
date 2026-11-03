"""Observed manipulation outcomes, distinct from action-regression metrics."""
import numpy as np
from proactive_vla.phase_gate import CausalPhaseGate, PhaseGateConfig


class ManipulationMetrics:
    def __init__(self, phase_config=None, collision_force_n=1.):
        self.gate = CausalPhaseGate(PhaseGateConfig(**(phase_config or {})))
        self.collision_force_n = collision_force_n
        self.grasp = self.lift = False
        self.collision_steps = self.collision_events = 0
        self.in_contact = False
        self.max_lift_m = 0.
        self.initial_z = None

    def observe(self, t, ee, obj, velocity, previous_command, collision_force_n=None, ee_rotation_xyz=None):
        self.gate.observe(t, ee, obj, velocity, previous_command, ee_rotation_xyz=ee_rotation_xyz)
        if self.initial_z is None:
            self.initial_z = float(obj[2])
        self.max_lift_m = max(self.max_lift_m, float(obj[2])-self.initial_z)
        self.grasp |= self.gate.confirmed_relative is not None
        self.lift |= self.grasp and self.max_lift_m >= .02
        if collision_force_n is not None:
            contact = collision_force_n > self.collision_force_n
            self.collision_steps += int(contact)
            self.collision_events += int(contact and not self.in_contact)
            self.in_contact = contact

    def finish(self, termination_terms, task, success, instrumented):
        return {"grasp": bool(self.grasp), "lift": bool(self.lift),
            "place": bool(success and task in ("place", "long-horizon")), "completion": bool(success),
            "drops": int(bool(termination_terms.get("object_dropping", False))),
            "timeouts": int(bool(termination_terms.get("time_out", False))),
            "collisions": self.collision_events if instrumented else None,
            "collision_steps": self.collision_steps if instrumented else None,
            "max_lift_m": self.max_lift_m, "termination_terms": termination_terms}


def summarize_cases(cases):
    if not cases:
        raise ValueError("An empty manipulation suite cannot be summarized")
    if any(case.get("collisions") is None for case in cases):
        raise ValueError("Collision instrumentation was absent from the manipulation suite")
    return {"cases": len(cases), **{key: sum(float(c[key]) for c in cases)/len(cases)
        for key in ("grasp", "lift", "place", "completion", "drops", "collisions", "timeouts")}}
