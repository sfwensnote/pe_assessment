"""Simple rep counter based on phase transitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RepCounter:
    """Counts repetitions for single-user realtime sessions."""

    action_type: str
    cooldown_frames: int = 10

    def __post_init__(self) -> None:
        self.total_reps = 0
        self.seen_bottom = False
        self._cooldown = 0

    def update(self, phase: int) -> int:
        """Update with current phase and return total reps."""
        if self._cooldown > 0:
            self._cooldown -= 1

        bottom_phase, finish_phases = self._phase_rules(self.action_type)

        if phase == bottom_phase:
            self.seen_bottom = True

        if self.seen_bottom and phase in finish_phases and self._cooldown == 0:
            self.total_reps += 1
            self.seen_bottom = False
            self._cooldown = self.cooldown_frames

        return self.total_reps

    @staticmethod
    def _phase_rules(action_type: str) -> tuple[int, set[int]]:
        """Return bottom phase and valid finish phases."""
        rules = {
            "pushup": (2, {0, 4}),
            "squat": (2, {0, 4}),
            "situp": (2, {0, 4}),
        }
        return rules.get(action_type, (2, {0, 4}))
