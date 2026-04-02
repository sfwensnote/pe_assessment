"""Simple rep counter based on phase transitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RepCounter:
    """Counts repetitions for single-user realtime sessions."""

    action_type: str
    cooldown_frames: int = 10

    def __post_init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self.total_reps = 0
        self.seen_bottom = False
        self._cooldown = 0

    def _tick(self) -> None:
        if self._cooldown > 0:
            self._cooldown -= 1

    def set_action_type(self, action_type: str, reset: bool = False) -> None:
        """Update target action and optionally reset state for a new motion pattern."""
        if action_type == self.action_type:
            return

        self.action_type = action_type
        if reset:
            self._reset_state()

    def update(self, phase: int) -> int:
        """Update with current phase and return total reps."""
        self._tick()

        bottom_phase, finish_phases = self._phase_rules(self.action_type)

        if phase == bottom_phase:
            self.seen_bottom = True

        if self.seen_bottom and phase in finish_phases and self._cooldown == 0:
            self.total_reps += 1
            self.seen_bottom = False
            self._cooldown = self.cooldown_frames

        return self.total_reps

    def update_from_signal(self, signal: float, recent_low: float | None = None) -> int:
        """Count reps from an action-specific scalar motion signal."""
        thresholds = self._signal_rules(self.action_type)
        if thresholds is None:
            return self.total_reps

        self._tick()
        low, high = thresholds

        low_probe = signal if recent_low is None else recent_low
        if low_probe <= low:
            self.seen_bottom = True

        if self.seen_bottom and signal >= high and self._cooldown == 0:
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

    @staticmethod
    def _signal_rules(action_type: str) -> tuple[float, float] | None:
        """Return low/high thresholds for angle-based rep counting."""
        rules = {
            "pushup": (105.0, 155.0),
            "pullup": (105.0, 155.0),
            "squat": (110.0, 155.0),
        }
        return rules.get(action_type)
