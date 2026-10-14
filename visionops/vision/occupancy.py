from dataclasses import dataclass, field


@dataclass
class OccupancyState:
    initial: int = 0
    entries: int = 0
    exits: int = 0
    inconsistencies: list[str] = field(default_factory=list)

    @property
    def current(self) -> int:
        value = self.initial + self.entries - self.exits
        if value < 0:
            return 0
        return value

    def record_entry(self) -> None:
        self.entries += 1

    def record_exit(self) -> None:
        self.exits += 1
        if self.initial + self.entries - self.exits < 0:
            self.inconsistencies.append("Occupancy went negative; reconciliation needed.")

    def reconcile(self, occupancy: int) -> None:
        self.initial = max(occupancy, 0)
        self.entries = 0
        self.exits = 0

