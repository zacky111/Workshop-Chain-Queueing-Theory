# customer.py
from dataclasses import dataclass, field

@dataclass
class Customer:
    id: int
    cls: int                  # klasa klienta (1..4)
    route: list               # sekwencja węzłów do odwiedzenia (list of node ids)
    route_pos: int = 0        # indeks aktualnego węzła w trasie
    stats: dict = field(default_factory=lambda: {"visits": 0})

    def current_node(self):
        if self.route_pos < len(self.route):
            return self.route[self.route_pos]
        return None

    def advance(self):
        self.route_pos += 1
        if self.route_pos >= len(self.route):
            # wraca do początku (zamknięta pętla)
            self.route_pos = 0
