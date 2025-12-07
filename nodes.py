# nodes.py
import collections
import math
import random
from typing import Deque

class FIFONode:
    """Single-server FIFO queue."""
    def __init__(self, node_id, service_rate):
        self.node_id = node_id
        self.rate = service_rate
        self.queue: Deque = collections.deque()
        self.busy = False
        self.busy_count = 0   # 0 or 1
        # Statystyka czasowa
        self.last_change_time = 0.0
        self.area_queue = 0.0
        self.area_busy = 0.0
        self.queue_length = 0

    def arrival(self, sim_time, customer, schedule_start):
        """customer arrives: either starts service or queues"""
        # update time-weighted stats
        self._update_areas(sim_time)
        if not self.busy:
            # start service immediately
            self.busy = True
            self.busy_count = 1
            self._update_areas(sim_time)
            return schedule_start(sim_time, customer, self.service_time())
        else:
            self.queue.append(customer)
            self.queue_length = len(self.queue)
            return None

    def service_time(self):
        # exponential
        if self.rate <= 0:
            return float('inf')
        return random.expovariate(self.rate)

    def service_completion(self, sim_time, schedule_start):
        """called when a service ends: start next if queue nonempty"""
        self._update_areas(sim_time)
        if self.queue:
            next_cust = self.queue.popleft()
            self.queue_length = len(self.queue)
            self._update_areas(sim_time)
            return schedule_start(sim_time, next_cust, self.service_time())
        else:
            self.busy = False
            self.busy_count = 0
            self._update_areas(sim_time)
            return None

    def _update_areas(self, sim_time):
        dt = sim_time - self.last_change_time
        if dt < 0:
            dt = 0
        self.area_queue += self.queue_length * dt
        self.area_busy += self.busy_count * dt
        self.last_change_time = sim_time

class ISNode:
    """Infinite-server node: każdy przychodzący klient od razu zaczyna obsługę"""
    def __init__(self, node_id, service_rate):
        self.node_id = node_id
        self.rate = service_rate
        self.active = 0
        self.last_change_time = 0.0
        self.area_active = 0.0

    def arrival(self, sim_time, customer, schedule_start):
        self._update_areas(sim_time)
        self.active += 1
        self._update_areas(sim_time)
        return schedule_start(sim_time, customer, self.service_time())

    def service_completion(self, sim_time):
        self._update_areas(sim_time)
        if self.active > 0:
            self.active -= 1
        self._update_areas(sim_time)

    def service_time(self):
        if self.rate <= 0:
            return float('inf')
        return random.expovariate(self.rate)
    
    def _update_areas(self, sim_time):
        dt = sim_time - self.last_change_time
        if dt < 0:
            dt = 0
        self.area_active += self.active * dt
        self.last_change_time = sim_time

