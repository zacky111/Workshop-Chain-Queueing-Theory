# sim_engine.py
import heapq
import math
import random
from collections import defaultdict

from nodes import FIFONode, ISNode
from customer import Customer
import params as P

# event types: ('end_service', time, customer, node_id)
# We'll store events as tuples (time, seq, event_type, payload...)
# seq to break ties

class Simulator:
    def __init__(self, params=P):
        self.time = 0.0
        self.event_queue = []
        self._seq = 0
        self.nodes = {}
        self.customers = []
        self.stats = {
            "node_visits": defaultdict(int),
            "node_completions": defaultdict(int)
        }
        random.seed(P.SEED)
        self._init_nodes(params)
        self._init_customers(params)

    def _init_nodes(self, params):
        for nid, ntype in params.NODE_TYPES.items():
            rate = params.SERVICE_RATES.get(nid, 1.0)
            if ntype == 1:
                self.nodes[nid] = FIFONode(nid, rate)
            elif ntype == 3:
                self.nodes[nid] = ISNode(nid, rate)
            else:
                raise ValueError("Unknown node type: %s" % ntype)

    def _init_customers(self, params):
        cid = 0
        for cls_id, count in params.POPULATION.items():
            route_template = params.ROUTES[cls_id]
            for i in range(count):
                cust = Customer(id=cid, cls=cls_id, route=list(route_template))
                self.customers.append(cust)
                cid += 1
        # place all customers initially at node 1 (they will be arrivals at t=0)
        for cust in self.customers:
            self._schedule_arrival(0.0, cust, cust.current_node())

    def _push_event(self, time, etype, *payload):
        heapq.heappush(self.event_queue, (time, self._seq, etype, payload))
        self._seq += 1

    def _schedule_arrival(self, time, customer, node_id):
        # schedule an arrival event -> handled immediately in loop
        self._push_event(time, "arrival", customer, node_id)

    def _schedule_end_service(self, time, customer, node_id):
        self._push_event(time, "end_service", customer, node_id)

    def run(self, until=P.SIM_TIME):
        while self.event_queue:
            time, seq, etype, payload = heapq.heappop(self.event_queue)
            if time > until:
                break
            self.time = time
            if etype == "arrival":
                customer, node_id = payload
                self._handle_arrival(customer, node_id)
            elif etype == "end_service":
                customer, node_id = payload
                self._handle_end_service(customer, node_id)
            else:
                raise ValueError("Unknown event type: %s" % etype)

        # finalize area stats to 'until' time
        for node in self.nodes.values():
            if hasattr(node, 'last_change_time'):
                node._update_areas(until)
        return self.collect_results(until)

    def _handle_arrival(self, customer, node_id):
        node = self.nodes[node_id]
        self.stats["node_visits"][node_id] += 1
        # schedule_start is a callback that will create an end_service event after service_time
        def schedule_start(now, cust, service_time):
            end_t = now + service_time
            self._schedule_end_service(end_t, cust, node_id)
            return end_t

        res = node.arrival(self.time, customer, schedule_start)
        # no further immediate action needed

    def _handle_end_service(self, customer, node_id):
        node = self.nodes[node_id]
        # update node completion stats
        self.stats["node_completions"][node_id] += 1

        # For IS nodes, decrement active; for FIFO node, start next if in queue
        if isinstance(node, ISNode):
            node.service_completion(self.time)
        else:
            node.service_completion(self.time, lambda now, cust, st: self._schedule_end_service(now + st, cust, node_id))

        # Now route customer to next node according to its class route and P_6_TO_8 rule
        # advance customer's route index
        # If current node is last in route, advance wraps to 0 (Customer returns to node 1)
        # But we need to implement the special probabilistic branching at node 6:
        cur_node = node_id
        customer.advance()

        # handle special branching: if the route expects 7 after 6 but P_6_TO_8 allows jumping
        next_node = customer.current_node()
        # If next_node is 7 but we may jump directly to 8 per P_6_TO_8, do that probabilistically
        if cur_node == 6:
            # If class route next node would be 7 and node 8 is after that, we can probabilistically skip 7
            if next_node == 7:
                if random.random() < P.P_6_TO_8:
                    # skip 7: set route_pos to the position of 8 in route (or if not present,
                    # we will set next_node = 8 explicitly)
                    # Attempt to find 8 in the route; if not found, send to 1
                    try:
                        pos8 = customer.route.index(8)
                        customer.route_pos = pos8
                        next_node = customer.current_node()
                    except ValueError:
                        # not found — send to node 8 anyway
                        next_node = 8
            # Otherwise follow regular route
        # safety: if next_node is None (should not), send to 1
        if next_node is None:
            next_node = 1
            customer.route_pos = 0

        # schedule arrival at next_node at current time (instant transfer)
        self._schedule_arrival(self.time, customer, next_node)

    def collect_results(self, until):
        results = {}
        results['time'] = until
        results['node_visits'] = dict(self.stats['node_visits'])
        results['node_completions'] = dict(self.stats['node_completions'])
        # per-node statistics
        per_node = {}
        for nid, node in self.nodes.items():
            if isinstance(node, FIFONode):
                avg_q = node.area_queue / until if until > 0 else 0.0
                util = node.area_busy / until
                per_node[nid] = {
                    "avg_queue_len": avg_q,
                    "utilization": util,
                    "area_queue": node.area_queue,
                    "area_busy": node.area_busy
                }
            else:
                avg_active = node.area_active / until if until > 0 else 0.0
                per_node[nid] = {
                    "avg_active_servers": avg_active,
                    "area_active": node.area_active
                }
        results['per_node'] = per_node
        return results

# local imports used conditionally — to avoid circular import issues
from nodes import ISNode, FIFONode
