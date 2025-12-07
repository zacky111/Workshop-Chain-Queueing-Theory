from sim_engine import Simulator
import params as P
import pprint

def main():
    sim = Simulator(params=P)
    res = sim.run(until=P.SIM_TIME)
    print("=== Symulacja sieci kolejkowej ===")
    print(f"Czas symulacji: {res['time']}")
    print("\nLiczba odwiedzin / zakończeń na węzłach:")
    for nid in sorted(res['node_visits'].keys()):
        visits = res['node_visits'].get(nid, 0)
        completions = res['node_completions'].get(nid, 0)
        print(f"  Węzeł {nid}: odwiedziny={visits}, zakończenia={completions}")
    print("\nStatystyki per-węzeł:")
    pprint.pprint(res['per_node'], width=120)

if __name__ == "__main__":
    main()
