from sim_engine import Simulator
import src.params as P
import pprint
from old.sum_method_old import sum_method


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
    
    print("\n=== METODA SUM ===")
    lambdas, K_ir, err = sum_method()

    print("Przepustowości klas (λ_r):")
    for r in sorted(lambdas):
        print(f"  Klasa {r}: λ_r = {lambdas[r]:.5f}")

    print("\nŚrednia liczba klientów klasy r w węźle i (K_ir):")
    for r in K_ir:
        print(f"\nKlasa {r}:")
        for i in K_ir[r]:
            print(f"  Węzeł {i}: {K_ir[r][i]:.5f}")

    print(f"\nBłąd iteracji: {err}")

if __name__ == "__main__":
    main()
