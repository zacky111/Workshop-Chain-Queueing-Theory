from sim_engine import Simulator
import params as P
import pprint
from sum_method import sum_method
from utils import compute_e_ir


def main():
    print("\n=== METODA SUM ===")

    # to be used in sum_method
    e_ir=compute_e_ir(P.ROUTES, len(P.NODE_TYPES), list(P.POPULATION.keys()))

    lambdas, K_ir, err = sum_method()

    print("Przepustowości klas (λ_r):")
    for r in lambdas:
        print(f"  Klasa {r}: λ_r = {lambdas[r]:.5f}")

    """print("Przepustowość całkowita λ = " +
          f"{sum(lambdas[r] for r in lambdas):.5f}")"""

    print("\nŚrednia liczba klientów klasy r w węźle i (K_ir):")
    for r in K_ir:
        print(f"\nKlasa {r}:")
        for i in K_ir[r]:
            print(f"  Węzeł {i}: {K_ir[r][i]:.5f}")

    print(f"\nBłąd iteracji: {err}")

if __name__ == "__main__":
    main()
