from sim_engine import Simulator
import src.params as P
import pprint
from sum_method import sum_method
from src.utils import compute_e_ir


def main():
    print("\n=== METODA SUM ===")

    lambdas, K_ir, err, err_mean = sum_method()

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

    print(f"\nBłąd ostatniej iteracji: {err}")
    print(f"Średni błąd z wszystkich iteracji: {err_mean}")


if __name__ == "__main__":
    main()
