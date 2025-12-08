import math
import params as P

# ustawienia numeryczne stabilizujące
MAX_F = 1e12        # zastępuje "inf" dla f_ir przy przeciążeniu
MIN_S = 1e-12       # zabezpieczenie przed dzieleniem przez zero
MIN_LAMBDA = 1e-12  # minimalna dopuszczalna wartość lambda
RELAX_ALPHA = 0.4   # współczynnik relaksacji (0 < alpha <= 1)


def f_ir(lambda_r, mu_i, node_type):
    """
    Funkcja f_ir(λ_ir) dla typu węzła:
      - typ 1: FIFO 1-serwer → K = ρ / (1 - ρ) przy ρ = λ/μ
      - typ 3: IS (infinite server) → K = λ / μ
    Zwracamy skończoną wartość; przy przeciążeniu zwracamy MAX_F.
    """
    # ochronne obcięcie
    if lambda_r <= 0:
        return 0.0

    if node_type == 3:
        # infinite-server: liniowo zależne od λ
        # Jeśli mu_i == 0: zwracamy bardzo duże
        if mu_i <= 0:
            return MAX_F
        return lambda_r / mu_i

    if node_type == 1:
        # single-server FIFO
        if mu_i <= 0:
            return MAX_F
        rho = lambda_r / mu_i
        # jeśli rho >= 1 → system przeciążony; zamiast inf zwracamy dużą liczbę
        if rho >= 0.999999:
            return MAX_F
        # standardowa formuła K = rho / (1 - rho)
        return rho / (1.0 - rho)

    # Nieobsługiwany typ -> ostrzeżenie przez wyjątek
    raise ValueError("Nieznany typ węzła w f_ir(): %r" % (node_type,))


def sum_method(max_iter=10000, eps=1e-6, verbose=False):
    """
    Metoda SUM dla sieci wieloklasowej z relaksacją.
    Zwraca (lambdas, K_ir, err) gdzie:
      - lambdas: dict {r: lambda_r}
      - K_ir: dict {r: {i: K_ir}}
      - err: końcowy błąd (sum of squared diffs)
    """

    R = len(P.POPULATION)
    N = len(P.NODE_TYPES)


    #return lambdas, K_ir, err
