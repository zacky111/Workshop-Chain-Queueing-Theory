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

    # inicjalizacja lambda: mała dodatnia wartość
    lambdas = {r: 1e-4 for r in range(1, R + 1)}

    err = float('inf')
    for it in range(1, max_iter + 1):
        candidate = {}
        # oblicz kandydatów new_lambda bez relaksacji
        for r in range(1, R + 1):
            S = 0.0
            lam_r = max(lambdas[r], MIN_LAMBDA)
            for i in range(1, N + 1):
                mu_i = P.SERVICE_RATES.get(i, 1.0)
                node_type = P.NODE_TYPES.get(i, 1)
                S += f_ir(lam_r, mu_i, node_type)

            # zabezpieczenie przed zerem
            if S <= 0:
                S = MIN_S

            candidate[r] = P.POPULATION[r] / S

        # relaksowana aktualizacja i ograniczenia (dodatniość)
        err = 0.0
        for r in range(1, R + 1):
            new_val = candidate[r]
            # ogranicz nową wartość do nieujemnej i minimalnej
            if new_val <= 0:
                new_val = MIN_LAMBDA
            # relaksacja
            lambdas[r] = (1.0 - RELAX_ALPHA) * lambdas[r] + RELAX_ALPHA * new_val
            # błąd (zmiana kwadratowa)
            diff = lambdas[r] - ((1.0 - RELAX_ALPHA) * lambdas[r] + RELAX_ALPHA * new_val)  # równanie daje 0, więc lepiej liczmy bez relaksacji
            # poprawione: liczymy różnicę względem "candidate" (surowy postulat)
            diff = lambdas[r] - new_val
            err += diff * diff

        if verbose and (it % 100 == 0 or it == 1):
            print(f"[SUM] iter {it:5d} err={err:.6e} lambdas=" +
                  ", ".join(f"{r}:{lambdas[r]:.6e}" for r in sorted(lambdas.keys())))

        if err < eps:
            if verbose:
                print(f"[SUM] zbieżność osiągnięta w iteracji {it}, err={err:.6e}")
            break

    # oblicz K_ir na podstawie finalnych lambdas
    K_ir = {r: {} for r in range(1, R + 1)}
    for r in range(1, R + 1):
        lam_r = max(lambdas[r], MIN_LAMBDA)
        for i in range(1, N + 1):
            mu_i = P.SERVICE_RATES.get(i, 1.0)
            node_type = P.NODE_TYPES.get(i, 1)
            K_ir[r][i] = f_ir(lam_r, mu_i, node_type)

    return lambdas, K_ir, err
