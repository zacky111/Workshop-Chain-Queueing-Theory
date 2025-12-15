import math
import src.params as P

# ustawienia numeryczne stabilizujące


#oznaczenia
"""
lamba --> przepustowosc
mu --> współczynnik czasu obsługi

rho --> współczynnik obciążenia wykorzystania:  rho = lambda / mu
rho_i --> stopień obciążenia węzła i (war. obciążenia: >1)

K_ir_mean --> srednia liczba klientow klasy r w wezle i

K --> ogólna ilość klientów w systemie
K_r --> ilość klientów klasy r w systemie

"""

def f_ir(lambda_ir, mu_ir, node_type, K = P.K):
    """
    Funkcja f_ir(λ_ir) dla typu węzła:
      - typ 1: FIFO 1-serwer → K = ρ / (1 - ρ) przy ρ = λ/μ
      - typ 3: IS (infinite server) → K = λ / μ
    Zwracamy skończoną wartość; przy przeciążeniu zwracamy MAX_F.
    
    Dane wejściowe:
        - lambda_r: przepustowość klasy r (λ_r)
        - mu_i: szybkość obsługi węzła i (μ_i)
        - node_type: typ węzła i (1 lub 3)

    Dane wyjściowe:
        - f_ir: średnia liczba zgłoszeń klasy r w węźle i (K_ir) <-- wart. średnia
    """
    # ochronne obcięcie
    if lambda_ir <= 0:
        return 0.0

    rho_ir = lambda_ir / mu_ir

    rho_i = 1

    # Jeśli mu_i == 0: zwracamy bardzo duże wartości (przeciążenie) <-- moze powodowac bledy
    if mu_ir <= 0:
            return P.MAX_F
    
    if node_type == 3:
        # infinite-server (IS), brak kolejek
        return rho_ir  # K_ir_mean = lambda_ir / mu_ir

    if node_type == 1:
        # single-server FIFO
        # jeśli rho >= 1 → system przeciążony; zamiast inf zwracamy dużą liczbę
        if rho_ir >= 0.999999: # <-- moze powodowac bledy
            return P.MAX_F
        
        return rho_ir / (1.0 - (((K-1)/K) * rho_i)) 

    # Nieobsługiwany typ -> ostrzeżenie przez wyjątek
    raise ValueError("Nieznany typ węzła w f_ir(): %r" % (node_type,))


def sum_method(max_iter=10000, eps=1e-6, verbose=False, R=len(P.POPULATION), I = len(P.NODE_TYPES)):
    """
    Metoda SUM dla sieci wieloklasowej z relaksacją.
    Dane wejściowe:
    - max_iter - maksymalna liczba iteracji
    - eps - kryterium zbieżności (błąd)
    - verbose - czy wypisywać postęp co 100 iteracji
    - R - liczba klas klientów
        - r - konkretna klasa (1..R)
    - I - liczba węzłów sieci
        - i - konkretny węzeł (1..I)

   Dane wyjściowe:
      - lambdas: dict {r: lambda_r}
      - K_ir: dict {r: {i: K_ir}}
      - err: końcowy błąd (sum of squared diffs)
    """
    

    # inicjalizacja lambda: mała dodatnia wartość
    lambdas = {r: 1e-4 for r in range(1, R + 1)}

    err = float('inf')
    for it in range(1, max_iter + 1):
        candidate = {}
        # oblicz kandydatów new_lambda bez relaksacji
        for r in range(1, R + 1):
            S = 0.0
            lam_r = max(lambdas[r], P.MIN_LAMBDA)
            for i in range(1, I + 1):
                mu_i = P.SERVICE_RATES.get(i, 1.0)
                node_type = P.NODE_TYPES.get(i, 1)
                S += f_ir(lam_r, mu_i, node_type)

            # zabezpieczenie przed zerem
            if S <= 0:
                S = P.MIN_S

            candidate[r] = P.POPULATION[r] / S

        # relaksowana aktualizacja i ograniczenia (dodatniość)
        err = 0.0
        for r in range(1, R + 1):
            new_val = candidate[r]
            # ogranicz nową wartość do nieujemnej i minimalnej
            if new_val <= 0:
                new_val = P.MIN_LAMBDA
            # relaksacja
            lambdas[r] = (1.0 - P.RELAX_ALPHA) * lambdas[r] + P.RELAX_ALPHA * new_val
            # błąd (zmiana kwadratowa)
            diff = lambdas[r] - ((1.0 - P.RELAX_ALPHA) * lambdas[r] + P.RELAX_ALPHA * new_val)  # równanie daje 0, więc lepiej liczmy bez relaksacji
            if new_val <= 0:
                new_val = P.MIN_LAMBDA
            # relaksacja
            lambdas[r] = (1.0 - P.RELAX_ALPHA) * lambdas[r] + P.RELAX_ALPHA * new_val
            # błąd (zmiana kwadratowa)
            diff = lambdas[r] - ((1.0 - P.RELAX_ALPHA) * lambdas[r] + P.RELAX_ALPHA * new_val)  # równanie daje 0, więc lepiej liczmy bez relaksacji
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
        lam_r = max(lambdas[r], P.MIN_LAMBDA)
        for i in range(1, I + 1):
            mu_i = P.SERVICE_RATES.get(i, 1.0)
            node_type = P.NODE_TYPES.get(i, 1)
            K_ir[r][i] = f_ir(lam_r, mu_i, node_type)

    return lambdas, K_ir, err
