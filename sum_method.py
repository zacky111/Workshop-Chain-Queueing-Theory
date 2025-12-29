import math
from typing import Dict

import numpy as np

from src import params
from src.utils import compute_e_ir

def f_ir(lambdas: Dict[int, float], e_ir: Dict[int, Dict[int, float]]):
    """
    Oblicza f_ir(i,r) dla wszystkich węzłów i i klas r przy zadanych
    współczynnikach przepływu lambdas (słownik r->lambda_r) oraz e_ir
    (słownik i->{r: e_ir}). Zwraca słownik f_ir w kształcie {i: {r: f_ir}}.

    Przyjęte aproksymacje:
      - węzły typu 3 (IS / "infinite-server"): f_ir = lambda_r * e_ir / mu_i
      - węzły typu 1 (FIFO single-server, M/M/1 aprox.): liczymy
        Lambda_i = sum_r lambda_r * e_ir,  rho = Lambda_i / mu_i
        jeśli rho < 1: L_i = rho / (1 - rho) => f_ir = (lambda_r*e_ir / Lambda_i) * L_i
        jeśli rho >= 1: f_ir ustawiamy na params.MAX_F proporcjonalnie do udziału klasy
    Bezpieczne zachowania:
      - małe Lambda_i ~ 0 => f_ir = 0
      - lambda_r poniżej MIN_LAMBDA będzie zastąpione MIN_LAMBDA aby uniknąć dzielenia przez zero
    """
    I = len(params.NODE_TYPES)
    classes = sorted(lambdas.keys())
    # init f_ir
    f = {i: {r: 0.0 for r in classes} for i in range(1, I + 1)}

    # przygotuj e_ir - jeśli nie podano, oblicz z ROUTES
    if e_ir is None:
        e_ir = compute_e_ir(params.ROUTES, I, classes)

    # obliczamy dla każdego węzła całościowy napływ Lambda_i
    for i in range(1, I + 1):
        mu = params.SERVICE_RATES.get(i, None)
        node_type = params.NODE_TYPES.get(i, 1)

        # składnik napływu klasy r do węzła i: lambda_r * e_ir(i,r)
        arrivals = {}
        Lambda_i = 0.0
        for r in classes:
            lam_r = max(lambdas.get(r, 0.0), params.MIN_LAMBDA)
            e = e_ir.get(i, {}).get(r, 0.0)
            arrivals[r] = lam_r * e
            Lambda_i += arrivals[r]

        # jeśli praktycznie brak napływu -> wszystkie f_ir = 0
        if Lambda_i < params.MIN_S:
            for r in classes:
                f[i][r] = 0.0
            continue

        # obsługa węzła wg typu
        if node_type == 3:
            # IS / M/M/∞ (średnia liczba = arrival_rate * mean_service_time)
            # mean service time = 1 / mu
            if mu is None or mu <= 0:
                # brak poprawnego mu -> traktujemy jako bardzo duże obciążenie
                for r in classes:
                    f[i][r] = params.MAX_F * (arrivals[r] / Lambda_i if Lambda_i > 0 else 0.0)
            else:
                for r in classes:
                    f[i][r] = arrivals[r] / mu  # oczekiwana liczba klas r w i
        else:
            # typ 1 (FIFO single-server) -> M/M/1 aproksymacja
            if mu is None or mu <= 0:
                for r in classes:
                    f[i][r] = params.MAX_F * (arrivals[r] / Lambda_i if Lambda_i > 0 else 0.0)
            else:
                rho = Lambda_i / mu
                if rho >= 1.0:
                    # przeciążenie -> przybliżamy bardzo dużą liczbą
                    for r in classes:
                        f[i][r] = params.MAX_F * (arrivals[r] / Lambda_i)
                else:
                    L_i = rho / (1.0 - rho)  # średnia liczba w M/M/1
                    for r in classes:
                        share = arrivals[r] / Lambda_i if Lambda_i > 0 else 0.0
                        f[i][r] = share * L_i

    return f


def fix_ir(i,
           r,
           lambda_ir,
           mu_i,
           node_type=params.NODE_TYPES,
           node_m_servers=params.NODE_M_SERVERS,
           visit_ratios=params.VISIT_RATIOS,
           K=params.K,
           MAX_F=1e12,
           MIN_S=1e-12):
    """
    Funkcja FIX_i^r z metody SUM (bez jawnego rho).

    i – numer węzła
    r – numer klasy
    lambda_ir – intensywność klasy r w węźle i (z poprzedniej iteracji)
    lambda_i_all – suma intensywności wszystkich klas w węźle i
    mu_i – szybkość obsługi węzła i
    node_type – typ węzła
    """


    # ===== FIFO (typ 1) =====
    if node_type[i] in [1, 2, 4]:

        if  node_m_servers[i] == 1:

            return (visit_ratios[r][i] / mu_i) * (1/(1 - (K-1) / K) * (lambda_ir / mu_i))
        
        else:
            raise ValueError(f"Nieobsługiwana ilość serwerów w węźle: {node_m_servers[i]}")

    # ===== IS (typ 3) =====
    elif node_type[i] == 3:

        return visit_ratios[r][i] / mu_i

    else:
        raise ValueError(f"Nieobsługiwany typ węzła: {node_type}")


# inicjalizacja
f_ir = {
    (i, r): 1.0
    for i in params.NODE_TYPES
    for r in params.POPULATION
}

errs = []  # zapisujemy err (diff) w każdej iteracji

for iteration in range(params.MAX_ITER):
    # --- krok 1: liczymy mianowniki ---
    denom = {}
    for r in params.POPULATION:
        denom[r] = sum(
            params.VISIT_RATIOS[r][i] * f_ir[(i, r)]
            for i in params.NODE_TYPES
        )

    # --- krok 2: liczymy lambda_ir ---
    lambda_ir = {}
    lambda_i_all = {i: 0.0 for i in params.NODE_TYPES}

    for r in params.POPULATION:
        for i in params.NODE_TYPES:
            val = (params.POPULATION[r] * params.VISIT_RATIOS[r][i]) / max(denom[r], params.MIN_S)
            lambda_ir[(i, r)] = val
            lambda_i_all[i] += val

    # --- krok 3: FIX ---
    new_f_ir = {}
    for r in params.POPULATION:
        for i in params.NODE_TYPES:
            new_f_ir[(i, r)] = fix_ir(
                i=i,
                r=r,
                lambda_ir=lambda_ir[(i, r)],
                mu_i=params.SERVICE_RATES.get(i, None),
            )

    # --- krok 4: sprawdzamy zbieżność ---
    diff = max(abs(new_f_ir[k] - f_ir[k]) for k in f_ir)
    errs.append(diff)  # zapisujemy err tej iteracji
    f_ir = new_f_ir

    if diff < params.EPS:
        break

# przygotowujemy wynik w formacie {i: {r: f_ir}}
final_f_ir = {i: {r: 0.0 for r in params.POPULATION} for i in params.NODE_TYPES}
for (i, r), val in f_ir.items():
    final_f_ir[i][r] = val

# obliczamy średni błąd z zapisanych iteracji
err_mean = float(np.mean(errs)) if errs else 0.0

def sum_method():
    """
    Metoda SUM do wyznaczania przepustowości klas λ_r oraz średniej liczby klientów klasy r w węźle i (K_ir).
    Zwraca krotkę (lambdas, K_ir, err, err_mean), gdzie:
      - lambdas to słownik r->λ_r
      - K_ir to słownik r->{i: K_ir}
      - err to błąd ostatniej iteracji
      - err_mean to średni błąd po iteracjach
    """
    # końcowe lambdas
    lambdas = {}
    for r in params.POPULATION:
        denom = sum(
            params.VISIT_RATIOS[r][i] * final_f_ir[i][r]
            for i in params.NODE_TYPES
        )
        lambdas[r] = params.POPULATION[r] / max(denom, params.MIN_S)

    # końcowe K_ir
    K_ir = {}
    for r in params.POPULATION:
        K_ir[r] = {}
        for i in params.NODE_TYPES:
            K_ir[r][i] = lambdas[r] * params.VISIT_RATIOS[r][i] * final_f_ir[i][r]

    # błąd ostatniej iteracji (ostatni zapis w errs)
    err = float(errs[-1]) if errs else 0.0

    return lambdas, K_ir, err, err_mean




