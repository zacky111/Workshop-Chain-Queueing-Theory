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
        # oblicz współczynniki odwiedzin e_ir (format: e_ir[i][r])
        e_ir = compute_e_ir(params.ROUTES, I, list(params.POPULATION.keys()))

    # obliczamy dla każdego węzła całościowy napływ Lambda_i
    Lambda = {i: 0.0 for i in range(1, I + 1)}
    for i in range(1, I + 1):
        for r in classes:
            visit = e_ir.get(i, {}).get(r, 0.0)
            lam_r = max(lambdas.get(r, 0.0), params.MIN_S)
            Lambda[i] += lam_r * visit

    # teraz liczymy f[i][r] według typu węzła
    for i in range(1, I + 1):
        mu_i = params.SERVICE_RATES.get(i, None)
        node_type = params.NODE_TYPES[i]
        for r in classes:
            visit = e_ir.get(i, {}).get(r, 0.0)
            lam_ir = max(lambdas.get(r, 0.0) * visit, params.MIN_S)
            if node_type == 3:  # IS
                if mu_i is None or mu_i <= 0:
                    f[i][r] = params.MAX_F * (lam_ir / max(Lambda[i], params.MIN_S))
                else:
                    f[i][r] = lam_ir / mu_i
            elif node_type in [1, 2, 4]:  # FIFO ~ M/M/1
                if mu_i is None or mu_i <= 0:
                    f[i][r] = params.MAX_F * (lam_ir / max(Lambda[i], params.MIN_S))
                else:
                    rho = Lambda[i] / mu_i
                    if rho >= 1.0:
                        f[i][r] = params.MAX_F * (lam_ir / max(Lambda[i], params.MIN_S))
                    else:
                        L_i = rho / (1.0 - rho)
                        f[i][r] = (lam_ir / max(Lambda[i], params.MIN_S)) * L_i
            else:
                raise ValueError(f"Nieobsługiwany typ węzła: {node_type}")

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

            rho_ir=visit_ratios[r][i] / max(mu_i, MIN_S)
            
            rho_i = 0.0
            for _ in range(r):
                rho_i += visit_ratios[r][i] / max(mu_i, MIN_S)
            
            print(f"rho_i: {rho_i}")
            return (rho_ir) * (1/ max(((1 - ((K-1) / K)) * rho_i, MIN_S)))
        
        else:
            raise ValueError(f"Nieobsługiwana ilość serwerów w węźle: {node_m_servers[i]}")

    # ===== IS (typ 3) =====
    elif node_type[i] == 3:

        return visit_ratios[r][i] / mu_i

    else:
        raise ValueError(f"Nieobsługiwany typ węzła: {node_type}")


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



# === GŁÓWNY ALGORYTM METODY SUM ===
# inicjalizacja
f_ir_val = {
    (i, r): 0.00001
    for i in params.NODE_TYPES
    for r in params.POPULATION
}

errs = []  # zapisujemy err (diff) w każdej iteracji

for iteration in range(params.MAX_ITER):
    # --- krok 1: liczymy mianowniki ---
    denom = {}
    for r in params.POPULATION:
        denom[r] = sum(
            params.VISIT_RATIOS[r][i] * f_ir_val[(i, r)]
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
    diff = math.sqrt(sum((new_f_ir[k] - f_ir_val[k])**2 for k in f_ir_val))
    print(f"Iteracja {iteration + 1}: błąd = {diff}")
    errs.append(diff)  # zapisujemy err tej iteracji
    f_ir_val = new_f_ir

    if diff < params.EPS:
        print("ended on iteration", iteration + 1)
        break

# przygotowujemy wynik w formacie {i: {r: f_ir}}
final_f_ir = {i: {r: 0.0 for r in params.POPULATION} for i in params.NODE_TYPES}
for (i, r), val in f_ir_val.items():
    final_f_ir[i][r] = val

# obliczamy średni błąd z zapisanych iteracji
if errs:
    N = min(100, len(errs))  # weź ostatnie N iteracji (burn-in)
    err_mean = float(np.mean(errs[-N:]))
    err_median = float(np.median(errs))
    err_max = float(np.max(errs))
else:
    err_mean = err_median = err_max = 0.0







