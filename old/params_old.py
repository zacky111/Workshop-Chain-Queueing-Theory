# params.py
# Parametry sieci i klasy klientów.

from src.utils import compute_visit_ratios

# Liczba klientów (zamknięta populacja) przypisana do każdej klasy:
# K_r = POPULATION[R] # liczba klientów klasy r
POPULATION = {
    1: 5,   # klasa 1 — uszkodzenia elektryczne
    2: 4,   # klasa 2 — uszkodzenia mechaniczne
    3: 3,   # klasa 3 — mieszane
    4: 2    # klasa 4 — uproszczone zlecenia
}



# Całkowita liczba klientów w systemie
K = sum(POPULATION.values())


# Węzły (numery 1..8) i ich typy:
# typ 1 -> FIFO single-server (kolejka, 1 serwer)
# typ 3 -> IS infinite-server (brak kolejek, każdy od razu zaczyna obsługę)
NODE_TYPES = {
    1: 1,  # Przyjmowanie zgłoszenia (FIFO)
    2: 1,  # Dział elektryczny (FIFO)
    3: 1,  # Dział mechaniczny (FIFO)
    4: 3,  # Testy elektryczne - automatyczne (IS)
    5: 3,  # Testy mechaniczne - automatyczne (IS)
    6: 1,  # Wycena/dokumentacja (FIFO)
    7: 3,  # Obsługa klienta - wydanie urządzenia (IS)
    8: 3   # Stała eksploatacja (IS)
}

# Liczba serwerów w węźle (używane tylko dla typów IS, gdzie jest więcej niż 1 serwer)
# odpowiada za parametr m w rozkładzie M/M/m
NODE_M_SERVERS = {
    1: 1,  # Przyjmowanie zgłoszenia
    2: 1,  # Dział elektryczny
    3: 1,  # Dział mechaniczny
    4: 3,  # Testy elektryczne - automatyczne
    5: 3,  # Testy mechaniczne - automatyczne
    6: 1,  # Wycena/dokumentacja
    7: 2,  # Obsługa klienta - wydanie urządzenia
    8: 10  # Stała eksploatacja
}

NODE_NAMES = {
    1: "Przyjmowanie zgłoszenia",
    2: "Dział elektryczny",
    3: "Dział mechaniczny",
    4: "Testy elektryczne - automatyczne",
    5: "Testy mechaniczne - automatyczne",
    6: "Wycena/dokumentacja",
    7: "Obsługa klienta - wydanie urządzenia",
    8: "Stała eksploatacja"
}

# mu_i - szybkości obsługi węzłów (1 / średni czas obsługi [min])
# przez to że czasy obługi nie są zależne od klasy, to mu_i == mu_ir
# Dla rozkładu wykładniczego: time ~ Exp(rate)
SERVICE_RATES = {
    1: 1/3.0,   # node 1: średnio 3.0 czasu na obsługę
    2: 1/20.0,  # node 2: np. naprawa elektryczna średnio 20.0
    3: 1/25.0,  # node 3: naprawa mechaniczna średnio 25.0
    4: 1/5.0,   # node 4: testy elektryczne automatyczne średnio 5.0
    5: 1/6.0,   # node 5: testy mechaniczne automatyczne średnio 6.0
    6: 1/4.0,   # node 6: wycena/dokumentacja średnio 4.0
    7: 1/2.0,   # node 7: obsługa klienta średnio 2.0
    8: 1/10.0   # node 8: stała eksploatacja średnio 10.0
}

# Trasy klas (lista węzłów do odwiedzenia; po dojściu do ostatniego węzła wraca do 1).
# ROUTING[class][node] = [(next_node, probability), ...]
ROUTES = {
    1: {   # klasa 1 — elektryczne
        1: [(2, 1.0)],                 # 1 → 2
        2: [(4, 0.7), (6, 0.3)],       # 2 → 4
        4: [(6, 1.0)],                 # 4 → 6
        6: [(7, 0.7), (8, 0.3)],       # rozgałęzienie 6 → 7 lub 6 → 8
        7: [(8, 1.0)],                 # 7 → 8
        8: [(1, 1.0)],                 # cykl zamknięty
    },

    2: {   # klasa 2 — mechaniczne
        1: [(3, 1.0)],
        3: [(5, 0.7), (6, 0.3)],
        5: [(6, 1.0)],
        6: [(7, 0.7), (8, 0.3)],
        7: [(8, 1.0)],
        8: [(1, 1.0)],
    },

    3: {   # klasa 3 — mieszane
        1: [(2, 1.0)],
        2: [(3, 1.0)],
        3: [(4, 1.0)],
        4: [(5, 1.0)],
        5: [(6, 1.0)],
        6: [(7, 0.7), (8, 0.3)],
        7: [(8, 1.0)],
        8: [(1, 1.0)],
    },

    4: {   # klasa 4 — uproszczone
        1: [(6, 1.0)],
        6: [(7, 0.7), (8, 0.3)],  # uproszczone: brak 6→8
        7: [(8, 1.0)],
        8: [(1, 1.0)],
    },
}

NUM_NODES = len(NODE_TYPES)

# Współczynniki liczby wizyt e_ir dla każdej klasy r i węzła i
# VISIT_RATIOS[r][i] == e_ir
VISIT_RATIOS = {
    r: compute_visit_ratios(ROUTES[r], NUM_NODES)
    for r in ROUTES
}

for r, visits in VISIT_RATIOS.items():
    assert abs(visits[1] - 1.0) < 1e-8, f"e_1{r} != 1"

for r, visits in VISIT_RATIOS.items():
    for i, e_ir in visits.items():
        assert e_ir >= 0, "ujemne visit ratio"

# Print obliczonych e_ir
"""
print("Współczynniki liczby wizyt e_ir:")
for r in VISIT_RATIOS:
    print(f"Klasa {r}:")
    for i in VISIT_RATIOS[r]:
        print(f"  Węzeł {i}: e_ir = {VISIT_RATIOS[r][i]:.5f}")
"""

# Obliczenie z warunku początkowego maksymalnych przepustowości klas
MAX_LAMBDA = {}
for r, visits in VISIT_RATIOS.items():
    limits = []
    for i, e_ir in visits.items():
        if e_ir > 0:
            limits.append(SERVICE_RATES[i] * NODE_M_SERVERS[i] / e_ir)
    MAX_LAMBDA[r] = min(limits)

# Czas symulacji [min]
SIM_TIME = 20000.0 

# Ziarno generatora losowego (dla powtarzalności)
SEED = 42

## metoda sum parametry
MAX_F = 1e12        # zastępuje "inf" dla f_ir przy przeciążeniu
MIN_S = 1e-12       # zabezpieczenie przed dzieleniem przez zero
MIN_LAMBDA = 1e-12  # minimalna dopuszczalna wartość lambda
RELAX_ALPHA = 0.4   # współczynnik relaksacji (0 < alpha <= 1)

MAX_ITER = 1000    # maksymalna liczba iteracji
EPS = 1e-5         # tolerancja zbieżności