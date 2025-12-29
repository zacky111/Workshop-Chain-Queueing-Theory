## e_ir computation
import numpy as np
import pandas as pd

def compute_e_ir(ROUTES, I, classes):
    """
    Oblicza liczby wizyt e_ir dla każdej klasy r i stacji i.
    ROUTES: słownik tras
    I: liczba stacji
    classes: lista klas np. [1,2,3,4]
    """
    e_ir = {i: {r: 0.0 for r in classes} for i in range(1, I+1)}

    for r in classes:
        # budujemy macierz P^(r)
        P = np.zeros((I, I))
        routes_r = ROUTES.get(r, {})
        for j in range(1, I+1):
            if j in routes_r:
                for nxt, p in routes_r[j]:
                    P[j-1, nxt-1] += p

        # układ: e^T = e^T P, czyli (P^T - I)^T e = 0
        A = (P.T - np.eye(I))
        # zastępujemy ostatni wiersz warunkiem normalizacji
        A[-1, :] = np.ones(I)
        b = np.zeros(I)
        b[-1] = 1.0

        e = np.linalg.lstsq(A, b, rcond=None)[0]

        for i in range(I):
            e_ir[i+1][r] = float(e[i])

    return e_ir

"""
e_ir=compute_e_ir(ROUTES, len(NODE_TYPES), list(POPULATION.keys()))
df = pd.DataFrame(e_ir).T
df = df.map(lambda x: 0 if abs(x) < 1e-10 else x)
print(df.round(3))
"""


def compute_visit_ratios(routes, num_nodes, entry_node=1, tol=1e-12, max_iter=10000):
    """
    Liczy e_ir dla jednej klasy w sieci ZAMKNIĘTEJ
    jako wektor własny macierzy przejść (power method).
    """

    # e^(0)
    e = {i: 1.0 for i in range(1, num_nodes + 1)}

    for _ in range(max_iter):
        new_e = {i: 0.0 for i in range(1, num_nodes + 1)}

        # e^{k+1} = e^k P
        for i, transitions in routes.items():
            for j, p in transitions:
                new_e[j] += e[i] * p

        # normalizacja: e_entry = 1
        factor = new_e[entry_node]
        if factor == 0:
            raise RuntimeError("entry node unreachable")

        for i in new_e:
            new_e[i] /= factor

        # test zbieżności
        diff = max(abs(new_e[i] - e[i]) for i in e)
        e = new_e

        if diff < tol:
            return e

    raise RuntimeError("compute_visit_ratios: brak zbieżności")
