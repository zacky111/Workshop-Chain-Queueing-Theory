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