import numpy as np
import math
from scipy.optimize import nnls
import json
import os

np.set_printoptions(precision=3, suppress=True)


class SummationMethod:
    def __init__(self, config_path: str = None) -> None:
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._set_defaults()

        self.calculate_E()

    # ------------------------------------------------------------------
    # DEFAULTS / CONFIG
    # ------------------------------------------------------------------

    def _set_defaults(self):
        self.r = 4
        self.n = 8
        self.service_type = np.array([1, 1, 1, 3, 3, 1, 3, 3])
        self.m = np.array([10, 10, 1, 10, 1, 1, 1, 1])
        self.mi = np.ones((self.n, self.r))
        self.p = np.zeros((self.r, self.n, self.n))
        self.K = np.array([2, 4, 2, 3], dtype=int)
        self.epsilon = 1e-7
        self.num_of_iterations = 200

        self.e = np.zeros((self.n, self.r))
        self.lambdas = np.array([self.epsilon] * self.r)
        self.K_ir = np.zeros((self.n, self.r))
        self.T_ir = np.zeros((self.n, self.r))

    def load_config(self, filepath: str):
        with open(filepath, "r") as f:
            cfg = json.load(f)

        self.r = cfg["r"]
        self.n = cfg["n"]
        self.service_type = np.array(cfg["service_type"])
        self.m = np.array(cfg["m"])
        self.mi = np.array(cfg["mi"])
        self.p = np.array(cfg["p"])
        self.K = np.array(cfg["K"], dtype=int)
        self.epsilon = cfg["epsilon"]
        self.num_of_iterations = cfg["num_of_iterations"]

        self.e = np.zeros((self.n, self.r))
        self.lambdas = np.array([self.epsilon] * self.r)
        self.K_ir = np.zeros((self.n, self.r))
        self.T_ir = np.zeros((self.n, self.r))

    def save_config(self, filepath: str):
        config = {
            'r': int(self.r),
            'n': int(self.n),
            'service_type': self.service_type.tolist(),
            'm': self.m.tolist(),
            'mi': self.mi.tolist(),
            'p': self.p.tolist(),
            'K': self.K.tolist(),
            'epsilon': float(self.epsilon),
            'num_of_iterations': int(self.num_of_iterations)
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

    # ------------------------------------------------------------------
    # HELPERS (STABILITY)
    # ------------------------------------------------------------------

    def safe_ro(self, ro, eps=1e-8):
        return min(max(ro, 0.0), 1.0 - eps)

    # ------------------------------------------------------------------
    # VISIT RATIOS
    # ------------------------------------------------------------------

    def calculate_E(self):
        for r in range(self.r):
            A = self.p[r].T - np.eye(self.n)
            b = np.zeros(self.n)
            b[0] = 1
            x, _ = nnls(A, b)
            self.e[:, r] = x

    # ------------------------------------------------------------------
    # LOADS
    # ------------------------------------------------------------------

    def calculate_Ro_ir(self, i, r):
        if self.service_type[i] == 1:
            ro = self.lambdas[r] * self.e[i, r] / (self.m[i] * self.mi[i, r])
        else:
            ro = self.lambdas[r] * self.e[i, r] / self.mi[i, r]
        return self.safe_ro(ro)

    def calculate_Ro_i(self, i):
        ro = 0.0
        for r in range(self.r):
            ro += self.calculate_Ro_ir(i, r)
        return self.safe_ro(ro)

    # ------------------------------------------------------------------
    # LAMBDAS
    # ------------------------------------------------------------------

    def reset_lambdas(self):
        self.lambdas = np.array([self.epsilon] * self.r)

    def run_iteration_method_for_Lambda_r(self, relaxation=0.01):
        convergence = []

        for it in range(self.num_of_iterations):
            prev = self.lambdas.copy()

            for r in range(self.r):
                denom = 0.0
                for i in range(self.n):
                    ro_i = self.calculate_Ro_i(i)
                    denom += self.e[i, r] / self.mi[i, r] / (1 - ro_i + 1e-8)

                new_lambda = self.K[r] / denom if denom > 0 else 0.0
                # relaksacja
                self.lambdas[r] = relaxation * new_lambda + (1 - relaxation) * self.lambdas[r]

            err = np.linalg.norm(self.lambdas - prev)
            convergence.append(err)

            if err < self.epsilon:
                return it + 1, convergence

        return self.num_of_iterations, convergence

    # ------------------------------------------------------------------
    # K_ir + NORMALIZATION
    # ------------------------------------------------------------------

    def calculate_K_ir(self):
        K_ir = np.zeros((self.n, self.r))
        for r in range(self.r):
            total_e = np.sum(self.e[:, r])
            if total_e > 0:
                K_ir[:, r] = self.K[r] * self.e[:, r] / total_e
            else:
                K_ir[:, r] = self.K[r] / self.n
        return K_ir

    # ------------------------------------------------------------------
    # FULL SUM
    # ------------------------------------------------------------------

    def run_SUM(self, alpha=0.5):
        """
        Pełna iteracja metody SUM z relaksacją K_ir.
        alpha: float, relaksacja K_ir (0 < alpha <= 1)
        """
        # inicjalizacja
        self.reset_lambdas()

        self.K_ir = np.zeros((self.n, self.r))
        for r in range(self.r):
            self.K_ir[:, r] = self.K[r] / self.n

        convergence = []

        for it in range(self.num_of_iterations):
            prev_K_ir = self.K_ir.copy()
            prev_lambdas = self.lambdas.copy()

            # aktualizacja lambd
            _, _ = self.run_iteration_method_for_Lambda_r()

            # nowe K_ir
            K_new = self.calculate_K_ir()
            K_new = self.normalize_K_ir(K_new)
            K_new[K_new < 0] = 0.0

            # relaksacja
            self.K_ir = alpha * K_new + (1 - alpha) * prev_K_ir

            # błąd względem poprzedniego K_ir
            err = np.max(np.abs(self.K_ir - prev_K_ir))
            convergence.append(err)

            if err < self.epsilon:
                print(f"SUM converged in {it+1} iterations (error={err:.3e})")
                return it + 1, convergence

        print(f"SUM reached max iterations ({self.num_of_iterations}), last error={err:.3e}")
        return self.num_of_iterations, convergence



    # ------------------------------------------------------------------
    # TIMES
    # ------------------------------------------------------------------

    def calculate_T_ir(self):
        """
        Czas przebywania T_ir = K_ir / (lambda_r * e_ir)
        """
        for i in range(self.n):
            for r in range(self.r):
                lam_ir = self.lambdas[r] * self.e[i, r]
                if lam_ir > 0:
                    self.T_ir[i, r] = self.K_ir[i, r] / lam_ir
                else:
                    self.T_ir[i, r] = 0.0


# ----------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sm = SummationMethod()
    iterations, convergence = sm.run_SUM()

    print("lambdas:\n", sm.lambdas)
    print("K_ir:\n", sm.K_ir)
    print("sum K_ir (== K):\n", np.sum(sm.K_ir, axis=0))

    sm.calculate_T_ir()
    print("T_ir:\n", sm.T_ir)
