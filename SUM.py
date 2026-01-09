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
    
    def _set_defaults(self):
        self.r = 4
        self.n = 8
        self.service_type = np.array([1, 1, 1, 3, 3, 1, 3, 3])
        self.m = np.array([10, 10, 1, 10, 1, 1, 1, 1])
        self.mi = np.array([
            [2, 2, 2, 2],
            [3, 3, 3, 3],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [1, 1, 1, 1]
        ])
        self.p = np.array([
            [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.7, 0, 0.3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.7, 0.3],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.7, 0.3, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.7, 0.3],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.7, 0.3],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0]
            ],
            [
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.7, 0.3],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0]
            ]
        ])
        self.K = np.array([2, 4, 2, 3])
        self.epsilon = 1e-07
        self.num_of_iterations = 200
        
        self.e = np.zeros(shape=(self.n, self.r))
        self.lambdas = np.array([self.epsilon] * self.r)
        self.T_ir = np.zeros(shape=(self.n, self.r))
        self.K_ir = np.zeros(shape=(self.n, self.r))
    
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
    
    def load_config(self, filepath: str):
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        self.r = config['r']
        self.n = config['n']
        self.service_type = np.array(config['service_type'])
        self.m = np.array(config['m'])
        self.mi = np.array(config['mi'])
        self.p = np.array(config['p'])
        self.K = np.array(config['K'])
        self.epsilon = config['epsilon']
        self.num_of_iterations = config['num_of_iterations']
        
        self.e = np.zeros(shape=(self.n, self.r))
        self.lambdas = np.array([self.epsilon] * self.r)
        self.T_ir = np.zeros(shape=(self.n, self.r))
        self.K_ir = np.zeros(shape=(self.n, self.r))
    
    def calculate_Ro_ir(self, i, r):
        Ro_ir = 0
        if self.service_type[i] == 1:
            Ro_ir = self.lambdas[r] * self.e[i, r] / (self.m[i] * self.mi[i, r])
        else:
            Ro_ir = self.lambdas[r] * self.e[i, r] / self.mi[i, r]
        return Ro_ir

    def calculate_E(self):
        for r in range(self.r):
            A = self.p[r].T - np.diag([0] + [1] * (self.n - 1))
            b = np.array([1] + [0] * (self.n - 1))
            x, _ = nnls(A, b)
            self.e[:, r] = x
    
    def calculate_Ro_i(self, i):
        Ro_i = 0
        for r in range(self.r):
            if self.service_type[i] == 1:
                Ro_i += self.lambdas[r] * self.e[i, r] / (self.m[i] * self.mi[i, r])
            else:
                Ro_i += self.lambdas[r] * self.e[i, r] / (self.mi[i, r])
        return Ro_i
    
    def calculate_P_mi(self, i, ro_i):
        sum_ = 0
        m_i = int(self.m[i])
        for ki in range(m_i):
            sum_ += ((m_i * ro_i) ** ki) / math.factorial(ki)
        factor1 = ((m_i * ro_i) ** self.m[i]) / (math.factorial(m_i) * (1 - ro_i))
        factor2 = 1 / (sum_ + ((m_i * ro_i) ** self.m[i]) / math.factorial(m_i) * 1 / (1 - ro_i))
        return factor1 * factor2

    def calcucate_Fix_ir(self, ro_i, i, r):
        e_ir = self.e[i, r]
        mi_ir = self.mi[i, r]
        if self.service_type[i] in [1, 2, 4]:
            m_i = int(self.m[i])
            K = np.sum(self.K)
            if self.service_type[i] == 1 and m_i > 1:
                P_mi = self.calculate_P_mi(i, ro_i)
                return e_ir / mi_ir + ((e_ir / (mi_ir * m_i)) / (1 - ro_i * (K - m_i - 1) / (K - m_i))) * P_mi
            else:
                return (e_ir / mi_ir) / (1 - ro_i * (K-1) / K)
        else:
            return e_ir / mi_ir
    
    def run_iteration_method_for_Lambda_r(self):
        current_error = None
        iterations_run = 0
        for i in range(self.num_of_iterations):
            if current_error is not None and current_error <= self.epsilon:
                iterations_run = i
                break
            else:
                prev_lambdas_r = self.lambdas.copy()
                self._calculate_Lambda_r()
                current_error = self.calculate_Error(prev_lambdas_r, self.lambdas)
                iterations_run = i + 1
        return iterations_run
    
    def _calculate_Lambda_r(self):
        for r in range(self.r):
            sum_of_Fix_ir = 0
            for i in range(self.n):
                Ro_i = self.calculate_Ro_i(i)
                sum_of_Fix_ir += self.calcucate_Fix_ir(Ro_i, i, r)
            if sum_of_Fix_ir == 0:
                self.lambdas[r] = 0
            else:
                self.lambdas[r] = self.K[r] / sum_of_Fix_ir
    
    def calculate_Error(self, prev_lambda_r, lambda_r):
        return np.sqrt(np.sum((prev_lambda_r - lambda_r) ** 2))
    
    def calculate_K_ir(self):
        K_matrix = np.zeros(shape=(self.n, self.r), dtype=np.float32)
        for i in range(self.n):
            for r in range(self.r):
                mi_ir = self.mi[i, r]
                if self.service_type[i] in [1, 2, 4]:
                    K = np.sum(self.K)
                    ro_i = self.calculate_Ro_i(i)
                    ro_ir = self.calculate_Ro_ir(i, r)
                    m_i = self.m[i]
                    if self.service_type[i] == 1 and m_i > 1:
                        P_mi = self.calculate_P_mi(i, ro_i)
                        K_matrix[i, r] = m_i * ro_ir + (ro_ir / (1 - ro_i * (K - m_i - 1) / (K - m_i))) * P_mi
                    else:
                        denom = 1 - ro_i * (K-1)/K
                        K_matrix[i, r] = ro_ir / denom
                else:
                    lambda_ir = self.lambdas[r] * self.e[i, r]
                    K_matrix[i, r] = lambda_ir / mi_ir
        self.K_ir = K_matrix
        return K_matrix
    
    def calculate_T_ir(self):
        for i in range(self.n):
            for r in range(self.r):
                lambda_ir = self.lambdas[r] * self.e[i, r]
                if lambda_ir:
                    self.T_ir[i, r] = self.K_ir[i, r] / lambda_ir
                else:
                    self.T_ir[i, r] = 0

# --- Uruchomienie ---
if __name__ == '__main__':
    sm = SummationMethod()
    print("Matrix e_ir (średnia liczba wizyt (visit ratios)):\n", sm.e,"\n")
    sm.run_iteration_method_for_Lambda_r()
    print("Lambdas (intensywnosc przeplywu kazdej z klas):\n", sm.lambdas,"\n")
    sm.calculate_K_ir()
    print("K_ir (srednia ilosc zgloszen klasy r w węźle i (w tym zgloszenia w obsludze i kolejce)):\n", sm.K_ir,"\n")
    print("K:\n", np.sum(sm.K_ir, axis=0),"\n")
    sm.calculate_T_ir()
    print("T_ir (sredni czas przebywania klasy r w węźle i):\n", sm.T_ir,"\n")


