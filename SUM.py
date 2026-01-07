import numpy as np
import math
from scipy.optimize import nnls

class SummationMethod:
    def __init__(self) -> None:
        # Liczba klas i węzłów
        self.r = 4  # klasy: Uszkodzenia elektryczne, mechaniczne, mieszane, uproszczone
        self.n = 8  # węzły
        
        # Typy węzłów: 1=FIFO, 3=IS
        self.service_type = np.array([1, 1, 1, 3, 3, 1, 3, 3])
        
        # Kanały obsługi w węzłach
        self.m = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        
        # Pojemność serwerów dla każdej klasy
        self.mi = np.array([
            [3, 3, 3, 3],  # 1: Przyjmowanie zgłoszenia
            [2, 2, 2, 2],  # 2: Dział elektryczny
            [2, 2, 2, 2],  # 3: Dział mechaniczny
            [1, 1, 1, 1],  # 4: Testy elektryczne
            [1, 1, 1, 1],  # 5: Testy mechaniczne
            [2, 2, 2, 2],  # 6: Wycena/dokumentacja
            [1, 1, 1, 1],  # 7: Obsługa klienta
            [1, 1, 1, 1]   # 8: Stała eksploatacja
        ])
        
        self.p = np.array([

        # Klasa 1: 1 → 2 → 4 → 6 → 7 → 8 → 1
        [
        [0, 1, 0, 0, 0, 0, 0, 0],  # 1 → 2
        [0, 0, 0, 1, 0, 0, 0, 0],  # 2 → 4
        [0, 0, 1, 0, 0, 0, 0, 0],  # 3 nieosiągalny → zostaje w 3
        [0, 0, 0, 0, 0, 1, 0, 0],  # 4 → 6
        [0, 0, 0, 0, 1, 0, 0, 0],  # 5 nieosiągalny → zostaje w 5
        [0, 0, 0, 0, 0, 0, 0.7, 0.3], # 6 → 7 (0.7), 6 → 8 (0.3)
        [0, 0, 0, 0, 0, 0, 0, 1],  # 7 → 8
        [1, 0, 0, 0, 0, 0, 0, 0]   # 8 → 1
        ],

        # Klasa 2: 1 → 3 → 5 → 6 → 7 → 8 → 1
        [
        [0, 0, 1, 0, 0, 0, 0, 0],  # 1 → 3
        [0, 1, 0, 0, 0, 0, 0, 0],  # 2 nieosiągalny → zostaje w 2
        [0, 0, 0, 0, 1, 0, 0, 0],  # 3 → 5
        [0, 0, 0, 1, 0, 0, 0, 0],  # 4 nieosiągalny → zostaje w 4
        [0, 0, 0, 0, 0, 1, 0, 0],  # 5 → 6
        [0, 0, 0, 0, 0, 0, 0.7, 0.3], # 6 → 7 (0.7), 6 → 8 (0.3)
        [0, 0, 0, 0, 0, 0, 0, 1],  # 7 → 8
        [1, 0, 0, 0, 0, 0, 0, 0]   # 8 → 1
        ],

        # Klasa 3: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 1
        [
        [0, 1, 0, 0, 0, 0, 0, 0],  # 1 → 2
        [0, 0, 1, 0, 0, 0, 0, 0],  # 2 → 3
        [0, 0, 0, 1, 0, 0, 0, 0],  # 3 → 4
        [0, 0, 0, 0, 1, 0, 0, 0],  # 4 → 5
        [0, 0, 0, 0, 0, 1, 0, 0],  # 5 → 6
        [0, 0, 0, 0, 0, 0, 0.7, 0.3], # 6 → 7 (0.7), 6 → 8 (0.3)
        [0, 0, 0, 0, 0, 0, 0, 1],  # 7 → 8
        [1, 0, 0, 0, 0, 0, 0, 0]   # 8 → 1
        ],

        # Klasa 4: 1 → 6 → 7 → 8 → 1 z rozgałęzieniem 6 → 7 / 8
        [
        [0, 0, 0, 0, 0, 1, 0, 0],  # 1 → 6
        [0, 1, 0, 0, 0, 0, 0, 0],  # 2 nieosiągalny → zostaje w 2
        [0, 0, 1, 0, 0, 0, 0, 0],  # 3 nieosiągalny → zostaje w 3
        [0, 0, 0, 1, 0, 0, 0, 0],  # 4 nieosiągalny → zostaje w 4
        [0, 0, 0, 0, 1, 0, 0, 0],  # 5 nieosiągalny → zostaje w 5
        [0, 0, 0, 0, 0, 0, 0.7, 0.3], # 6 → 7 (0.7), 6 → 8 (0.3)
        [0, 0, 0, 0, 0, 0, 0, 1],  # 7 → 8
        [1, 0, 0, 0, 0, 0, 0, 0]   # 8 → 1
        ]

        ])
        
        # Średnia liczba zgłoszeń dla klas
        self.K = np.array([2, 4, 2, 3])
        
        # Parametry obliczeniowe SUM
        self.epsajlon = 1e-05
        self.e = np.zeros(shape=(self.n, self.r))  # średnia liczba wizyt
        self.lambdas = np.array([self.epsajlon] * self.r)
        self.num_of_iterations = 200
        self.T_ir = np.zeros(shape=(self.n, self.r))
        self.K_ir = np.zeros(shape=(self.n, self.r))
        
        # Obliczenie macierzy e
        self.calculate_E()
    
    # --- Funkcje SUM pozostają niezmienione ---
    def calculate_Ro_ir(self, i, r):
        Ro_ir = 0
        if self.service_type[i] == 1: # Typ 1, (m_i >= 1)
            Ro_ir = self.lambdas[r] * self.e[i, r] / (self.m[i] * self.mi[i, r])
        else: # Typ 2, 3, 4
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
        for i in range(self.num_of_iterations):
            if current_error is not None and current_error <= self.epsajlon:
                print(f"terminate {i}")
                break
            else:
                prev_lambdas_r = self.lambdas.copy()
                self._calculate_Lambda_r()
                current_error = self.calculate_Error(prev_lambdas_r, self.lambdas)
    
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
                        if denom <= 0:
                            K_matrix[i, r] = ro_ir 
                        else:
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
    print("Matrix e_ir:\n", sm.e)
    sm.run_iteration_method_for_Lambda_r()
    print("Lambdas:\n", sm.lambdas)
    sm.calculate_K_ir()
    print("K_ir:\n", sm.K_ir)
    print("K:\n", np.sum(sm.K_ir, axis=0))
    sm.calculate_T_ir()
    print("T_ir:\n", sm.T_ir)
