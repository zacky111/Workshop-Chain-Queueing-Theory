import numpy as np
import math
from scipy.optimize import nnls

# ustawienie formatu wypisywania (np. 3 miejsca po przecinku)
np.set_printoptions(precision=3, suppress=True)

class SummationMethod:
    def __init__(self, r=None, n=None, service_type=None, m=None, mi=None, p=None, K=None, epsilon=None, num_of_iterations=None) -> None:
        # Domyślne wartości (tak jak wcześniej)
        default_r = 4
        default_n = 8

        # Jeśli macierz mi/p została przekazana, odczytujemy r i n z niej
        if mi is not None:
            mi_arr = np.array(mi)
            n_from_mi, r_from_mi = mi_arr.shape
            self.mi = mi_arr
            self.n = n_from_mi
            self.r = r_from_mi
        else:
            self.r = r if r is not None else default_r
            self.n = n if n is not None else default_n
            # domyślna mi (tak jak wcześniej)
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

        # Typy węzłów: 1=FIFO, 3=IS
        self.service_type = np.array(service_type) if service_type is not None else np.array([1, 1, 1, 3, 3, 1, 3, 3])

        # Kanały obsługi w węzłach
        self.m = np.array(m) if m is not None else np.array([1, 1, 1, 1, 1, 1, 1, 1])

        # P macierze (r x n x n)
        if p is not None:
            self.p = np.array(p)
        else:
            # domyślne p (jak wcześniej)
            self.p = np.array([
                # Klasa 1: 1 → 2 → 4 → 6 → 7 → 8 → 1
                [
                [0, 1, 0, 0, 0, 0, 0, 0],  # 1 → 2
                [0, 0, 0, 0.7, 0, 0.3, 0, 0],  # 2 → 4
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
                [0, 0, 0, 0, 0.7, 0.3, 0, 0],  # 3 → 5
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
        self.K = np.array(K) if K is not None else np.array([2, 4, 2, 3])
        
        # Parametry obliczeniowe SUM
        self.epsilon = epsilon if epsilon is not None else 1e-05
        self.e = np.zeros(shape=(self.n, self.r))  # średnia liczba wizyt
        self.lambdas = np.array([self.epsilon] * self.r)
        self.num_of_iterations = num_of_iterations if num_of_iterations is not None else 200
        self.T_ir = np.zeros(shape=(self.n, self.r))
        self.K_ir = np.zeros(shape=(self.n, self.r))
        
        # Obliczenie macierzy e
        self.calculate_E()
    
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
            if current_error is not None and current_error <= self.epsilon:
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

    # Nowe pomocnicze metody:
    def to_dict(self):
        return {
            "r": int(self.r),
            "n": int(self.n),
            "service_type": self.service_type.tolist(),
            "m": self.m.tolist(),
            "mi": self.mi.tolist(),
            "p": self.p.tolist() if hasattr(self, "p") else None,
            "K": self.K.tolist(),
            "epsilon": float(self.epsilon),
            "num_of_iterations": int(self.num_of_iterations)
        }

    def set_params(self, params: dict):
        # params może zawierać dowolne z kluczy: r,n,service_type,m,mi,p,K,epsilon,num_of_iterations
        if "mi" in params:
            mi_arr = np.array(params["mi"])
            self.mi = mi_arr
            self.n, self.r = mi_arr.shape
        if "r" in params:
            self.r = int(params["r"])
        if "n" in params:
            self.n = int(params["n"])
        if "service_type" in params:
            self.service_type = np.array(params["service_type"])
        if "m" in params:
            self.m = np.array(params["m"])
        if "p" in params:
            self.p = np.array(params["p"])
        if "K" in params:
            self.K = np.array(params["K"])
        if "epsilon" in params:
            self.epsilon = float(params["epsilon"])
        if "num_of_iterations" in params:
            self.num_of_iterations = int(params["num_of_iterations"])

        # Zresetuj wewnętrzne wektory zgodnie z wymiarami
        self.e = np.zeros(shape=(self.n, self.r))
        self.lambdas = np.array([self.epsilon] * self.r)
        self.T_ir = np.zeros(shape=(self.n, self.r))
        self.K_ir = np.zeros(shape=(self.n, self.r))

        # Przelicz e
        self.calculate_E()

# --- Uruchomienie ---
if __name__ == '__main__':
    sm = SummationMethod()
    print("Matrix e_ir (średnia liczba wizyt (visit ratios)):\n", sm.e,"\n")
    sm.run_iteration_method_for_Lambda_r()
    print("Lambdas (intensywnosc przeplywu kazdej z klas):\n", sm.lambdas,"\n")
    sm.calculate_K_ir()
    print("K_ir (srednia ilosc zgloszen klasy r w węźle i (w tym zgloszenia w obsludze i kolejce)):\n", sm.K_ir,"\n")
    sm.calculate_T_ir()
    print("T_ir (sredni czas przebywania klasy r w węźle i):\n", sm.T_ir,"\n")
