        """
        Metoda SUM z relaksacją lambd, bez szybkiego zbiegania.
        """
        self.K_ir = np.zeros((self.n, self.r))
        for r in range(self.r):
            self.K_ir[:, r] = self.K[r] / self.n

        for it in range(self.num_of_iterations):
            lambdas_prev = self.lambdas.copy()

            # aktualizacja lambd z relaksacją
            self.run_iteration_method_for_Lambda_r(alpha=alpha)

            # nowe K_ir
            K_ir_new = self.calculate_K_ir()
            K_ir_new = self.normalize_K_ir(K_ir_new)

            error_in_SUM = self.calculate_Error(lambdas_prev, self.lambdas)
            print(f"Iter {it+1}, SUM error: {error_in_SUM:.6e}")

            self.K_ir = K_ir_new

            if error_in_SUM < self.epsilon:
                print(f"SUM converged in {it+1} iterations")
                break