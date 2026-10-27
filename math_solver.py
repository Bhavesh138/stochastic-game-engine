"""
Analytical Dynamic Programming Solver for Non-Stationary Stochastic Games
Calculates exact probabilities, expectations, and variances using finite field modular arithmetic.
"""

MOD = 1_000_000_007

def mod_add(a: int, b: int, m: int = MOD) -> int:
    return ((a % m) + (b % m)) % m

def mod_sub(a: int, b: int, m: int = MOD) -> int:
    return ((a % m) - (b % m) + m) % m

def mod_mul(a: int, b: int, m: int = MOD) -> int:
    return ((a % m) * (b % m)) % m

def mod_div(a: int, b: int, m: int = MOD) -> int:
    # Fermat's Little Theorem for modular inverse
    return mod_mul(a, pow(b, m - 2, m), m)

class AnalyticalDPSolver:
    def __init__(self, modulo: int = MOD):
        self.mod = modulo

    def compute_probability_table(self, max_a: int, max_b: int):
        """
        Computes DP table where dp[i][j] is the exact probability
        of Alice winning i rounds and Bob winning j rounds.
        """
        dp = [[0] * (max_b + 1) for _ in range(max_a + 1)]
        dp[1][1] = 1  # Base case: 1 win each

        for i in range(1, max_a + 1):
            for j in range(1, max_b + 1):
                if i == 1 and j == 1:
                    continue
                alice_last = mod_mul(dp[i - 1][j], j, self.mod)
                bob_last = mod_mul(dp[i][j - 1], i, self.mod)
                total_games = mod_add(i, j, self.mod)
                
                if total_games > 1:
                    term1 = mod_div(alice_last, total_games - 1, self.mod)
                    term2 = mod_div(bob_last, total_games - 1, self.mod)
                    dp[i][j] = mod_add(term1, term2, self.mod)
        return dp

    def expected_net_score(self, total_rounds: int) -> int:
        """
        Calculates E[sum(X_i)] modulo 10^9+7 where X_i in {+1, 0, -1}
        """
        dp = self.compute_probability_table(total_rounds, total_rounds)
        e_sum = 0
        for i in range(1, total_rounds):
            prob = dp[i][total_rounds - i]
            net_points = 2 * i - total_rounds
            contribution = mod_mul(net_points, prob, self.mod)
            e_sum = mod_add(e_sum, contribution, self.mod)
        return e_sum

    def variance_net_score(self, total_rounds: int) -> int:
        """
        Calculates Var(sum(X_i)) modulo 10^9+7
        """
        dp = self.compute_probability_table(total_rounds, total_rounds)
        var_sum = 0
        for i in range(1, total_rounds):
            prob = dp[i][total_rounds - i]
            diff = 2 * i - total_rounds
            diff_sq = mod_mul(diff, diff, self.mod)
            contribution = mod_mul(diff_sq, prob, self.mod)
            var_sum = mod_add(var_sum, contribution, self.mod)
        return var_sum