"""
Execution entry point for Analytical DP and Monte Carlo Simulation Engine.
"""

from math_solver import AnalyticalDPSolver
from simulation import AdaptiveAgent, OpponentAgent, simulate_single_round, estimate_stopping_time

def run_benchmarks():
    print("=" * 60)
    print("  STOCHASTIC GAME SIMULATION & ANALYTICAL DP BENCHMARK")
    print("=" * 60)

    # 1. Analytical DP Verification
    solver = AnalyticalDPSolver()
    horizon = 20
    e_val = solver.expected_net_score(horizon)
    var_val = solver.variance_net_score(horizon)
    print(f"\n[1] Analytical DP Results (Horizon T={horizon}):")
    print(f"    E[Net Points]   (mod 10^9+7) : {e_val}")
    print(f"    Var(Net Points) (mod 10^9+7) : {var_val}")

    # 2. Monte Carlo Stopping-Time Estimation
    target_wins = 15
    trials = 10_000
    print(f"\n[2] Estimating Expected Stopping Time E[tau] (Target Wins={target_wins}, Trials={trials})...")
    e_tau = estimate_stopping_time(target_wins=target_wins, num_trials=trials)
    print(f"    E[tau] to achieve {target_wins} wins : {e_tau:.3f} rounds")

    # 3. Strategy Simulation (Adaptive vs Random Opponent)
    print(f"\n[3] Running 100,000 Rounds of Adaptive Policy vs Random Opponent...")
    alice = AdaptiveAgent()
    bob = OpponentAgent(policy="random")
    for _ in range(100_000):
        simulate_single_round(alice, bob)
    
    print(f"    Alice Final Score : {alice.points:.1f} (Win Rate: {alice.wins/len(alice.history):.2%})")
    print(f"    Bob Final Score   : {bob.points:.1f}")
    print("\nBenchmark completed successfully.")

if __name__ == "__main__":
    run_benchmarks()