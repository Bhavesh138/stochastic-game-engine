"""
Stochastic Game Engine and Monte Carlo Simulator
Simulates multi-stage adversarial interactions with state-dependent payoff matrices.
"""

import random
from typing import List, Tuple

# Actions: 0 = Attack, 1 = Balanced, 2 = Defence
ATTACK, BALANCED, DEFENCE = 0, 1, 2

class GamePayoff:
    @staticmethod
    def get_matrix(alice_pts: float, bob_pts: float) -> List[List[Tuple[float, float, float]]]:
        """
        Dynamically updates the state-dependent win/draw/loss probabilities.
        """
        total = alice_pts + bob_pts
        attack_prob = (bob_pts / total, 0.0, alice_pts / total) if total > 0 else (0.5, 0.0, 0.5)
        return [
            [attack_prob,        (0.7, 0.0, 0.3), (5 / 11, 0.0, 6 / 11)],
            [(0.3, 0.0, 0.7),    (1 / 3, 1 / 3, 1 / 3), (0.3, 0.5, 0.2)],
            [(6 / 11, 0.0, 5 / 11), (0.2, 0.5, 0.3), (0.1, 0.8, 0.1)]
        ]

class AdaptiveAgent:
    """Agent that chooses strategies using dynamic expected-value maximisation."""
    def __init__(self, name="Alice"):
        self.name = name
        self.points = 1.0
        self.wins = 0
        self.history = [1, 0]  # past results

    def play_move(self, opponent_pts: float) -> int:
        matrix = GamePayoff.get_matrix(self.points, opponent_pts)
        
        # Calculate expected payoffs against a randomized opponent
        # E[Action] = (1/3) * sum_{opp_move} (P(win)*1 + P(draw)*0.5)
        expected_payoffs = []
        for action in [ATTACK, BALANCED, DEFENCE]:
            ev = 0.0
            for opp_move in [ATTACK, BALANCED, DEFENCE]:
                p_win, p_draw, _ = matrix[action][opp_move]
                ev += (p_win * 1.0 + p_draw * 0.5) / 3.0
            expected_payoffs.append(ev)
        
        return int(expected_payoffs.index(max(expected_payoffs)))

    def observe(self, result: float):
        self.history.append(result)
        self.points += result
        if result == 1.0:
            self.wins += 1

class OpponentAgent:
    """Opponent capable of Deterministic-Reactive or Uniform-Random policies."""
    def __init__(self, policy: str = "random"):
        self.policy = policy
        self.points = 1.0
        self.history = [0, 1]

    def play_move(self) -> int:
        if self.policy == "random":
            return random.choice([ATTACK, BALANCED, DEFENCE])
        
        # Reactive Policy:
        last = self.history[-1]
        if last == 1.0:
            return DEFENCE
        elif last == 0.5:
            return BALANCED
        return ATTACK

    def observe(self, result: float):
        self.history.append(result)
        self.points += result

def simulate_single_round(alice: AdaptiveAgent, bob: OpponentAgent):
    matrix = GamePayoff.get_matrix(alice.points, bob.points)
    a_move = alice.play_move(bob.points)
    b_move = bob.play_move()

    p_win, p_draw, _ = matrix[a_move][b_move]
    r = random.random()

    if r < p_win:
        alice.observe(1.0)
        bob.observe(0.0)
    elif r < p_win + p_draw:
        alice.observe(0.5)
        bob.observe(0.5)
    else:
        alice.observe(0.0)
        bob.observe(1.0)

def estimate_stopping_time(target_wins: int, num_trials: int = 50_000) -> float:
    """Estimates E[tau], the expected number of rounds to reach `target_wins`."""
    total_rounds = 0
    for _ in range(num_trials):
        alice = AdaptiveAgent()
        bob = OpponentAgent(policy="reactive")
        rounds = 0
        while alice.wins < target_wins:
            simulate_single_round(alice, bob)
            rounds += 1
        total_rounds += rounds
    return total_rounds / num_trials