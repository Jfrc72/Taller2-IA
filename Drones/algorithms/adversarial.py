from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        # TODO: Implement your code here
        def minimax_value(s: GameState, agent_index: int, depth_left: int) -> float:
            if s.is_win() or s.is_lose() or depth_left == 0:
                return self.evaluation_function(s)

            num_agents = s.get_num_agents()
            legal = s.get_legal_actions(agent_index)
            if not legal:
                return self.evaluation_function(s)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth_left - 1 if next_agent == 0 else depth_left

            # MAX: drone
            if agent_index == 0:
                best = float("-inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    val = minimax_value(succ, next_agent, next_depth)
                    best = max(best, val)
                return best
            # MIN: hunters
            else:
                best = float("inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    val = minimax_value(succ, next_agent, next_depth)
                    best = min(best, val)
                return best

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        best_action = None
        best_value = float("-inf")

        num_agents = state.get_num_agents()

        for a in legal_actions:
            succ = state.generate_successor(0, a)
            val = minimax_value(succ, 1 % num_agents, self.depth)
            if val > best_value:
                best_value = val
                best_action = a

        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        # TODO: Implement your code here (BONUS)
        def alphabeta(
            s: GameState,
            agent_index: int,
            depth_left: int,
            alpha: float,
            beta: float,
        ) -> float:

            if s.is_win() or s.is_lose() or depth_left == 0:
                return self.evaluation_function(s)

            num_agents = s.get_num_agents()
            legal = s.get_legal_actions(agent_index)
            if not legal:
                return self.evaluation_function(s)

            next_agent = (agent_index + 1) % num_agents
            next_depth = depth_left - 1 if next_agent == 0 else depth_left

            if agent_index == 0:  # MAX
                v = float("-inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    val = alphabeta(succ, next_agent, next_depth, alpha, beta)
                    v = max(v, val)

                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:  # MIN
                v = float("inf")
                for a in legal:
                    succ = s.generate_successor(agent_index, a)
                    val = alphabeta(succ, next_agent, next_depth, alpha, beta)
                    v = min(v, val)

                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v

        legal_actions = state.get_legal_actions(0)
        if not legal_actions:
            return None

        alpha = float("-inf")
        beta = float("inf")

        best_action = None
        best_value = float("-inf")

        num_agents = state.get_num_agents()

        for a in legal_actions:
            succ = state.generate_successor(0, a)
            val = alphabeta(succ, 1 % num_agents, self.depth, alpha, beta)
            if val > best_value:
                best_value = val
                best_action = a
            alpha = max(alpha, best_value)

        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Expectimax agent with a mixed hunter model.

    Each hunter acts randomly with probability self.prob and greedily
    (worst-case / MIN) with probability 1 - self.prob.

    * When prob = 0:  behaves like Minimax (hunters always play optimally).
    * When prob = 1:  pure expectimax (hunters always play uniformly at random).
    * When 0 < prob < 1: weighted combination that correctly models the
      actual MixedHunterAgent used at game-play time.

    Chance node formula:
        value = (1 - p) * min(child_values) + p * mean(child_values)
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using expectimax with mixed hunter model.

        Tips:
        - Drone nodes are MAX (same as Minimax).
        - Hunter nodes are CHANCE with mixed model: the hunter acts greedily with
          probability (1 - self.prob) and uniformly at random with probability self.prob.
        - Mixed expected value = (1-p) * min(child_values) + p * mean(child_values).
        - When p=0 this reduces to Minimax; when p=1 it is pure uniform expectimax.
        - Do NOT prune in expectimax (unlike alpha-beta).
        - self.prob is set via the constructor argument prob.
        """
    # TODO: Implement your code here 

    def get_action(self, state: GameState) -> Directions | None:
    #:D riquitii
        def expectimax(state_actual: GameState, agent_index: int, depth_left: int) -> float:
            # nodos extremos (hojitas)
            if state_actual.is_win() or state_actual.is_lose() or depth_left == 0:
                return self.evaluation_function(state_actual)

            acciones = state_actual.get_legal_actions(agent_index)
            if not acciones:
                return self.evaluation_function(state_actual)

            num_agents = state_actual.get_num_agents()
            siguiente_agente = (agent_index + 1) % num_agents

            if siguiente_agente == 0:
                siguiente_depth = depth_left - 1
            else:
                siguiente_depth = depth_left

            # dron (max)
            if agent_index == 0:
                mejor_valor = float("-inf")

                for accion in acciones:
                    sucesor = state_actual.generate_successor(agent_index, accion)
                    valor = expectimax(sucesor, siguiente_agente, siguiente_depth)

                    if valor > mejor_valor:
                        mejor_valor = valor

                return mejor_valor

            # turno del cazador, nodo probabilistico (azar)
            valores = []

            for accion in acciones:
                sucesor = state_actual.generate_successor(agent_index, accion)
                valor = expectimax(sucesor, siguiente_agente, siguiente_depth)
                valores.append(valor)

            peor_caso = min(valores)
            promedio = sum(valores) / len(valores)

            return (1 - self.prob) * peor_caso + self.prob * promedio

        acciones_drone = state.get_legal_actions(0)
        if not acciones_drone:
            return None

        mejor_accion = None
        mejor_valor = float("-inf")
        num_agents = state.get_num_agents()

        for accion in acciones_drone:
            sucesor = state.generate_successor(0, accion)
            valor = expectimax(sucesor, 1 % num_agents, self.depth)

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

        return mejor_accion