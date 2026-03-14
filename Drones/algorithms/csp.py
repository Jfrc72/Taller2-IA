from __future__ import annotations

from typing import TYPE_CHECKING
from collections import deque
from copy import deepcopy

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    estats = {"assignments": 0, "backtracks": 0}
    def backtrack(assignment: dict[str, str]):
        if csp.is_complete(assignment):
            return assignment

        no_asignado = csp.get_unassigned_variables(assignment)
        var = no_asignado[0]

        for value in csp.domains[var]:
            estats["assignments"] += 1

        for valor in csp.domains[var]:
            if csp.is_consistent(var, valor, assignment):
                csp.assign(var, valor, assignment)

                resultado = backtrack(assignment)
                if resultado is not None:
                    return resultado

                csp.unassign(var, assignment)
                estats["backtracks"] += 1

        return None
    resultado = backtrack({})
    print(f"Asignaciones intentadas: {estats['assignments']}")
    print(f"Backtracks: {estats['backtracks']}")
    return resultado
    


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    estats = {"assignments": 0, "backtracks": 0}
    def forward_check(var: str, assignment: dict[str, str]):
        dominios_guardados = {v: list(dom) for v, dom in csp.domains.items()}

        for vecino in csp.get_neighbors(var):
            if vecino in assignment:
                continue

            nuevo_dominio = []
            for val in csp.domains[vecino]:
                if csp.is_consistent(vecino, val, assignment):
                    nuevo_dominio.append(val)

            csp.domains[vecino] = nuevo_dominio

            if len(nuevo_dominio) == 0:
                return None

        return dominios_guardados

    def restore_domains(dominios_guardados: dict[str, list[str]]) -> None:
        csp.domains = dominios_guardados

    def backtrack(assignment: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(assignment):
            return assignment

        no_asignado = csp.get_unassigned_variables(assignment)
        var = no_asignado[0]

        for valor in csp.domains[var]:
            estats["assignments"] += 1
            if csp.is_consistent(var, valor, assignment):
                csp.assign(var, valor, assignment)

                guardado = forward_check(var, assignment)
                if guardado is not None:
                    resultado = backtrack(assignment)
                    if resultado is not None:
                        return resultado
                    restore_domains(guardado)

                csp.unassign(var, assignment)
                estats["backtracks"] += 1

        return None

    resultado = backtrack({})

    print("Asignaciones intentadas:", estats["assignments"])
    print("Backtracks:", estats["backtracks"])
    return resultado


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """

    def revise(xi, xj, domains, assignment):
        revised = False
        to_remove = []

        for vi in domains[xi]:
            supported = False
            for vj in domains[xj]:
                temp = assignment.copy()
                temp[xi] = vi
                temp[xj] = vj

                if csp.is_consistent(xi, vi, temp) and csp.is_consistent(xj, vj, temp):
                    supported = True
                    break

            if not supported:
                to_remove.append(vi)

        for v in to_remove:
            domains[xi].remove(v)
            revised = True

        return revised


    def ac3(domains, assignment, queue=None):

        if queue is None:
            queue = deque(
                (xi, xj)
                for xi in csp.variables
                for xj in csp.get_neighbors(xi)
            )
        else:
            queue = deque(queue)

        while queue:
            xi, xj = queue.popleft()

            if revise(xi, xj, domains, assignment):

                if len(domains[xi]) == 0:
                    return False

                for xk in csp.get_neighbors(xi):
                    if xk != xj:
                        queue.append((xk, xi))

        return True


    def backtrack(assignment, domains):

        if csp.is_complete(assignment):
            return assignment

        var = csp.get_unassigned_variables(assignment)[0]

        for value in domains[var]:

            if csp.is_consistent(var, value, assignment):

                csp.assign(var, value, assignment)

                new_domains = deepcopy(domains)
                new_domains[var] = [value]

                queue = [(neighbor, var) for neighbor in csp.get_neighbors(var)]

                if ac3(new_domains, assignment, queue):
                    result = backtrack(assignment, new_domains)
                    if result is not None:
                        return result

                csp.unassign(var, assignment)

        return None


    domains = deepcopy(csp.domains)

    if not ac3(domains, {}):
        return None

    return backtrack({}, domains)


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """

    def forward_check(var, value, domains, assignment):

        for neighbor in csp.get_neighbors(var):
            if neighbor in assignment:
                continue

            to_remove = []

            for neighbor_value in domains[neighbor]:
                temp = assignment.copy()
                temp[var] = value
                temp[neighbor] = neighbor_value

                if not (
                        csp.is_consistent(var, value, temp)
                        and csp.is_consistent(neighbor, neighbor_value, temp)
                ):
                    to_remove.append(neighbor_value)

            for v in to_remove:
                domains[neighbor].remove(v)

            if len(domains[neighbor]) == 0:
                return False

        return True


    def select_unassigned_variable_mrv(assignment, domains):
        """
        Hacemos el MRV para la smallest domain
        """
        unassigned = csp.get_unassigned_variables(assignment)
        return min(unassigned, key=lambda var: len(domains[var]))


    def order_domain_values_lcv(var, assignment, domains):
        """
        LCV: menos eliminaciones mejor
        """
        def count_conflicts(value):
            eliminated = 0

            for neighbor in csp.get_neighbors(var):
                if neighbor in assignment:
                    continue

                for neighbor_value in domains[neighbor]:
                    temp = assignment.copy()
                    temp[var] = value
                    temp[neighbor] = neighbor_value

                    if not (
                            csp.is_consistent(var, value, temp)
                            and csp.is_consistent(neighbor, neighbor_value, temp)
                    ):
                        eliminated += 1

            return eliminated

        return sorted(domains[var], key=count_conflicts)


    def backtrack(assignment, domains):
        if csp.is_complete(assignment):
            return assignment

        var = select_unassigned_variable_mrv(assignment, domains)

        for value in order_domain_values_lcv(var, assignment, domains):
            if csp.is_consistent(var, value, assignment):
                csp.assign(var, value, assignment)

                new_domains = deepcopy(domains)
                new_domains[var] = [value]

                if forward_check(var, value, new_domains, assignment):
                    result = backtrack(assignment, new_domains)
                    if result is not None:
                        return result

                csp.unassign(var, assignment)

        return None


    domains = deepcopy(csp.domains)
    return backtrack({}, domains)
