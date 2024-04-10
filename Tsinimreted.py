import pulp
import numpy as np
import pandas as pd
from tqdm import tqdm
from Solution import *


class Tsinimreted(object):
    def __init__(self, tu, p, ouptut_message = None):
        self._model = pulp.LpProblem("Districting Problem", pulp.LpMinimize)
        self.tu = tu
        self.p = p
        self.mu = self.calculate_mu()
        self._message = ouptut_message


    def calculate_mu(self):
        sum = 0
        for id in self.tu.get_all_ids():
            sum += self.tu.get_demand(id)
        return sum/self.p

    def create_model(self):
        self.create_variables()
        self.create_objective()
        self.create_constraints()

    # Method for creating decision variables
    def create_variables(self):
        """
        Create the decision variables used in the model.
        """
        # Binary variables x_ijk to define if location i is served by center j
        self._x = pulp.LpVariable.dicts('TU Assignation', 
                                        ((i, j) for i in self.tu.get_all_ids() for j in self.tu.get_all_ids()),
                                        lowBound=0,cat=pulp.LpBinary)
        
        # Auxiliary variables for positive and negative deviations from average demand
        self.d_plus = pulp.LpVariable.dicts('Positive Deviation',
                                            (j for j in self.tu.get_all_ids()),
                                            lowBound=0, cat=pulp.LpContinuous)
        self.d_minus = pulp.LpVariable.dicts('Negative Deviation',
                                            (j for j in self.tu.get_all_ids()),
                                            lowBound=0, cat=pulp.LpContinuous)
                                        


    # Method for creating the objective function
    def create_objective(self):
        """
        Create the objective function of the model.
        """
        #1. Objective is to minimize Cost
        # Parameters for weighting the importance of demand variance vs. distance
        weight_demand = 10000
        weight_distance = 1
        self._model += (
        weight_demand * pulp.lpSum(self.d_plus[j] + self.d_minus[j] for j in self.tu.get_all_ids()) +
        weight_distance * pulp.lpSum(self._x[i, j] * self.tu.get_distance_from_tu(i, j)
                                     for i in self.tu.get_all_ids() for j in self.tu.get_all_ids()),
        "Weighted Objective"
    )


    # Method for creating the constraints
    def create_constraints(self):
        """
        Create the model constraints.
        """
        #2. Only one link between each TU and a district representative
        for i in self.tu.get_all_ids():
            self._model += pulp.lpSum(self._x[i, j] for j in self.tu.get_all_ids()) == 1

        #3. Number of Representatives of Districts = p
        self._model += pulp.lpSum(self._x[j,j] for j in self.tu.get_all_ids()) == self.p
        
        average_demand = self.calculate_mu()
        for j in self.tu.get_all_ids():
            self._model += pulp.lpSum(self._x[i,j] * self.tu.get_demand(i) for i in self.tu.get_all_ids()) - average_demand == (self.d_plus[j] - self.d_minus[j]), f"Demand Balance for District {j}"

        #5. Can only assign TU to representative of a district
        for i in self.tu.get_all_ids():
            for j in self.tu.get_all_ids():
                self._model += self._x[i,j] <= self._x[j,j]
        


    # Method for writing the MILP model to a file
    def write_milp(self):
        """
        Write the model to a file.
        """
        self._model.writeLP("Dist_milp.lp")

    # Method for solving the MILP model
    def solve_milp(self):
        """
        Solve the model using the chosen solver.
        """
        # Solve using the default solver used by PuLP
        #self._model.solve()

        #Solving using GUROBI or CPLEX (choose one of the two depending on which one is available)
        #Use the command pulp.pulpTestAll() to check which solvers are available on your machine

        #self._model.solve(pulp.GUROBI_CMD(options=[("MIPgap", 0.03)]))
        #self._model.solve(pulp.GUROBI(msg=1, gapRel=0.001))

        #self._model.solve(pulp.CPLEX_CMD(path = '/Users/d0li/Desktop/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex', msg=0))
        self._model.solve(pulp.CPLEX_PY(msg=1, gapRel=0.005))

    def get_solution(self):
        """
        Create a solution object from the decision variables computed.
        """
        sol = Solution(modelType='Tsinimreted')
        if self._model.status > 0:
            sol._tu = self.tu
            sol._of = pulp.value(self._model.objective)
            sol._alloc = {key: var.varValue for key, var in self._x.items() if var.varValue > 0.2}
            sol._p = self.p
            sol._mu = self.mu
            sol._scena = self.tu._scena
            sol._xij = self._x
            sol._output = self._message
        
        return sol
