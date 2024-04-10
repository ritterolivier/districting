import pulp
import numpy as np
import pandas as pd
from tqdm import tqdm
from Solution import *


class D_Demand(object):
    def __init__(self, tu, p, ouptut_message = None):
        # Initialize MILP model and instance
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

    def create_variables(self):
        self._x = pulp.LpVariable.dicts('TU_Assignation', 
                                        ((i, j) for i in self.tu.get_all_ids() for j in self.tu.get_all_ids()),
                                        cat=pulp.LpBinary)
        self._d_max = pulp.LpVariable.dicts("MaxDist", 
                                            (j for j in self.tu.get_all_ids()), 
                                            lowBound=0, cat=pulp.LpContinuous)
        self._d_overall_max = pulp.LpVariable("OverallMaxDist", lowBound=0, cat=pulp.LpContinuous)

    def create_objective(self):
        self._model += self._d_overall_max

    def create_constraints(self):
        for i in self.tu.get_all_ids():
            self._model += pulp.lpSum(self._x[i, j] for j in self.tu.get_all_ids()) == 1

        self._model += pulp.lpSum(self._x[j, j] for j in self.tu.get_all_ids()) == self.p

        for j in self.tu.get_all_ids():
            for i in self.tu.get_all_ids():
                self._model += self._d_max[j] >= self._x[i, j] * self.tu.get_distance_from_tu(i, j)
                self._model += self._d_overall_max >= self._d_max[j]

        for i in self.tu.get_all_ids():
            for j in self.tu.get_all_ids():
                self._model += self._x[i, j] <= self._x[j, j]
        


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
        sol = Solution(modelType='D_Distance')
        if self._model.status > 0:
            sol._tu = self.tu
            sol._of = pulp.value(self._model.objective)
            sol._alloc = {key: var.varValue for key, var in self._x.items() if var.varValue > 0}
            sol._p = self.p
            sol._mu = self.mu
            sol._scena = self.tu._scena
            sol._xij = self._x
            sol._output = self._message
        
        return sol
