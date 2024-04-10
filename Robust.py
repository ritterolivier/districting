import pulp
import numpy as np
import pandas as pd
from tqdm import tqdm
from Solution import *

class Robust(object):
    def __init__(self, tu, p, alpha = 0.05, high_demand_factor=1.2, low_demand_factor=0.8, ouptut_message = None):
        self._model = pulp.LpProblem("Districting Problem", pulp.LpMinimize)
        self.tu = tu
        self.p = p
        self.alpha = alpha
        self.high_demand_factor = high_demand_factor
        self.low_demand_factor = low_demand_factor
        self.high_demand  =  self.estimate_high_demand()
        self.low_demand  = self.estimate_low_demand()

        self.mu_high = self.calculate_mu('high')
        self.mu_low = self.calculate_mu('low')
        self.mu = self.calculate_mu('')
        self._message = ouptut_message

    def calculate_mu(self, type):
        sum = 0
        for id in self.tu.get_all_ids():
            if type == 'high':
                sum += self.high_demand[id]
            elif type == 'low':
                sum += self.low_demand[id]
            else:
                sum += self.tu.get_demand(id)
        return sum/self.p
    
    def estimate_high_demand(self):
        # Adjust the demand for each TU by a factor for high demand scenario
        high_demand = {tu_id: self.tu.get_demand(tu_id) * self.high_demand_factor for tu_id in self.tu.get_all_ids()}
        return high_demand
    
    def estimate_low_demand(self):
        # Adjust the demand for each TU by a factor for low demand scenario
        low_demand = {tu_id: self.tu.get_demand(tu_id) * self.low_demand_factor for tu_id in self.tu.get_all_ids()}
        return low_demand


    # Method for creating the model
    def create_model(self):
        # Define decision variables, objective function, and constraints
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


   # Method for creating the objective function
    def create_objective(self):
        """
        Create the objective function of the model.
        """
        #1. Objective is to minimize Cost
        self._model += pulp.lpSum(self._x[i, j] * self.tu.get_distance_from_tu(i,j) for i in self.tu.get_all_ids() for j in self.tu.get_all_ids())


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
        
        #4. Constrain the low demand per District
        for j in self.tu.get_all_ids():
            self._model += (1 - self.alpha) * self.mu_low * self._x[j,j] <= pulp.lpSum(self._x[i,j] * self.low_demand[i] for i in self.tu.get_all_ids())
            self._model += (1 + self.alpha) * self.mu_low * self._x[j,j] >= pulp.lpSum(self._x[i,j] * self.low_demand[i] for i in self.tu.get_all_ids())
        
        #4 bis. Constrain the high demand per District
        for j in self.tu.get_all_ids():
            self._model += (1 - self.alpha) * self.mu_high * self._x[j,j] <= pulp.lpSum(self._x[i,j] * self.high_demand[i] for i in self.tu.get_all_ids())
            self._model += (1 + self.alpha) * self.mu_high * self._x[j,j] >= pulp.lpSum(self._x[i,j] * self.high_demand[i] for i in self.tu.get_all_ids())


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

        #self._model.solve(pulp.CPLEX_CMD(path = '/Users/d0li/Desktop/CPLEX_Studio2211/cplex/bin/arm64_osx/cplex', msg=0))
        self._model.solve(pulp.CPLEX_CMD(msg=0))


    def get_solution(self):
        """
        Create a solution object from the decision variables computed.
        """
        sol = Solution(modelType='Robust')
        if self._model.status > 0:
            sol._tu = self.tu
            sol._of = pulp.value(self._model.objective)
            sol._alloc = {key: var.varValue for key, var in self._x.items() if var.varValue > 0}
            sol._p = self.p
            sol._mu = self.mu
            sol._scena = self.tu._scena
            sol._xij = self._x
            sol._obj = pulp.value(self._model.objective)
            sol._output = self._message
        
        return sol
