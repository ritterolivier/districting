import pulp
import numpy as np
import pandas as pd
from tqdm import tqdm
from Solution import *

class Stochastic(object):
    def __init__(self, tu, p, alpha = 0.05, alloc = None, scenario = None, ouptut_message = None):
        self._model = pulp.LpProblem("Districting Problem", pulp.LpMinimize)
        self.tu = tu
        self.p = p
        self._cost_dic = self.fill_cost_dict()
        self.alpha = alpha
        self.alloc = alloc
        self._scena = scenario
        self._message = ouptut_message
        
        # Creating the probabilities dictionnary (1 = 20% below intermediate, 2 = intermediate, 3 = 20% above intermediate)
        self._probabilities = {
            1: {1: 1/3, 2: 1/3, 3: 1/3},  
            2: {1: 1/6, 2: 1/6, 3: 2/3}, 
            3: {1: 1/6, 2: 2/3, 3: 1/6},  
            4: {1: 2/3, 2: 1/6, 3: 1/6}  
        }

        self.mu = self.calculate_mu()

    def fill_cost_dict(self):
        to_return = {"h": {}, "g": {}}
        for j in self.tu.get_all_ids():
            to_return["h"][j] = 10000
            to_return["g"][j] = 10000

        return to_return

    def calculate_mu(self):
        total_demand = 0

        # Access the dictionary of probabilities for the current scenario
        current_probabilities = self._probabilities[self._scena]

        # Iterate over each scenario probability within the selected scenario
        for scenario, prob in current_probabilities.items():
            # Set the scenario in the TerritorialUnit instance to get the correct demand dict
            self.tu.create_demand_dict(scenario)

            # Calculate total demand for this scenario
            scenario_demand = sum(self.tu.get_demand(id) for id in self.tu.get_all_ids())
            
            # Weight the scenario's total demand by its probability and add to the total demand
            weighted_demand = scenario_demand * prob
            total_demand += weighted_demand

        # Compute the weighted average demand (mu_hat) and divide by p
        # This reflects the expected demand per center, based on the probabilities of each scenario
        mu_hat = total_demand / self.p
        #print("MU : ", mu_hat)
        return mu_hat





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
        
        # Continuous variables phi to define demand shortage in district j for each scenario s
        self._phi = pulp.LpVariable.dicts('Phi assignation', 
                                        ((j, s) for j in self.tu.get_all_ids() for s in self._probabilities[self._scena].keys()),
                                        lowBound=0,cat=pulp.LpContinuous)
        
        # Continuous variables psi to define demand surplus in district j for each scenario s
        self._psi = pulp.LpVariable.dicts('Psi assignation', 
                                        ((j, s) for j in self.tu.get_all_ids() for s in self._probabilities[self._scena].keys()),
                                        lowBound=0,cat=pulp.LpContinuous)


    # Method for creating the objective function
    def create_objective(self):
        """
        Create the objective function of the model.
        """
        # Objective is to minimize Cost
        self._model += (
            # Transportation Cost
            pulp.lpSum(self.tu.get_distance_from_tu(i, j) * self._x[i, j] for i in self.tu.get_all_ids() for j in self.tu.get_all_ids()) +
            
            # Surplus Cost, weighted by scenario probabilities
            pulp.lpSum(self._cost_dic["g"][j] * pulp.lpSum(prob * self._psi[j, s] for s, prob in self._probabilities[self._scena].items()) for j in self.tu.get_all_ids()) +
            
            # Shortage Cost, weighted by scenario probabilities
            pulp.lpSum(self._cost_dic["h"][j] * pulp.lpSum(prob * self._phi[j, s] for s, prob in self._probabilities[self._scena].items()) for j in self.tu.get_all_ids())
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

        #5. Can only assign TU to representative of a district
        for i in self.tu.get_all_ids():
            for j in self.tu.get_all_ids():
                self._model += self._x[i,j] <= self._x[j,j]

        #6. Districts representatives and TUs allocation are the same as the Deterministic model
        for j, tus in self.alloc.items():
            # Ensure the district representative j is set to 1
            self._model += self._x[j, j] == 1
            for tu in tus:
                self._model += self._x[tu, j] == 1


        # Iterate over each district j and scenario s
        for j in self.tu.get_all_ids():
            for s in self._probabilities[self._scena].keys():
                self.tu.create_demand_dict(s)
                # Calculate the scenario-specific total demand for district j
                scenario_specific_demand = pulp.lpSum(self._x[i, j] * self.tu.get_demand(i) for i in self.tu.get_all_ids())
                
                # Implementing the lower bound constraint
                self._model += (1 - self.alpha) * self.mu * self._x[j, j] <= scenario_specific_demand - self._psi[j, s] + self._phi[j, s]
                
                # Implementing the upper bound constraint
                self._model += scenario_specific_demand - self._psi[j, s] + self._phi[j, s] <= (1 + self.alpha) * self.mu * self._x[j, j]

            
        #14. Constraining value of phi to be positive (redondant with lowerbound = 0)
        for j in  self.tu.get_all_ids():
            for s in self._probabilities[self._scena].keys():
                self._model += self._phi[j, s] >= 0
            
        #15. Constraining value of psi to be positive (redondant with lowerbound = 0)
        for j in  self.tu.get_all_ids():
            for s in self._probabilities[self._scena].keys():
                self._model += self._psi[j, s] >= 0


        
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
        sol = Solution(modelType='Stochastic')

        if self._model.status > 0:
            sol._tu = self.tu
            sol._of = pulp.value(self._model.objective)
            sol._alloc = {key: var.varValue for key, var in self._x.items() if var.varValue > 0}
            sol._p = self.p
            sol._mu = self.mu
            sol._scena = self.tu._scena
            sol._output = self._message
        
        return sol