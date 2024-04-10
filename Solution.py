import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Solution:
    # constructor
    def __init__(self, modelType=None, tu=None, of=None, alloc=None, scena=None, p = None, mu = None, rep = None, xij = None, output = None):
        self._modelType = modelType
        self._tu = tu
        self._of = of
        self._alloc = alloc
        self._scena = scena
        self._p = p
        self._mu = mu
        self._rep = rep
        self._xij = xij
        self._output = output

    def get_scena(self):
        return self._scena
    
    def get_xij(self):
        return self._xij

    def get_modelType(self):
        return self._modelType
    
    def get_output(self):
        return self._output
    
    def get_of(self):
        return self._of
    
    def get_p(self):
        return self._p
    
    def get_mu(self):
        return self._mu

    def get_rep(self):
        return self._rep

    def get_representative(self):
        representative_id = []
        for (tu1, tu2), value in self._alloc.items():
            if tu1 == tu2 and value == 1:
                representative_id.append(tu1)
        self._rep = representative_id

    def get_tu_association(self):
        # Create a dict with centers as keys and the list of villages associated as values
        tu_dict = {}
        self.get_representative()
        for reps in self._rep :
            for (tu1, tu2), value in self._alloc.items():
                if value > 0:
                    if tu2 in tu_dict:
                        if tu1 not in tu_dict[tu2]:  # avoid duplicate entries
                            tu_dict[tu2].append(tu1)
                    else:
                        tu_dict[tu2] = [tu1]
        return tu_dict


    def to_print_det(self):
        tu_rep_association = self.get_tu_association()
        reps = self._rep

        print("Representatives : ", reps)
        for rep, tu in tu_rep_association.items():
            print(f"Representative {rep}: Territorial Units : {', '.join(map(str, tu))}") 