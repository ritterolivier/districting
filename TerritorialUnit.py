import numpy as np
import pandas as pd
from TUinstance import *
import math
from scipy.spatial import distance


class TerritorialUnit(TUinstance):
    def __init__(self, datafile = "data.csv"):
        super().__init__(datafile)
        self._localisation_dict = self.create_localisation_dict()
        self._dist = self.create_distance_matrix()

    def to_print(self):
        print(self._tu)

    def print_dist(self):
        print(self._dist)

    def create_localisation_dict(self):
        localisation_dict = self._tu[["North", "East"]].apply(tuple, axis=1).to_dict()
        return localisation_dict
    
    def get_distanceMatrix(self):
        return self._dist

    def get_localisation_dict(self):
        return self._localisation_dict

    def create_demand_dict(self, scenario):
        if scenario == 1:
            good_col = "Ds1"
        elif scenario == 2:
            good_col = "Ds2"
        else : 
            good_col = "Ds3"

        self._demand_dict = self._tu[good_col].to_dict()
        self._scena = scenario

    def get_demand(self, id):
        return self._demand_dict.get(id)
    
    def get_demand_dict(self):
        return self._demand_dict

    def get_all_ids(self):
        return list(self._localisation_dict.keys())

    def get_distance_from_tu(self, tu1, tu2):
        return self._dist[tu1][tu2]

    def calculate_distance(self, point1, point2):
        north1, east1 = point1
        north2, east2 = point2
        distance = math.sqrt((east2 - east1) ** 2 + (north2 - north1) ** 2)
        return distance

    def haversine_distance(self, point1, point2):
        lat1, lon1 = point1
        lat2, lon2 = point2
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        return math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)) * 6371

    def create_distance_matrix(self):
        distance_dict = {}
        df = self._tu

        for tu_id1, tu_loc1 in self.get_localisation_dict().items():
            distance_dict[tu_id1] = {}
            for tu_id2, tu_loc2 in self.get_localisation_dict().items():
                    distance_dict[tu_id1][tu_id2] = self.calculate_distance(tu_loc1, tu_loc2)

        return distance_dict