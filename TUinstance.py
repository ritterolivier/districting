import numpy as np
import pandas as pd


class TUinstance():

    # Constructor
    def __init__(self, datafile=""):
        """
        Collect the informations from the input file to build the job-shop scheduling instance.
        """
        #Check that the file is an excel spreadsheet
        if datafile[-4:] == ".csv":
            #Get the informations on the processing times of each job
            columns = ["ID", "PRO_COM", "Ds1", "Ds2", "Ds3", "East", "North"]
            self._tu = pd.read_csv(datafile, header=None, names=columns, index_col = 0)
        else:
            raise NameError("Expected a '.xlsx' file")

    def get_processtimes(self):
        return self._tu

        
