import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class Charts():

    def __init__(self, solution=None, tu=None, time=None, scena = None):
        self._solution = solution
        self._tu = tu
        self._time = time
        self._scena = scena
        self._modelType = solution.get_modelType()

    def plot_solution_det(self, show):
        tu_locations = self._tu.get_localisation_dict()

        fig, ax = plt.subplots(figsize=(10, 6))

        tu_rep_association = self._solution.get_tu_association()

        colors = plt.cm.get_cmap('tab20', len(self._solution.get_rep()))

        for rep_index, (rep, tus) in enumerate(tu_rep_association.items()):
            rep_color = colors(rep_index)  
            for tu in tus:
                lat, lon = tu_locations[tu]
                if tu == rep:
                    ax.scatter(lon, lat, color=rep_color, edgecolor='black', marker='^')
                    ax.text(lon, lat, str(tu), fontsize=9, fontweight='bold')
                else:
                    ax.scatter(lon, lat, color=rep_color, edgecolor='black')
                    ax.text(lon, lat, str(tu), fontsize=9)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Territorial Units and Their Representatives')
        plt.grid(True)

        filename = f'Charts/{self._solution.get_output()}_Cost{self._solution.get_of()}_Scena{self._scena}.png'
        fig.savefig(filename)
        if show == True:
            plt.show()
