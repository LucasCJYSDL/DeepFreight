from environment.truck import Truck
import numpy as np
from environment import settings

# assume constant truck size
class TruckGenerator(object):
    def __init__(self, num_trucks, truck_size,  path_object):

        self.path_obj = path_object
        self.key_list = self.path_obj.getKeys()
        self.node_probability = self.path_obj.get_node_prob()
        self.nodeToIndex, _ = self.path_obj.get_node2ind()

        # Validate num_trucks and truck_size
        if num_trucks < 1:
            ValueError("Error in TruckGenerator! num_trucks must be greater than zero")
        self.number_trucks = num_trucks

        if truck_size < 1:
            ValueError("Error in TruckGenerator! truck_size must be greater than zero")
        self.truck_size = truck_size

    def getTruckData(self):
        """
        Generates pseudo-data for truck based off of the parameters in the constructor
        Truck ID's are only unique for this generated set
        :return: List of Truck objects
        """
        truck_list = []
        np.random.seed()
        self.source_distribution = [0 for _ in range(settings.NUM_HOPS)]
        virtual_list = []

        for i in range(self.number_trucks):
            loc = np.random.choice(self.key_list, p=self.node_probability)
            temp_term = {'id': i, 'size': self.truck_size, 'loc': loc}
            virtual_list.append(temp_term)
            truck_list.append(Truck(i, self.truck_size, loc, self.path_obj))

            # statistics
            self.source_distribution[self.nodeToIndex[loc]] += 1

        return truck_list


