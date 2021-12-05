
import numpy as np
from environment import settings
from environment.package import Package


class PackageGenerator(object):
    def __init__(self, num_packages, min_package_size, max_package_size, path_object):

        self.path_obj = path_object
        self.key_list = self.path_obj.getKeys()
        self.node_probability = self.path_obj.get_node_prob()
        self.nodeToIndex, _ = self.path_obj.get_node2ind()

        if min_package_size > max_package_size:
            ValueError("Error in PackageGenerator Init! Min package size must be less than max package size")

        self.number_packages = num_packages  # number of packages to create
        self.package_min = min_package_size  # minimum package size (inclusive)
        self.package_max = max_package_size  # maximum package size (inclusive)

        self.destination_probability = {}
        # calculate probabilities for node selection
        self.createDestProbability()


    def getPackageData(self):
        """
        Generates pseudo-data for packages based off of the parameters in the constructor
        :return: List of Package Objects
        """
        package_list = []
        np.random.seed()
        virtual_list = []
        self.source_statistics = [0 for _ in range(settings.NUM_HOPS)]
        self.destination_statistics = [[0 for _ in range(settings.NUM_HOPS)] for _ in range(settings.NUM_HOPS)]
        for i in range(self.number_packages):
            size = np.random.randint(self.package_min, self.package_max)
            # destination = np.random.choice(self.key_list, p=self.node_probability)
            # source = np.random.choice(self.key_list, p=self.distance_probability[destination])
            source = np.random.choice(self.key_list, p=self.node_probability)
            destination = np.random.choice(self.key_list, p=self.destination_probability[source])
            temp_term = {'src': source, 'dst': destination, 'size': size}
            virtual_list.append(temp_term)
            package_list.append(Package(source, destination, size, self.path_obj))

            # calculate statistics
            self.source_statistics[self.nodeToIndex[source]] += 1
            self.destination_statistics[self.nodeToIndex[source]][self.nodeToIndex[destination]] += 1
        package_list.sort(key=lambda x: (x.source, x.destination, -x.size))

        return package_list


    def createDestProbability(self):

        for node_src in self.key_list:
            # set the probability to 1/(time^1/2)
            dist_array = [self.path_obj.getETA(node_src, node_des) ** .5 for node_des in self.key_list]
            dist_array = [1 / d if d > 0.0 else 0.0 for d in dist_array]
            dist_array = [dist_array[i]*self.node_probability[i] for i in range(len(self.node_probability))]
            # normalize the percentage to 1 and add it to list
            total = sum(dist_array)
            self.destination_probability[node_src] = [x / total for x in dist_array]
