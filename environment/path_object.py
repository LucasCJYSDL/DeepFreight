import pickle
import random
import numpy as np

class PathFinder(object):
    def __init__(self, filename: str, num_loc=-1):

        self.dict = pickle.load(open(filename, "rb"))
        # self.key_list = [key for key in self.dict.keys()]
        self.key_list = [0, 1, 3, 4, 7, 9, 10, 23, 26, 27]
        self.population_list = [61342, 14657, 40085, 34313, 34478, 25597, 15560, 5160, 38654, 2777]
        if num_loc>0 and num_loc<=len(self.key_list):
            # self.key_list = random.sample(self.key_list, num_loc)
            self.key_list = self.key_list[:num_loc]
        self.key_list = sorted(self.key_list)

    def getETA(self, src: int, destination: int) -> float:

        return self.dict[src][destination][1]

    def getDistance(self, src: int, destination: int) -> float:

        return self.dict[src][destination][0]

    def getMapSize(self) -> int:
        #return len(self.dict)
        return len(self.key_list)

    def getKeys(self) -> list:

        return self.key_list

    def get_node2ind(self):
        nodeToIndex = {}
        indexToNode = {}
        for i in range(len(self.key_list)):
            nodeToIndex[self.key_list[i]] = i
            indexToNode[i] = self.key_list[i]

        return nodeToIndex, indexToNode

    def get_node_prob(self):

        node_prob = []
        sum_population = np.array(self.population_list).sum()
        for i in range(len(self.population_list)):
            node_prob.append(float(self.population_list[i])/sum_population)

        return node_prob

    def dijkstra(self, source, destination):
        node_list = {}
        for node in self.key_list:
            node_list[node] = {}
            node_list[node]['time'] = float('inf')
            node_list[node]['last_node'] = None
        node_list[source]['time'] = 0
        Q = list(node_list.keys())
        while len(Q) != 0:
            u = min(Q, key=lambda x: node_list[x]['time'])
            if u == destination:
                break
            Q.remove(u)
            for n in self.key_list:
                if n != u:
                    temp_time = node_list[u]['time'] + self.getETA(u, n)
                    if temp_time < node_list[n]['time']:
                        node_list[n]['time'] = temp_time
                        node_list[n]['last_node'] = u
        path_list = [destination]
        cur_node = destination
        while node_list[cur_node]['last_node'] != None:
            cur_node = node_list[cur_node]['last_node']
            path_list.insert(0, cur_node)

        return path_list

if __name__ == '__main__':
    pf = PathFinder('./inputs/eta_file', 10)
    # print(pf.getMapSize())
    # print(pf.getKeys())
    # print(pf.dict)
    # node2ind, ind2node = pf.get_node2ind()
    # print(node2ind)
    # key_file = pickle.load(open('./inputs/key_file', 'rb'))
    # for i in pf.key_list:
    #     print(key_file[i])
    # print(pf.get_node_prob())
    total_eta = 0
    for i in pf.key_list:
        for j in pf.key_list:
            total_eta += pf.getETA(i,j)
    print(total_eta)
    print(total_eta/90.0/3600.0)
    print(total_eta / 3600.0/ 20.0)