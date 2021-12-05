from environment import settings
import numpy as np

INF = float('inf')  # Represents infinity distance away for dijkstra
# defining Edge class, it will be used in Graph class
class Edge:
    def __init__(self, v_id: int, src: int, destination: int, travel_time: float, depart: float, cap: int):

        self.vehicle = v_id
        self.src = src
        self.destination = destination
        self.eta = travel_time
        self.depart = depart
        self.cap = cap

    def available(self, curr_time: float, size: int) -> bool:
        return curr_time <= self.depart and size <= self.cap

    def __str__(self) -> str:
        return "Vehicle {}; {} -> {}, depart: {}; cap {}; eta {};\n".format(self.vehicle, self.src, self.destination,
                                                                            self.depart, self.cap, self.eta)

# defining Node class, it will be used in Graph class
class Node:
    def __init__(self):
        """
        Creates Node object with no edges
        """
        self.edges = []
        # Following are used for dijkstra
        self.time = INF
        self.prev_edge = None
        self.label = 0

    def reset(self):
        """
        Resets the variables used in dijkstra
        :return: None
        """
        self.time = INF
        self.prev_edge = None
        self.label = 0

    def addEdge(self, edge: Edge):
        self.edges.append(edge)

    def __iter__(self):
        return iter(self.edges)

# defining Graph class using Edge and Node classes
class Graph:
    def __init__(self):
        self.node_list = {}
        self.edge_list = []

    def addEdge(self, new_edge: Edge) -> None:
        """
        adds an existing edge object to the graph
        :param new_edge: An edge object to be added to the graph
        :return: None
        """
        if new_edge.destination not in self.node_list:
            self.node_list[new_edge.destination] = Node()
        if new_edge.src not in self.node_list:
            self.node_list[new_edge.src] = Node()

        self.edge_list.append(new_edge)
        self.node_list[new_edge.src].addEdge(new_edge)

    def find_all_path(self, src, dst, size):

        path_list = []
        if src not in self.node_list:
            return path_list
        self.reset()
        path = [Edge(0, 0, 0, 0, 0, 0)]
        self.node_list[src].time = 0

        def dfs(cur_node):
            # print(len(path))
            # for p in path:
            #     print(p)
            # print("#############")
            if cur_node == dst:
                path_list.append(path[1:])
                path.pop()
                self.node_list[cur_node].time = INF
                return
            self.node_list[cur_node].label = 1
            for e in self.node_list[cur_node].edges:
                # print(e)
                # print("$$$$$$$$$$")
                temp_depart = e.depart
                temp_dst = e.destination
                temp_cap = e.cap
                if (self.node_list[cur_node].time <= temp_depart) and \
                        (self.node_list[temp_dst].label == 0) and (size <= temp_cap):
                    path.append(e)
                    self.node_list[temp_dst].time = temp_depart + e.eta
                    dfs(temp_dst)
            self.node_list[cur_node].label = 0
            self.node_list[cur_node].time = INF
            path.pop()

        dfs(src)
        path_to_return = []
        path_num = len(path_list)
        # print("number of possible path: ", path_num)
        if path_num == 0:
            return path_to_return
        time_cost_list = []
        for i in range(path_num):
            temp_path = path_list[i]
            time_cost = 0
            time_total = 0
            # print("path #{}:".format(i))
            for e in temp_path:
                # print("edge: ", e)
                # time_total += e.eta
                if e.cap == settings.TRUCK_SIZE:
                    time_cost += e.eta
            time_total = temp_path[-1].depart + temp_path[-1].eta
            time_cost_list.append((i, time_cost, time_total))
        time_cost_list.sort(key=lambda x: (x[1], x[2]))
        shortest_path = path_list[time_cost_list[0][0]]
        for e in shortest_path:
            # print("edge to choose: ", e)
            e.cap -= size
            path_to_return.append((e.vehicle, e.src, e.destination))

        return path_to_return

    def __str__(self) -> str:
        res = ""
        for e in self.edge_list:
            res += str(e)
        return res

    def reset(self):
        """
        Resets all the dijkstra related variables for all nodes in the graph
        :return: None
        """
        for key, n in self.node_list.items():
            n.reset()


# *******************MAIN FUNCTION USED FOR MATCHING********************
def match_packages(graph, p_list, nodeToIndex, delivery_req):

    v_dict = {}
    match_num = 0
    unserved_packages = [0 for _ in range(settings.NUM_HOPS)]
    new_delivery_req = delivery_req.copy()
    for i in range(len(p_list)):
        pack = p_list[i]

        if pack.source in graph.node_list and pack.destination in graph.node_list:

            package_path = graph.find_all_path(pack.source, pack.destination, pack.size)
            if len(package_path)>0:
                match_num += 1
                new_delivery_req[nodeToIndex[pack.source]][nodeToIndex[pack.destination]] -= pack.size/float(settings.NUM_REQUESTS_PER_DAY)/float(settings.MAX_VOLUME)*2.0
            else:
                unserved_packages[nodeToIndex[pack.source]] += pack.size
            for v_id, source, destination in package_path:
                if v_id not in v_dict:
                    v_dict[v_id] = set()
                v_dict[v_id].add(pack)
        else:
            unserved_packages[nodeToIndex[pack.source]] += pack.size

    unserved_packages = np.array(unserved_packages) / float(settings.NUM_REQUESTS_PER_DAY)

    return v_dict, len(p_list)-match_num, new_delivery_req, unserved_packages.tolist()
