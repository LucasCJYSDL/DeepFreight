from environment.path_object import PathFinder
from environment.matcher import match_packages, Graph
from environment import settings
import numpy as np
from environment.truck_generator import TruckGenerator
from environment.package_generator import PackageGenerator


class Simulator(object):
    def __init__(self):

        self.path_object = PathFinder(settings.ETA_FILE, settings.NUM_HOPS)
        self.node_list = self.path_object.getKeys()  # Create and initialize package array
        self.number_of_nodes = len(self.node_list)
        self.nodeToIndex, self.indexToNode = self.path_object.get_node2ind()
        self.truck_generator = TruckGenerator(settings.NUM_TRUCKS, settings.TRUCK_SIZE, self.path_object)
        self.package_generator = PackageGenerator(settings.NUM_REQUESTS_PER_DAY, settings.MIN_VOLUME, settings.MAX_VOLUME, self.path_object)


    def reset(self, episode):

        if episode%settings.DAYS_CYCLE==0:
            self.truck_list = self.truck_generator.getTruckData()
            self.truck_dict = {}
            for t in self.truck_list:
                # print(t)
                self.truck_dict[t.id] = t
        else:
            for t in self.truck_list:
                t.reset()

        self.package_list = self.package_generator.getPackageData()
        self.packageArray = []
        self.updatePackageArray()
        self.old_time = 0.
        self.old_fuel = 0.
        self.old_served = 0
        self.path_list = [[0 for _ in range(settings.NUM_HOPS)] for _ in range(settings.NUM_HOPS)]


    def updatePackageArray(self):
        """
        Initializes the package array with total capacity going from source to destination
        :return: None
        """
        self.packageArray = [[0 for _ in range(self.number_of_nodes)] for _ in range(self.number_of_nodes)]
        for p in self.package_list:
            self.packageArray[self.nodeToIndex[p.source]][self.nodeToIndex[p.destination]] += p.size
        for i in range(self.number_of_nodes):
            for j in range(self.number_of_nodes):
                self.packageArray[i][j] = self.packageArray[i][j] / float(settings.NUM_REQUESTS_PER_DAY) / float(settings.MAX_VOLUME) * 2.0

    def updateTruckArray(self):
        """
        Iterates through truck_list and counts number of vehicles at each node
        :return: None
        """
        truckArray = [0 for _ in range(self.number_of_nodes)]
        timeArray = []
        for t in self.truck_list:
            truckArray[self.nodeToIndex[t.location]] += 1
            time_left = (settings.SIMULATION_TIME - t.cumu_time)/float(settings.SIMULATION_TIME)
            timeArray.append(time_left)
        return truckArray, timeArray


    def get_state_obs(self, epoch_num):

        self.updatePackageArray()
        delivery_req = self.packageArray.copy()

        truck_array, time_array = self.updateTruckArray()
        # new_delivery_req, unserved_packages = self.match_delivery(delivery_req)
        new_delivery_req = delivery_req
        unserved_packages = [1 for _ in range(settings.NUM_HOPS)]
        for i in range(settings.NUM_HOPS):
            for j in range(settings.NUM_HOPS):
                if new_delivery_req[i][j]<1e-6:
                    new_delivery_req[i][j]=0.0

        # state = (delivery_req, truck_array, time_array)
        state_delivery = np.reshape(new_delivery_req, newshape=[-1])
        # state = np.concatenate((state_delivery, truck_array, time_array))
        if epoch_num > settings.NUM_HOPS - 1:
            epoch_num = settings.NUM_HOPS - 1
        epoch_array = np.zeros((settings.NUM_HOPS, ))
        epoch_array[epoch_num] = 1.
        state = np.concatenate((state_delivery, truck_array, epoch_array))

        obs_list, mask_list = [], []
        for t in self.truck_list:
            time_left = (settings.SIMULATION_TIME - t.cumu_time)/float(settings.SIMULATION_TIME)
            # obs = (new_delivery_req, self.nodeToIndex[t.location], time_left)
            # # obs = (delivery_req, self.nodeToIndex[t.location], time_left)
            obs_location = np.zeros([settings.NUM_HOPS])
            obs_location[self.nodeToIndex[t.location]] = 1.0
            # obs = np.concatenate((state_delivery, obs_location, [time_left]))
            obs = np.concatenate((state_delivery, obs_location, epoch_array))
            obs_list.append(obs)

            mask = t.get_mask(self.path_list)
            mask_list.append(mask)

        assert len(obs_list)==settings.NUM_TRUCKS, len(obs_list)
        assert len(mask_list)==settings.NUM_TRUCKS, len(mask_list)

        return state, np.array(obs_list), np.array(mask_list), unserved_packages

    def step(self, action_list, epoch_num):

        assert action_list.shape[0]==settings.NUM_TRUCKS, action_list.shape
        idx = 0
        for t in self.truck_list:
            act = action_list[idx]
            src = self.nodeToIndex[t.location]
            t.step(epoch_num, act)
            dst = self.nodeToIndex[t.location]
            if src != dst:
                self.path_list[src][dst] = 1
            idx += 1

    def match_delivery(self, delivery_req=None, is_final=False):

        if delivery_req==None:
            self.updatePackageArray()
            delivery_req = self.packageArray.copy()
        graph = Graph()
        t_list = self.truck_list.copy()
        for t in t_list:
            if len(t.edge_list)>0:
                for edge in t.edge_list:
                    edge.cap = settings.TRUCK_SIZE
                    graph.addEdge(edge)
        p_list = self.package_list.copy()
        delivery_dict, self.request_not_matched, new_delivery_req, unserved_packages = match_packages(graph, p_list, self.nodeToIndex, delivery_req)

        print("3: ", self.request_not_matched)
        if is_final:
            for t_id, t_val in delivery_dict.items():
                self.truck_dict[t_id].packages = self.truck_dict[t_id].packages.union(t_val)
                self.truck_dict[t_id].total_serviced = len(self.truck_dict[t_id].packages)

        return new_delivery_req, unserved_packages

    def get_reward(self):

        request_not_matched = self.request_not_matched
        empty_vehicle = 0
        for t in self.truck_list:
            t.get_stats()
            if len(t.packages)==0:
                empty_vehicle += 1
        empty_ratio = float(empty_vehicle)/settings.NUM_TRUCKS
        time_array = np.array([t.time_used for t in self.truck_list])
        fuel_array = np.array([t.fuel_consumption for t in self.truck_list])
        mean_time = time_array.mean()
        mean_fuel = fuel_array.mean()
        request_served = settings.NUM_REQUESTS_PER_DAY - request_not_matched
        # reward = settings.TIME_REWARD * mean_time + settings.FUEL_REWARD * mean_fuel + settings.REQUEST_REWARD * request_served
        reward = (settings.FUEL_REWARD * mean_fuel + settings.REQUEST_REWARD * request_served)
        return empty_ratio, reward, request_not_matched, mean_fuel
    
    def get_epoch_reward(self):

        request_not_matched = self.request_not_matched
        for t in self.truck_list:
            t.get_stats()
        time_array = np.array([t.time_used for t in self.truck_list])
        fuel_array = np.array([t.fuel_consumption for t in self.truck_list])
        mean_time = time_array.mean()
        mean_fuel = fuel_array.mean()
        request_served = settings.NUM_REQUESTS_PER_DAY - request_not_matched
        delta_time = mean_time - self.old_time
        delta_fuel = mean_fuel - self.old_fuel
        delta_served = request_served - self.old_served
        reward = settings.TIME_REWARD * delta_time + settings.FUEL_REWARD * delta_fuel + settings.REQUEST_REWARD * delta_served
        self.old_time = mean_time
        self.old_fuel = mean_fuel
        self.old_served = request_served
        # reward = settings.TIME_REWARD * mean_time + settings.FUEL_REWARD * mean_fuel + settings.REQUEST_REWARD * request_served
        return reward

    def is_done(self, mask_list, unserved_packages):

        assert len(mask_list)==settings.NUM_TRUCKS, len(mask_list)
        is_terminal = []
        for i in range(settings.NUM_TRUCKS):
            is_terminal.append(mask_list[i][-1])

        # print(is_terminal)
        done = ((max(is_terminal) == 0) or (max(unserved_packages) < 1e-6))

        return done


    def final_pack(self):
        from environment.matcher import Edge
        for t in self.truck_list:
            if len(t.edge_list) > 0:
                temp_edge_list = []
                for edge in t.edge_list:
                    if edge.cap < settings.TRUCK_SIZE:
                        temp_edge_list.append(edge)

                if len(temp_edge_list) == 0:
                    t.edge_list = temp_edge_list
                    t.location = t.source
                    t.cumu_time = 0
                    t.total_moving = 0
                    t.total_distance = 0
                    continue
                ## period
                for i in range(len(temp_edge_list) - 1):
                    if temp_edge_list[i].destination != temp_edge_list[i + 1].src:
                        temp_path = self.path_object.dijkstra(temp_edge_list[i].destination,
                                                              temp_edge_list[i + 1].src)
                        assert len(temp_path) >= 2, len(temp_path)
                        cumu_time = temp_edge_list[i].depart + temp_edge_list[i].eta
                        cumu_index = i + 1
                        for j in range(len(temp_path) - 1):
                            eta = self.path_object.getETA(temp_path[j], temp_path[j + 1])
                            new_edge = Edge(t.id, temp_path[j], temp_path[j + 1], eta, cumu_time,
                                            settings.TRUCK_SIZE)
                            cumu_time += eta
                            temp_edge_list.insert(cumu_index, new_edge)
                            cumu_index += 1
                ## start location
                if temp_edge_list[0].src != t.source:
                    assert temp_edge_list[0].depart > 0, temp_edge_list[0].depart
                    temp_path = self.path_object.dijkstra(t.source, temp_edge_list[0].src)
                    assert len(temp_path) >= 2, len(temp_path)
                    cumu_time, cumu_index = 0, 0
                    for i in range(len(temp_path) - 1):
                        eta = self.path_object.getETA(temp_path[i], temp_path[i + 1])
                        new_edge = Edge(t.id, temp_path[i], temp_path[i + 1], eta, cumu_time, settings.TRUCK_SIZE)
                        cumu_time += eta
                        temp_edge_list.insert(cumu_index, new_edge)
                        cumu_index += 1
                t.edge_list = temp_edge_list
                t.location = t.edge_list[-1].destination
                t.cumu_time = t.edge_list[-1].depart + t.edge_list[-1].eta
                total_moving, total_distance = 0, 0
                for e in t.edge_list:
                    total_moving += e.eta
                    total_distance += self.path_object.getDistance(e.src, e.destination)
                t.total_moving = total_moving
                t.total_dis = total_distance

    def get_env_info(self):
        env_info = {}
        env_info['n_actions'] = settings.NUM_OUTPUT
        env_info['n_agents'] = settings.NUM_TRUCKS
        env_info['state_shape'] = settings.NUM_HOPS * settings.NUM_HOPS + settings.NUM_HOPS + settings.NUM_HOPS
        env_info['obs_shape'] = settings.NUM_HOPS * settings.NUM_HOPS + settings.NUM_HOPS + settings.NUM_HOPS
        env_info['episode_limit'] = settings.NUM_HOPS

        return env_info