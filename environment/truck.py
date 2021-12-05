#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from environment import settings
from environment.matcher import Edge

"defining a class for truck" 
class Truck(object):
    def __init__(self, vehicle_id, cap, location, path_object):
        # truck info
        self.id = vehicle_id            # id that is unique to each truck
        self.max_capacity = cap         # maximum capacity of the truck
        self.capacity = cap             # current capacity (used in package matching)
        self.path_obj = path_object
        self.key_list = self.path_obj.getKeys()
        self.nodeToIndex, self.indexToNode = self.path_obj.get_node2ind()
        self.cumu_time = 0             # time when starting trip
        self.location = location        # current location
        self.source = location

        # after execution info
        self.path = [location]          # history of where the truck has been
        self.packages = set()           # set of all packages serviced
        self.total_serviced = 0         # total packages serviced (used in reward equation)
        self.total_dis = 0              # total distance traveled (used in reward equation)
        # self.total_idle = 0             # total time the car was idle (only for stat purposes)
        self.total_moving = 0           # total time the car was moving
        # self.total_wait = 0             # total time the car was idle (only for stat purposes)
        self.edge_list = []
        self.is_terminal = False

    def reset(self):
        self.cumu_time = 0
        self.edge_list = []
        self.is_terminal = False
        self.total_dis = 0
        self.total_serviced = 0
        self.total_moving = 0
        self.packages = set()
        self.source = self.location
        self.path = [self.location]
        self.capacity = self.max_capacity

    def step(self, epoch_num, action):

        if int(action)==settings.NUM_OUTPUT-1:
            assert self.is_terminal==True
            return

        assert int(action)<settings.NUM_OUTPUT-1, int(action)
        wait_time = (int(action)//settings.NUM_HOPS)*settings.WAIT_TIME_INTERVAL
        assert int(action)//settings.NUM_HOPS < settings.NUM_WAIT_TIME
        destination = self.key_list[int(action)%settings.NUM_HOPS]
        self.wait(wait_time)
        self.move(destination)
        if self.is_terminal:
            self.cumu_time -= wait_time

        else:
            if (len(self.path)>1) and (self.path[0]==self.path[-1]):
                self.is_terminal = True
            if epoch_num >= settings.NUM_HOPS - 1:
                self.is_terminal = True

    def move(self, destination):
        # validate input parameters
        eta = self.path_obj.getETA(self.location, destination)
        distance = self.path_obj.getDistance(self.location, destination)

        # assign current trip info and change status
        if self.cumu_time + eta > settings.SIMULATION_TIME:
            self.is_terminal = True
            if not settings.SILENT:
                print("Truck {0} terminates in {1} at {2}".format(self.id, self.location, self.cumu_time + eta))

        else:
            new_edge = Edge(self.id, self.location, destination, eta, self.cumu_time, self.max_capacity)
            self.edge_list.append(new_edge)
            self.path.append(destination)
            self.cumu_time += eta
            if not settings.SILENT:
                print("Truck {0} move {1} -> {2}".format(self.id, self.location, destination))
            self.location = destination
            self.total_dis += distance
            self.total_moving += eta


    def wait(self, wait_time):
        if wait_time < 0:
            print("Error! Truck ID {},  Wait time cannot be less than zero".format(self.id))
            return

        self.cumu_time += wait_time

    def get_stats(self):
        time_used = self.cumu_time / (60.0 * 60.0)
        fuel_consumption = self.total_moving / (60.0 * 60.0) * settings.FUEL_PER_HOUR

        self.time_used = time_used
        self.fuel_consumption = fuel_consumption

    def show_info(self):
        # print("The packages delivered by {0} are listed as {1}".format(self.id, self.packages))
        print("The trajectory of {0} are listed as: ".format(self.id))
        for edge in self.edge_list:
            print(edge)

    def get_mask(self, path_list):

        if not self.is_terminal:
            mask = [0 for _ in range(settings.NUM_HOPS+1)]
            mask[-1] = 1
            if len(self.path)>1:
                for i in range(1, len(self.path)):
                    temp_ind = self.nodeToIndex[self.path[i]]
                    mask[temp_ind] = 1
        else:
            mask = [1 for _ in range(settings.NUM_HOPS + 1)]
            mask[-1] = 0

        if min(mask) > 0:
            mask[-1] = 0
            self.is_terminal = True

        return mask
