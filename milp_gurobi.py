import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time

class MILP_GUROBI:
    def __init__(self, depot_dict, eta_matrix, demand_matrix, demand_num_matrix, capacity, time_limit, optimize_time):

        self.depot_dict = depot_dict
        self.eta_matrix = eta_matrix
        self.demand_matrix = demand_matrix
        self.demand_num_matrix = demand_num_matrix
        self.capacity = capacity
        self.time_limit = time_limit
        self.loc_num = len(eta_matrix)
        self.veh_num = len(depot_dict)
        self.initialzeLP(optimize_time)

    def initialzeLP(self, optimize_time):

        self.solver = gp.Model('BASELINE_1')
        self.solver.setParam('TimeLimit', optimize_time)
        x, r, s, v, u = [], [], [], [], []
        # x: trajectory; r: delivery record; s: index; v: current volume; u: vehicle usage;
        # Create decision variables
        for i in range(self.loc_num):
            x_sub_matrix, r_sub_matrix = [], []
            for j in range(self.loc_num):
                x_row, r_row = [], []
                for k in range(self.veh_num):
                    x_row.append(self.solver.addVar(name='x(' + str(i) + ',' + str(j) + ',' + str(k) + ')', vtype=GRB.BINARY, lb=0, ub=1))
                    r_row.append(self.solver.addVar(name='r(' + str(i) + ',' + str(j) + ',' + str(k) + ')', vtype=GRB.BINARY, lb=0, ub=1))
                x_sub_matrix.append(x_row)
                r_sub_matrix.append(r_row)
            x.append(x_sub_matrix)
            r.append(r_sub_matrix)
        for i in range(self.loc_num):
            s_row, v_row = [], []
            for k in range(self.veh_num):
                s_row.append(self.solver.addVar(name='s('+str(i)+','+str(k)+')', vtype=GRB.INTEGER, lb=0))
                v_row.append(self.solver.addVar(name='v('+str(i)+','+str(k)+')', vtype=GRB.INTEGER, lb=0))
            s.append(s_row)
            v.append(v_row)
        for k in range(self.veh_num):
            u.append(self.solver.addVar(name='u('+str(k)+')', vtype=GRB.BINARY, lb=0, ub=1))

        # mt = self.solver.addVar(name='max travel time of the fleet', vtype=GRB.INTEGER, lb=0, ub=self.time_limit)
        # Create objective
        omega_1 = 0.5
        omega_2 = 0.04
        omega_3 = 0.5

        term_1 = None
        for i in range(self.loc_num):
            for j in range(self.loc_num):
                seg_num = None
                for k in range(self.veh_num):
                    if seg_num == None:
                        seg_num = x[i][j][k]
                    else:
                        seg_num += x[i][j][k]
                if term_1 == None:
                    term_1 = self.eta_matrix[i][j] * seg_num
                else:
                    term_1 += self.eta_matrix[i][j] * seg_num
        term_2 = None
        for i in range(self.loc_num):
            for j in range(self.loc_num):
                seg_pass = None
                for k in range(self.veh_num):
                    if seg_pass == None:
                        seg_pass = r[i][j][k]
                    else:
                        seg_pass += r[i][j][k]
                if term_2 == None:
                    term_2 = self.demand_matrix[i][j] * (1-seg_pass)
                else:
                    term_2 += self.demand_matrix[i][j] * (1-seg_pass)

        objective = omega_1*term_1/float(60*60*self.veh_num) + omega_2*term_2
        # objective = omega_2*term_2 + omega_3*mt/float(60*60)
        self.solver.setObjective(objective, GRB.MINIMIZE)

        # constraint 1: no self-loop
        for i in range(self.loc_num):
            for k in range(self.veh_num):
                self.solver.addConstr(x[i][i][k] == 0)
                self.solver.addConstr(r[i][i][k] == 0)

        # constraint 2: time limit
        for k in range(self.veh_num):
            const_2 = None
            for i in range(self.loc_num):
                for j in range(self.loc_num):
                    if const_2==None:
                        const_2 = self.eta_matrix[i][j]*x[i][j][k]
                    else:
                        const_2 += self.eta_matrix[i][j]*x[i][j][k]
            self.solver.addConstr(const_2 <= u[k]*self.time_limit)
            # self.solver.addConstr(const_2 <= u[k]*mt)

        # constraint 3: same depot
        for k in range(self.veh_num):
            depot = self.depot_dict[k]
            const_3_1 = None
            const_3_2 = None
            for i in range(self.loc_num):
                if const_3_1 == None:
                    const_3_1 = x[i][depot][k]
                    const_3_2 = x[depot][i][k]
                else:
                    const_3_1 += x[i][depot][k]
                    const_3_2 += x[depot][i][k]
            self.solver.addConstr(const_3_1 == u[k])
            self.solver.addConstr(const_3_2 == u[k])

        # constraint 4: form a large tour
        for k in range(self.veh_num):
            for i in range(self.loc_num):
                const_4_1 = None
                const_4_2 = None
                for j in range(self.loc_num):
                    if const_4_1 == None:
                        const_4_1 = x[i][j][k]
                        const_4_2 = x[j][i][k]
                    else:
                        const_4_1 += x[i][j][k]
                        const_4_2 += x[j][i][k]
                self.solver.addConstr(const_4_1 == const_4_2)
                self.solver.addConstr(const_4_1 <= u[k])

        # constraint 5: subtour elimination
        for k in range(self.veh_num):
            depot = self.depot_dict[k]
            self.solver.addConstr(s[depot][k] == 0)
            for i in range(self.loc_num):
                self.solver.addConstr(s[i][k] >= 0)
                if i == depot:
                    continue
                for j in range(self.loc_num):
                    self.solver.addConstr(s[i][k] >= 1+s[j][k]-self.loc_num*(1-x[j][i][k]))

        # constraint 6: truck number limit
        for i in range(self.loc_num):
            for j in range(self.loc_num):
                const_6 = None
                for k in range(self.veh_num):
                    if const_6 == None:
                        const_6 = r[i][j][k]
                    else:
                        const_6 += r[i][j][k]
                self.solver.addConstr(const_6 <= 1)

        # constraint 7: i->j exists
        for i in range(self.loc_num):
            for k in range(self.veh_num):
                depot = self.depot_dict[k]
                for j in range(self.loc_num):
                    if j != depot:
                        self.solver.addConstr(r[i][j][k]*(s[j][k]-s[i][k]) >= 0)
                    const_7_1 = None
                    const_7_2 = None
                    for l in range(self.loc_num):
                        if const_7_1 == None:
                            const_7_1 = x[i][l][k]
                            const_7_2 = x[l][j][k]
                        else:
                            const_7_1 += x[i][l][k]
                            const_7_2 += x[l][j][k]
                    self.solver.addConstr(r[i][j][k] <= const_7_1*const_7_2)

        # constraint 8: capacity limit
        for k in range(self.veh_num):
            depot = self.depot_dict[k]
            const_8_1 = None
            for i in range(self.loc_num):
                if const_8_1 == None:
                    const_8_1 = r[depot][i][k]*self.demand_matrix[depot][i]
                else:
                    const_8_1 += r[depot][i][k] * self.demand_matrix[depot][i]
            self.solver.addConstr(v[depot][k] == const_8_1)
        for k in range(self.veh_num):
            depot = self.depot_dict[k]
            for i in range(self.loc_num):
                if i == depot:
                    continue
                const_8_2, const_8_3, const_8_4 = None, None, None
                for l in range(self.loc_num):
                    if const_8_2 == None:
                        const_8_2 = v[l][k]*x[l][i][k]
                        const_8_3 = r[i][l][k]*self.demand_matrix[i][l]
                        const_8_4 = r[l][i][k]*self.demand_matrix[l][i]
                    else:
                        const_8_2 += v[l][k] * x[l][i][k]
                        const_8_3 += r[i][l][k] * self.demand_matrix[i][l]
                        const_8_4 += r[l][i][k] * self.demand_matrix[l][i]
                self.solver.addConstr(v[i][k] == const_8_2+const_8_3-const_8_4)
            for i in range(self.loc_num):
                for k in range(self.veh_num):
                    self.solver.addConstr(v[i][k] >= 0)
                    self.solver.addConstr(v[i][k] <= u[k]*self.capacity)

    def solve(self):
        status = self.solver.optimize()
        print(status)

    def getResult(self):
        print("Objective value: ", self.solver.ObjVal)
        # for v in self.solver.getVars():
        #     print(v.varName, " = ", v.x)
        return self.solver

    def saveModel(self):
        temp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
        file_name = 'lp_model/milp_'+temp_time+'.lp'
        self.solver.write(file_name)

    def result_process(self, result):

        x = [[[0 for k in range(self.veh_num)] for j in range(self.loc_num)] for i in range(self.loc_num)]
        r = [[[0 for k in range(self.veh_num)] for j in range(self.loc_num)] for i in range(self.loc_num)]
        s = [[0 for k in range(self.veh_num)] for i in range(self.loc_num)]
        v = [[0 for k in range(self.veh_num)] for i in range(self.loc_num)]
        u = [0 for k in range(self.veh_num)]
        for var in result.getVars():
            if 'm' in var.varName:
                max_travel_time = var.x
            else:
                temp = (var.varName.split('(')[1].split(')')[0].split(','))
                if 'x' in var.varName:
                    x[int(temp[0])][int(temp[1])][int(temp[2])] = var.x
                elif 'r' in var.varName:
                    r[int(temp[0])][int(temp[1])][int(temp[2])] = var.x
                elif 's' in var.varName:
                    s[int(temp[0])][int(temp[1])] = var.x
                elif 'v' in var.varName:
                    v[int(temp[0])][int(temp[1])] = var.x
                elif 'u' in var.varName:
                    u[int(temp[0])] = var.x
        # print("max travel time:\n", max_travel_time/float(60*60))
        print("x:\n", x)
        print("r:\n", r)
        print("s:\n", s)
        print("v:\n", v)
        print("u:\n", u)

        truck_in_use = 0
        for k in range(self.veh_num):
            if u[k]>0:
                truck_in_use += 1
        empty_ratio = 1.0 - float(truck_in_use)/self.veh_num

        print("empty ratio: ", empty_ratio)

        for k in range(self.veh_num):
            if u[k]==1:
                print("Truck {} is in use ...".format(k))
                print("Path segments are listed as: ")
                for i in range(self.loc_num):
                    for j in range(self.loc_num):
                        if x[i][j][k] == 1:
                            print(i, " ", j)
                print("Delivery demands are listed as: ")
                for i in range(self.loc_num):
                    for j in range(self.loc_num):
                        if r[i][j][k] == 1:
                            print(i, " ", j)
                print("Indexes are listed as: ")
                for i in range(self.loc_num):
                    print("({}, {}): {}".format(i,k,s[i][k]))
                print("Volumes are listed as: ")
                for i in range(self.loc_num):
                    print("({}, {}): {}".format(i, k, v[i][k]))

        time_use = 0
        for k in range(self.veh_num):
            term_1 = 0
            for i in range(self.loc_num):
                for j in range(self.loc_num):
                    term_1 += self.eta_matrix[i][j] * x[i][j][k]
            print("The time use of truck {} is {}.".format(k, term_1))
            time_use += term_1
        time_use = float(time_use)/(60.0*60.0*self.veh_num)
        print("The total time use is {}.".format(time_use))

        package_miss = 0
        for i in range(self.loc_num):
            for j in range(self.loc_num):
                seg_pass = 0
                for k in range(self.veh_num):
                    seg_pass += r[i][j][k]
                package_miss += self.demand_num_matrix[i][j] * (1 - seg_pass)
        print("The total number of the package miss is {}.".format(package_miss))

        # return empty_ratio, package_miss, time_use, max_travel_time/float(60*60)
        return empty_ratio, package_miss, time_use


if __name__ == '__main__':
    capacity = 30000
    time_limit = 48*60*60
    eta_matrix = [[0, 38347, 43615, 48257, 36955, 38083, 33276, 8899, 14592, 10255], \
                  [38082, 0, 41052, 31044, 30720, 20654, 26231, 43170, 23958, 30082], \
                  [43569, 41061, 0, 13532, 11564, 22238, 17423, 40069, 46532, 37097], \
                  [48337, 31181, 13686, 0, 12085, 18985, 18440, 47865, 45449, 41817], \
                  [36941, 30693, 11603, 11891, 0, 11870, 7037, 36469, 37083, 30421], \
                  [37900, 20608, 22148, 18827, 11816, 0, 9759, 41082, 33419, 31379], \
                  [33211, 26244, 17352, 18326, 7023, 9760, 0, 33039, 31142, 26691], \
                  [8777, 43217, 40091, 47576, 36274, 41121, 32889, 0, 22011, 13426], \
                  [14532, 24076, 46563, 45561, 37317, 33691, 31251, 21969, 0, 10930], \
                  [9974, 30117, 36984, 41522, 30221, 31348, 26542, 13336, 10848, 0]]
    demand_matrix = [[0, 4107, 9985, 8651, 9447, 7597, 4574, 2909, 18498, 1285], \
                     [3677, 0, 2205, 2029, 2704, 1909, 1005, 267, 2791, 237], \
                     [9416, 2275, 0, 8687, 8657, 5120, 3650, 778, 5159, 452], \
                     [6408, 2165, 8373, 0, 7573, 4533, 2857, 664, 4352, 397], \
                     [6177, 1804, 8671, 7140, 0, 4816, 4071, 597, 4209, 394], \
                     [4652, 1685, 4191, 4259, 5531, 0, 2449, 319, 3590, 199], \
                     [2829, 837, 2551, 2718, 3613, 2359, 0, 273, 2145, 150], \
                     [1828, 232, 619, 607, 616, 346, 352, 0, 925, 88], \
                     [15533, 3111, 5682, 5158, 5536, 3682, 2527, 1209, 0, 805], \
                     [1105, 77, 352, 192, 227, 185, 132, 104, 702, 0]]
    depot_dict = {0: 0, 1: 4, 2: 8, 3: 5, 4: 4, 5: 0, 6: 0, 7: 5, 8: 8, 9: 2, 10: 3, 11: 8, 12: 8, 13: 3, 14: 8, 15: 8, 16: 4, 17: 1, 18: 5, 19: 2}

    optimize_time = 3000
    milp = MILP_GUROBI(depot_dict, eta_matrix, demand_matrix, capacity, time_limit, optimize_time)
    milp.solve()
    result = milp.getResult()
    milp.result_process(result)
    # milp.saveModel()
