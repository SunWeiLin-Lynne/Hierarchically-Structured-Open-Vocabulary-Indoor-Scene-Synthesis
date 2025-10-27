import copy

import ipdb
import random
import gurobipy as gp
from gurobipy import *
import difflib
import math

from utils.visualize_topview import visualize_area_topview, visualize_area_obj_topview, visualize_all_area_obj_topview
from utils.parse_relative_position import get_close_scores
from utils.utils import get_3d_bdbox, OBJ_LIST, get_2d_bdbox, roty_3d
import numpy as np
ALL_RELATIONS = ["right of", "right touching", "left of", "left touching", "behind", "behind touching", "in front of",
                 "front touching", "right front of", "right front touching", "left front of",
                 "left front touching", "right behind of", "right behind touching", "left behind of",
                 "left behind touching", "above", "on top of", "below"]

# toolkit
def load_gpt_output(input):
    O_func = [] #[{label,bdb_x,bdb_y,bdb_z,bdb_l,bdb_w,bdb_h,theta},..]
    O_other = [] #[{bdb_x,bdb_y,bdb_z,bdb_l,bdb_w,bdb_h,theta,rel_pos, func_obj_index},..]
    rl, rw = float(input['length']), float(input['width'])
    for area in input['children']:
        if 'children' in area.keys():
            func_obj = area['children']
            func = {
                'label': func_obj['label'],
                'length': func_obj['length'],
                'width': func_obj['width'],
                'height': func_obj['height'],
                'obj_index': len(O_func),
                'on_the_ceiling': False,
                'description': func_obj['description']
            }
            if 'orientation' in func_obj.keys():
                func['orientation'] = func_obj['orientation']
            if 'rel_pos_constraints' in area.keys():
                func['rel_pos_constraint'] = area['rel_pos_constraints']
            O_func.append(func)
            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    other = {
                        'label': obj['label'],
                        'length': obj['length'],
                        'width': obj['width'],
                        'height': obj['height'],
                        'rel_position': obj['relative position'],
                        'func_obj_index': len(O_func) - 1,
                        'obj_index': len(O_other),
                        'on_the_ceiling': False,
                        'description': obj['description']
                    }
                    if 'orientation' in obj.keys():
                        other['orientation'] = obj['orientation']
                    O_other.append(other)
    return rl, rw, O_func, O_other


def add_optimization_for_traversal_merge_area_by_every_area(area, inform = None, type = 'one-time'):
    if type == 'one-time' or type == 'generate':
        area['optimization'] = ['area_local']
    elif type == 'modify':
        if area['modify_type'] == '(#)':
            area['optimization'] = ['area_local']
        else:
            area['optimization'] = []

    return area


def add_optimization_for_traversal_merge_area_by_every_obj(area, inform = None, type = 'one-time'):
    if type == 'one-time' or type == 'generate':
        if 'children' in area.keys():
            func_obj = area['children']
            func_obj['optimization'] = ['obj_global']
            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    obj['optimization'] = ['obj_global']
    elif type == 'modify':
        if 'children' in area.keys():
            func_obj = area['children']
            if func_obj['modify_type'] == '(#)' or area['modify_type'] == '(#)':
                func_obj['optimization'] = ['obj_global']
            else:
                func_obj['optimization'] = []
            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    if obj['modify_type'] == '(#)' or func_obj['modify_type'] == '(#)' or area['modify_type'] == '(#)':
                        obj['optimization'] = ['obj_global']
                    else:
                        obj['optimization'] = []
    return area


def add_optimization_for_merge_area(data, inform = None, type = 'one-time'):
    '''
    Args:
        data: hierarchy data(tree structure)
        inform: changed area/obj
        type: 'one-time', 'delete-area', 'delete-obj', 'add-area', 'add-obj', 'modify-area', 'modify-obj',

    Returns:
    '''
    if type == 'one-time' or type == 'generate':
        if 'children' in data.keys():
            for area in data['children']:
                area['optimization'] = ['area_global']
                # area['optimization'].append('area_global')
                if 'children' in area.keys():
                    func_obj = area['children']
                    func_obj['optimization'] = ['obj_local']
                    # func_obj['optimization'].append('obj_local')
                    if 'children' in func_obj.keys():
                        for obj in func_obj['children']:
                            obj['optimization'] = ['obj_local']
                            # obj['optimization'].append('obj_local')
    elif type == 'modify':
        if 'children' in data.keys():
            for area in data['children']:
                if area['modify_type'] == '(#)':
                    area['optimization'] = ['area_global']
                    # area['optimization'].append('area_global')
                else:
                    area['optimization'] = []
                if 'children' in area.keys():
                    func_obj = area['children']
                    if func_obj['modify_type'] == '(#)' or area['modify_type'] == '(#)':
                        func_obj['optimization'] = ['obj_local']
                        # func_obj['optimization'].append('obj_local')
                    else:
                        func_obj['optimization'] = []
                    if 'children' in func_obj.keys():
                        for obj in func_obj['children']:
                            if obj['modify_type'] == '(#)' or func_obj['modify_type'] == '(#)' or area['modify_type'] == '(#)':
                                obj['optimization'] = ['obj_local']
                                # obj['optimization'].append('obj_local')
                            else:
                                obj['optimization'] = []
    return data


def load_area_gpt_output(area):
    O_func = [] #[{label,bdb_x,bdb_y,bdb_z,bdb_l,bdb_w,bdb_h,theta},..]# bdb_l,bdb_w,bdb_h为3Dbounding的长宽高
    O_other = [] #[{bdb_x,bdb_y,bdb_z,bdb_l,bdb_w,bdb_h,theta,rel_pos, func_obj_index},..]
    if 'children' in area.keys():
        func_obj = area['children']
        func = {
            'label': func_obj['label'],
            'length': func_obj['length'],
            'width': func_obj['width'],
            'height': func_obj['height'],
            'obj_index': len(O_func),
            'on_the_ceiling': False,
            'description': func_obj['description']
        }
        if 'orientation' in func_obj.keys():
            func['orientation'] = func_obj['orientation']
        if 'optimization' in func_obj.keys():
            if len(func_obj['optimization']) !=0 and (func_obj['optimization'][0] == 'obj_local' or func_obj['optimization'][0] == 'obj_global'):
                func['is_optimized'] = True
            else:
                func['bdb_x'] = func_obj['bdb_x']
                func['bdb_y'] = func_obj['bdb_y']
                func['bdb_z'] = func_obj['bdb_z']
                func['bdb_l'] = func_obj['bdb_l']
                func['bdb_w'] = func_obj['bdb_w']
                func['bdb_h'] = func_obj['bdb_h']
                func['bdb_ori'] = func_obj['bdb_ori']
                func['is_optimized'] = False
        O_func.append(func)
        if 'children' in func_obj.keys():
            for obj in func_obj['children']:
                other = {
                    'label': obj['label'],
                    'length': obj['length'],
                    'width': obj['width'],
                    'height': obj['height'],
                    'rel_position': obj['relative position'],
                    'func_obj_index': len(O_func) - 1,
                    'obj_index': len(O_other),
                    'on_the_ceiling': False,
                    'description': obj['description']
                }
                if 'orientation' in obj.keys():
                    other['orientation'] = obj['orientation']
                if 'optimization' in obj.keys():
                    if len(obj['optimization']) != 0 and (obj['optimization'][0] == 'obj_local' or obj['optimization'][0] == 'obj_global'):
                        other['is_optimized'] = True
                    else:
                        other['bdb_x'] = obj['bdb_x']
                        other['bdb_y'] = obj['bdb_y']
                        other['bdb_z'] = obj['bdb_z']
                        other['bdb_l'] = obj['bdb_l']
                        other['bdb_w'] = obj['bdb_w']
                        other['bdb_h'] = obj['bdb_h']
                        other['bdb_ori'] = obj['bdb_ori']
                        other['is_optimized'] = False
                O_other.append(other)
    return O_func, O_other


def get_bdbox_l_w_h(o_i):
    new_o_i = copy.deepcopy(o_i)
    new_o_i['orientation'] = o_i['bdb_ori']
    bdb3d_corner = get_3d_bdbox(new_o_i)
    bdb_3d = [min(bdb3d_corner[:, 0]), min(bdb3d_corner[:, 1]), min(bdb3d_corner[:, 2]), max(bdb3d_corner[:, 0]), max(bdb3d_corner[:, 1]), max(bdb3d_corner[:, 2])]

    return round(abs(bdb_3d[3] - bdb_3d[0]), 2), round(abs(bdb_3d[4] - bdb_3d[1]),2),round(abs(bdb_3d[5] - bdb_3d[2]),2)


def get_2dbdbox_l_w(o_i):
    new_o_i = copy.deepcopy(o_i)
    new_o_i['orientation'] = o_i['bdb_ori']
    bdb2d_corner = get_2d_bdbox(new_o_i)
    bdb_2d = [min(bdb2d_corner[:, 0]), min(bdb2d_corner[:, 1]), max(bdb2d_corner[:, 0]), max(bdb2d_corner[:, 1])]
    return round(abs(bdb_2d[2] - bdb_2d[0]), 2), round(abs(bdb_2d[3] - bdb_2d[1]),2)


def get_area_length_width(obj_list):
    obj_bdbs = []
    for obj in obj_list:
        new_obj = copy.deepcopy(obj)
        new_obj['orientation'] = obj['bdb_ori']
        new_obj['left'] = obj['bdb_x']
        new_obj['top'] = obj['bdb_y']
        new_obj['length'] = obj['length']
        new_obj['width'] = obj['width']
        bdbox = get_2d_bdbox(new_obj)
        obj_bdbs.append(bdbox)
    area_bdb = [[round(min(np.array(obj_bdbs)[:, 0, 0].tolist() + np.array(obj_bdbs)[:, 3, 0].tolist()), 2),
                 round(min(np.array(obj_bdbs)[:, 0, 1].tolist() + np.array(obj_bdbs)[:, 3, 1].tolist()), 2)],
                [round(max(np.array(obj_bdbs)[:, 0, 0].tolist() + np.array(obj_bdbs)[:, 3, 0].tolist()), 2),
                 round(max(np.array(obj_bdbs)[:, 0, 1].tolist() + np.array(obj_bdbs)[:, 3, 1].tolist()), 2)]]
    return round(abs(area_bdb[0][0] - area_bdb[1][0]), 2), round(abs(area_bdb[0][1] - area_bdb[1][1]), 2), round((area_bdb[0][0] + area_bdb[1][0]) / 2, 3), round((area_bdb[0][1] + area_bdb[1][1]) / 2, 3)

    # area_data = {
    #     'length':round(abs(area_bdb[0][0] - area_bdb[1][0]), 2),
    #     'width':round(abs(area_bdb[0][1] - area_bdb[1][1]), 2),
    #     'bdb_ori':obj_list[0]['bdb_ori']
    # }
    # area_l, area_w = get_2dbdbox_l_w(area_data)
    # area_left, area_top = area_data['length']/2, area_data['width']/2
    #
    # return area_l, area_w, area_left, area_top


def use_rel_pos_constraint(o_i, constraints, room_type):
    o_1, o_r, o_2 = o_i['rel_position'][0], o_i['rel_position'][1], o_i['rel_position'][2]
    if constraints is None:
        return False, None
    else:
        bin = 0
        constraint = None
        for c in constraints:
            c_o_1 = c['cor_pred_obj']
            c_o_2 = c['cor_pred_func_obj']
            c_o_r = c['relative_position'][1]
            if c_o_1 == o_1.split(" ")[0] and c_o_2 == o_2.split(" ")[0] and c_o_r == o_r: 
                constraint = c
                return True, constraint
            if c_o_1 == o_1 and c_o_2 == o_2:
                bin = 1
                constraint = c
        if bin == 1:
            rel_similarity = get_close_scores(o_r, constraint['relative_position'][1], 1, cutoff=0)
            o_o_similarity = get_close_scores(o_1, constraint['relative_position'][0], 1, cutoff=0) 
            f_o_similarity = get_close_scores(o_2, constraint['relative_position'][2], 1, cutoff=0) 
            if rel_similarity >= 0.6 or (o_o_similarity >= 0.6 and f_o_similarity >= 0.6):
                return True, constraint
            else:
                return False, None
        return False, None


def parse_gpt_output_step_by_step(all_P, type = 'galo'):
    if type == 'local' or type == 'global':
        object_list = []
        for obj in all_P:
            res_obj = [obj["label"].split(" ")[0],
                       {
                           "length": obj["length"],
                           "width": obj["width"],
                           "height": obj["height"],
                           "left": obj["bdb_x"],
                           "top": obj["bdb_y"],
                           "depth": obj["bdb_z"],
                           "orientation": obj["bdb_ori"],
                           "description": obj['description']
                       }]
            object_list.append(res_obj)
    else:
        object_list = []
        for obj in all_P:
            res_obj = [obj["label"].split(" ")[0],
                       {
                           "length": obj["length"],
                           "width": obj["width"],
                           "height": obj["height"],
                           "left": obj["bdb_x"],
                           "top": obj["bdb_y"],
                           "depth": obj["bdb_z"],
                           "bdb_ori": obj["bdb_ori"],
                           "bdb_x": obj["bdb_x"],
                           "bdb_y": obj["bdb_y"],
                           "bdb_z": obj["bdb_z"],
                           "bdb_l": obj["bdb_l"],
                           "bdb_w": obj["bdb_w"],
                           "bdb_h": obj["bdb_h"],
                           "is_optimized": obj["is_optimized"],
                           "description": obj['description']
                       }]
            object_list.append(res_obj)


    return object_list


def optimize_local_func_obj(o_i, P, rl, rw):
    # Create a new model
    m = gp.Model()
    # Create variables
    x_i = m.addVar(lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="x")
    y_i = m.addVar(lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="y")
    ori_i = m.addVar(vtype=gp.GRB.INTEGER, name="ori_func")
    l_i = m.addVar(lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="bdb_l_func")
    w_i = m.addVar(lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="bdb_w_func")
    # Set objective function
    obj = sum((x_i - o_p['bdb_x']) * (x_i - o_p['bdb_x']) + (y_i - o_p['bdb_y']) * (y_i - o_p['bdb_y']) for o_p in P)
    m.setObjective(obj, gp.GRB.MAXIMIZE)
    # Add constraints
    c_3, c_4, c_5, c_6 = m.addVar(lb=0, vtype=gp.GRB.BINARY, name="c_3"), m.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                   name="c_4"), \
                         m.addVar(lb=0, vtype=gp.GRB.BINARY, name="c_5"), m.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                   name="c_6")
    # Orientation
    m.addConstr(ori_i + (1 - c_3) * (0 - ori_i) == 0, name="t_0")
    m.addConstr(ori_i + (1 - c_4) * (90 - ori_i) == 90, name="t_1")
    m.addConstr(ori_i + (1 - c_5) * (180 - ori_i) == 180, name="t_2")
    m.addConstr(ori_i + (1 - c_6) * (270 - ori_i) == 270, name="t_3")
    m.addConstr(c_3 + c_4 + c_5 + c_6 == 1, name="t_4")

    m.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270], [o_i['length'], o_i['width'], o_i['length'], o_i['width']], name="bdb_l_func")
    m.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270], [o_i['width'], o_i['length'], o_i['width'], o_i['length']], name="bdb_w_func")

    # Wall
    m.addConstr(y_i + (1 - c_3) * (w_i / 2 - y_i) == w_i / 2, name="w_0")
    m.addConstr(x_i + (1 - c_4) * (l_i / 2 - x_i) == l_i / 2, name="w_1")
    m.addConstr(y_i + (1 - c_5) * (rw - w_i / 2 - y_i) == rw - w_i / 2, name="w_2")
    m.addConstr(x_i + (1 - c_6) * (rl - l_i / 2 - x_i) == rl - l_i / 2, name="w_3")
    # m.addConstr(c_3 + c_4 + c_5 + c_6 >= 1)
    # Collision
    i = 0
    for o_p in P:
        x_i_x_p = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_[{i}]")
        y_i_y_p = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
        abs_x_i_x_p = m.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_[{i}]")
        abs_y_i_y_p = m.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_[{i}]")
        c_1, c_2 = m.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), m.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                    name=f"c_2_[{i}]")
        m.addConstr(x_i_x_p == x_i - o_p['bdb_x'], name=f"coll_x_d_[{i}]")
        m.addConstr(y_i_y_p == y_i - o_p['bdb_y'], name=f"coll_y_d_[{i}]")
        m.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_[{i}]")
        m.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_[{i}]")
        m.addConstr(((o_p['bdb_l'] + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
        m.addConstr(((o_p['bdb_w'] + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
        m.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
        i += 1
    # Out of bound
    m.addConstr(x_i >= l_i / 2, name='o_0')
    m.addConstr(x_i <= rl - l_i / 2, name='o_1')
    m.addConstr(y_i >= w_i / 2, name='o_2')
    m.addConstr(y_i <= rw - w_i / 2, name='o_3')
    # Solve it!
    m.optimize()
    if m.status == GRB.OPTIMAL:
        print("Model status: Optimal solution found")
        return x_i.x, y_i.x, o_i['height'] / 2, l_i.x, w_i.x, o_i['height'], ori_i.x
    elif m.status == GRB.INFEASIBLE:
        print("Model Status: The model is not feasible")
        return None
    elif m.status == GRB.UNBOUNDED:
        print("Model Status: The model is unbounded")
        return None
    else:
        print("Model status: Optimal solution not found")
        return None


def optimize_local_other_obj(o_i, P, up_P, down_P, O_func, rl, rw, rel_pos_constraint = None, room_type = 'bedroom'):
    r_i, func_index = o_i['rel_position'], o_i['func_obj_index']
    l_i, w_i, h_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
    l_i_func, w_i_func, h_i_func, x_i_func, y_i_func, z_i_func, ori_i = O_func[func_index]['bdb_l'], \
                                                                             O_func[func_index]['bdb_w'], \
                                                                             O_func[func_index]['bdb_h'], \
                                                                             O_func[func_index]['bdb_x'], \
                                                                             O_func[func_index]['bdb_y'], \
                                                                             O_func[func_index]['bdb_z'],\
                                                                             O_func[func_index]['bdb_ori']
    o_label, rel = o_i['label'], r_i[1]
    rel = difflib.get_close_matches(rel, ALL_RELATIONS, cutoff=0.0)[0]
    is_use, rel_pos_constraint_i = use_rel_pos_constraint(o_i, rel_pos_constraint, room_type)

    # Create a new model
    m1 = gp.Model()

    if rel_pos_constraint is not None and is_use:
        o_i['rel_position'][1] = rel_pos_constraint_i['relative_position'][1]
        rel_x_i = rel_pos_constraint_i['relative_coordinate'][0]
        rel_y_i = rel_pos_constraint_i['relative_coordinate'][1]
        rel_z_i = rel_pos_constraint_i['relative_coordinate'][2]
        rel_ori_i = rel_pos_constraint_i['rel_ori']

        if room_type == 'livingroom':
            if rel_ori_i < 0:
                rel_ori_i += 360
            o_i['bdb_ori'] = rel_ori_i
            l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
            o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i

        # Create variables
        x_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x")
        y_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y")
        x_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world")
        y_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world")

        # Set objective function
        obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1")
        obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2")
        m1.addConstr(obj1 == ((x_i * x_i) + (y_i * y_i)), name=f"set_obj1")
        m1.addConstr(obj2 == ((x_i - rel_x_i) * (x_i - rel_x_i) + (y_i - rel_y_i) * (y_i - rel_y_i)),
                     name=f"set_obj2")
        m1.setObjectiveN(obj2, index=0, priority=10)
        m1.setObjectiveN(obj1, index=1, priority=5)

        # x_i, y_i ->x_i_world, y_i_world
        cos_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"cos")
        sin_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"sin")
        theta_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"theta")
        m1.addConstr(theta_i == -ori_i / 180 * math.pi, name=f"theta_i")
        m1.addGenConstrCos(theta_i, cos_i, name=f"cos_theta_i_world")
        m1.addGenConstrSin(theta_i, sin_i, name=f"sin_theta_i_world")
        m1.addConstr(x_i_world == x_i_func + (x_i * cos_i - y_i * sin_i), name=f"x_i_world")
        m1.addConstr(y_i_world == y_i_func + (x_i * sin_i + y_i * cos_i), name=f"y_i_world")

        # 定义高度
        if rel == 'above' or rel == 'on':
            o_i['on_the_ceiling'] = True
            d_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"d")
            m1.addConstr(d_i == rel_z_i, name=f"as_z_i_world")

        # Add Constraints
        # Collision
        if rel == 'above' or rel == 'on':
            P_list = up_P
        else:
            P_list = P
        i = 0
        for o_p in P_list:
            x_i_x_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p[{i}]")
            y_i_y_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
            abs_x_i_x_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p[{i}]")
            abs_y_i_y_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p[{i}]")
            c_1, c_2 = m1.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), m1.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                          name=f"c_2[{i}]")
            m1.addConstr(x_i_x_p == x_i_world - o_p['bdb_x'], name=f"coll_x_i-x_p[{i}]")
            m1.addConstr(y_i_y_p == y_i_world - o_p['bdb_y'], name=f"coll_y_i-y_p[{i}]")
            m1.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_i-x_p[{i}]")
            m1.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_i-y_p[{i}]")
            m1.addConstr(((o_p['bdb_l'] + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
            m1.addConstr(((o_p['bdb_w'] + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
            m1.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
            i += 1

        # Out of bound
        m1.addConstr(x_i_world >= l_i / 2, name='o_0')
        m1.addConstr(x_i_world <= rl - l_i / 2, name='o_1')
        m1.addConstr(y_i_world >= w_i / 2, name='o_2')
        m1.addConstr(y_i_world <= rw - w_i / 2, name='o_3')

        # Solve it!
        m1.optimize()
        if m1.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            if rel == 'above' or rel == 'on':
                return x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2, True, is_use
            else:
                return x_i_world.x, y_i_world.x, h_i / 2, False, is_use
        elif m1.status == GRB.INFEASIBLE:
            print("Model Status: The model is not feasible")
            return None
        elif m1.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            return None
        else:
            print("Model status: Optimal solution not found")
            return None
    else:
        # Create variables
        d_i = m1.addVar(lb=0.0, ub=math.sqrt(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="d")
        theta_i = m1.addVar(lb=0, ub=2 * math.pi, vtype=gp.GRB.CONTINUOUS, name="theta")
        theta_i_world = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="theta_world")
        x_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
        y_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="y")
        x_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x_world")
        y_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="y_world")
        if "touching" in rel:
            weight1, weight2, delta = 1, 1, 0
        else:
            weight1, weight2, delta = 1, 1, 0.5

        # Set objective function
        if rel == 'left of' or rel == "left touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == theta_i * theta_i * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == "right of" or rel == "right touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (math.pi - theta_i) * (math.pi - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == 'in front of' or rel == "front touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta)  * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (math.pi / 2 - theta_i) * (math.pi / 2 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == 'behind' or rel == "behind touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (3 * math.pi / 2 - theta_i) * (3 * math.pi / 2 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the right front of' or rel == "right front touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (3 * math.pi / 4 - theta_i) * (3 * math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the left front of' or rel == "left front touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (math.pi / 4 - theta_i) * (math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the right behind of' or rel == "right behind touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (5 * math.pi / 4 - theta_i) * (5 * math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the left behind of' or rel == "left behind touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (7 * math.pi / 4 - theta_i) * (7 * math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == 'above' or rel == 'on top of':
            if rel == 'on top of':
                delta = 0
            else:
                delta = min(rl, rw) / 4
            if len(up_P) == 0:
                return x_i_func, y_i_func, (h_i_func + h_i) / 2 + delta + h_i_func / 2, True, False, is_use
            else:
                obj1 = m1.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="obj1")
                m1.addConstr(
                    obj1 == ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2,
                    name='set_obj1')
                obj2 = m1.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="obj2")
                m1.addConstr(obj2 == (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (
                            y_i_world - y_i_func), name='set_obj2')
                m1.setObjectiveN(obj1 + obj2, index=1, priority=10)
                # obj1 = ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2
                # obj2 = (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (y_i_world - y_i_func)
                # m1.setObjectiveN(obj1 + obj2, index=1, priority=10)
        elif rel == 'below':
            if len(down_P) == 0:
                return x_i_func, y_i_func, 0, True, True, is_use
            else:
                obj1 = m1.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="obj1")
                m1.addConstr(obj1 == (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (
                            y_i_world - y_i_func), name='set_obj1')
                m1.setObjectiveN(obj1, index=1, priority=10)
        else:
            ipdb.set_trace()

        # Add constraints
        # Collision
        if rel == 'above' or rel == 'on':
            P_list = up_P
        elif rel == 'below':
            P_list = down_P
        else:
            P_list = P
        i = 0
        for o_p in P_list:
            x_i_x_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_[{i}]")
            y_i_y_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
            abs_x_i_x_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p[{i}]")
            abs_y_i_y_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p[{i}]")
            c_1, c_2 = m1.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), m1.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                          name=f"c_2[{i}]")
            m1.addConstr(x_i_x_p == x_i_world - o_p['bdb_x'], name=f"coll_x_i-x_p_[{i}]")
            m1.addConstr(y_i_y_p == y_i_world - o_p['bdb_y'], name=f"coll_y_i-y_p[{i}]")
            m1.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_i-x_p_[{i}]")
            m1.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_i-y_p[{i}]")
            m1.addConstr(((o_p['bdb_l'] + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
            m1.addConstr(((o_p['bdb_w'] + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
            m1.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
            i += 1
        # Out of bound
        m1.addConstr(x_i_world >= l_i / 2, name='o_0')
        m1.addConstr(x_i_world <= rl - l_i / 2, name='o_1')
        m1.addConstr(y_i_world >= w_i / 2, name='o_2')
        m1.addConstr(y_i_world <= rw - w_i / 2, name='o_3')
        # Solve it!
        m1.optimize()
        if m1.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            print("__________________________________________________________________________")
            if rel == 'above' or rel == 'on top of':
                return x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2, True, False, is_use
            elif rel == 'below':
                return x_i_world.x, y_i_world.x, 0, False, True, is_use
            else:
                return x_i_world.x, y_i_world.x, h_i / 2, False, False, is_use
        elif m1.status == GRB.INFEASIBLE:
            print("Model Status: The model is not feasible")
            return None
        elif m1.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            return None
        else:
            print("Model status: Optimal solution not found")
            return None


def optimize_local(val_id, data, room_type = 'bedroom'):
    try:
        rl, rw, O_func, O_other = load_gpt_output(data)
    except:
        ipdb.set_trace()
    all_P = []
    P = [] 
    up_P = [] 
    down_P = [] 
    for o_i in O_func:
        if len(P) == 0:
            # 1) Determine the direction of the object
            ori_list = [0, 90, 180, 270]
            time = 0
            while 1:
                time += 1
                ori_i = random.choice(ori_list)
                o_i['bdb_ori'] = ori_i
                l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                if l_i <= rl and w_i <= rw:
                    break
                if time > 20:
                    print(ori_i)
                    ipdb.set_trace()
            o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
            # 2) Determine the position of the object
            while 1:
                if ori_i == 0:
                    y_i = w_i / 2
                    x_i = round(random.uniform(l_i / 2, rl - l_i / 2), 2)
                elif ori_i == 90:
                    x_i = l_i / 2
                    y_i = round(random.uniform(w_i / 2, rw - w_i / 2), 2)
                elif ori_i == 180:
                    y_i = rw - w_i / 2
                    x_i = round(random.uniform(l_i / 2, rl - l_i / 2), 2)
                elif ori_i == 270:
                    x_i = rl - l_i / 2
                    y_i = round(random.uniform(w_i / 2, rw - w_i / 2), 2)
                if x_i == l_i / 2 or x_i == rl - l_i / 2 or y_i == w_i / 2 or y_i == rw - w_i / 2:
                    break
            o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i, y_i, h_i / 2
            o_i['is_use_rel_lib'] = False
            P.append(o_i)
            all_P.append(o_i)
        else:
            try:
                x_i, y_i, z_i, l_i, w_i, h_i, ori_i = optimize_local_func_obj(o_i, P, rl, rw)
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i, y_i, z_i
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
                o_i['bdb_ori'] = ori_i
                o_i['is_use_rel_lib'] = False 
                P.append(o_i)
                all_P.append(o_i)
            except:
                pass

    for o_i in O_other:
        try:
            func_index = o_i['func_obj_index']
            ori_i_func = O_func[func_index]['bdb_ori']
            if 'rel_pos_constraint' in O_func[func_index].keys():
                rel_pos_constraint = O_func[func_index]['rel_pos_constraint']
            else:
                rel_pos_constraint = None
           
            if 'orientation' in o_i.keys():
                # delta = (O_func[func_index]['orientation'] - o_i['orientation'])+ori_i_func
                delta = ori_i_func - (O_func[func_index]['orientation'] - o_i['orientation'])
                if delta > 360:
                    delta -= 360
                elif delta < 0:
                    delta+=360
                o_i['bdb_ori'] = delta
                l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
            else:
                o_i['bdb_ori'] = ori_i_func
                l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
            x_i, y_i, z_i, is_up, is_down, is_use = optimize_local_other_obj(o_i, P, up_P, down_P, O_func, rl, rw, rel_pos_constraint, room_type)
            o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i, y_i, z_i
            o_i['is_use_rel_lib'] = is_use 
            if is_up:
                up_P.append(o_i)
            elif is_down:
                down_P.append(o_i)
            else:
                P.append(o_i)
            all_P.append(o_i)
        except:
            ipdb.set_trace()
            all_P.append({})
    object_list = parse_gpt_output_step_by_step(all_P, type = 'local')
    # updated hierarchy data
    for area in data['children']:
        func_obj = area['children']
        flag_func = 0
        for p in all_P:
            if p['label'] == func_obj['label']:
                flag_func = 1
                func_obj_infom = p
                func_obj['bdb_x'], func_obj['bdb_y'], func_obj['bdb_z'] = func_obj_infom['bdb_x'], \
                                                                          func_obj_infom['bdb_y'], \
                                                                          func_obj_infom['bdb_z']
                func_obj['bdb_l'], func_obj['bdb_w'], func_obj['bdb_h'] = func_obj_infom['bdb_l'], \
                                                                          func_obj_infom['bdb_w'], \
                                                                          func_obj_infom['bdb_h']
                func_obj['left'], func_obj['top'], func_obj['depth'] = func_obj['bdb_x'], func_obj['bdb_y'], func_obj[
                    'bdb_z']
                func_obj['orientation'] = func_obj_infom['orientation']
                func_obj['bdb_ori'] = func_obj_infom['bdb_ori']
        if flag_func == 0:
            print('The functional object has been deleted')
            func_obj['is_delete'] = True
            ipdb.set_trace()

        if 'children' in func_obj.keys():
            for obj in func_obj['children']:
                flag = 0
                for p in all_P:
                    if p['label'] == obj['label']:
                        flag = 1
                        obj_infom = p
                        obj['bdb_x'], obj['bdb_y'], obj['bdb_z'] = obj_infom['bdb_x'], \
                                                                   obj_infom['bdb_y'], \
                                                                   obj_infom['bdb_z']
                        obj['bdb_l'], obj['bdb_w'], obj['bdb_h'] = obj_infom['bdb_l'], \
                                                                   obj_infom['bdb_w'], \
                                                                   obj_infom['bdb_h']
                        obj['left'], obj['top'], obj['depth'] = obj['bdb_x'], obj['bdb_y'], obj['bdb_z']
                        obj['orientation'] = obj_infom['orientation']
                        obj['bdb_ori'] = obj_infom['bdb_ori']
                if flag == 0:
                    print('The object has been deleted')
                    obj['is_delete'] = True
                    ipdb.set_trace()
    return data, object_list, None

def optimize_global_set_obj_function_for_func_obj(model, O_func, rl, rw, obj_func_num):
    func_obj_size = len(O_func)
    x_func = model.addVars(func_obj_size, lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="x_func")
    y_func = model.addVars(func_obj_size, lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="y_func")

    # Set objective function
    obj = []
    for i in range(func_obj_size - 1):
        for j in range(i + 1, func_obj_size):
            obj.append((x_func[i] - x_func[j]) * (x_func[i] - x_func[j]) + (y_func[i] - y_func[j]) * (y_func[i] - y_func[j]))
    obj_func = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="obj_func")
    model.addConstr(obj_func == -quicksum(obj), name='constr_obj_func')
    # model.setObjective(sum(obj), gp.GRB.MAXIMIZE)
    # model.setObjectiveN(obj_func, index=obj_func_num, weight = 2)
    model.setObjectiveN(obj_func, index=obj_func_num, weight=0.1)
    obj_func_num += 1
    model.update()
    return model, obj_func_num


def optimize_global_add_func_obj_constraint(model, O_func, rl, rw):
    func_obj_size = len(O_func)
    x_func = [var for var in model.getVars() if "x_func" in var.VarName]
    y_func = [var for var in model.getVars() if "y_func" in var.VarName]
    # ori_func = model.addVars(func_obj_size, vtype=gp.GRB.INTEGER, name="ori_func")
    # bdb_l_func = model.addVars(func_obj_size, lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="bdb_l_func")
    # bdb_w_func = model.addVars(func_obj_size, lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="bdb_w_func")

    for i in range(len(O_func)):
        o_i = O_func[i]
        ori_i = o_i['orientation']
        o_i['bdb_ori'] = ori_i
        l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
        o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
        # l_i = bdb_l_func[i]
        # w_i = bdb_w_func[i]
        x_i = x_func[i]
        y_i = y_func[i]
        c_3, c_4, c_5, c_6 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_3_[{i}]"), \
                             model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_4_[{i}]"), \
                             model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_5_[{i}]"), \
                             model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_6_[{i}]")
        model.addConstr(ori_i + (1 - c_3) * (0 - ori_i) == 0, name = f"t_0_[{i}]")
        model.addConstr(ori_i + (1 - c_4) * (90 - ori_i) == 90, name = f"t_1_[{i}]")
        model.addConstr(ori_i + (1 - c_5) * (180 - ori_i) == 180, name = f"t_2_[{i}]")
        model.addConstr(ori_i + (1 - c_6) * (270 - ori_i) == 270, name=f"t_3_[{i}]")
        model.addConstr(c_3 + c_4 + c_5 + c_6 == 1, name=f"t_4_[{i}]")

        # model.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270], [o_i['length'], o_i['width'], o_i['length'], o_i['width']], name = f"bdb_l_func_[{i}]")
        # model.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270], [o_i['width'], o_i['length'], o_i['width'], o_i['length']], name=f"bdb_w_func_[{i}]")

        model.addConstr(y_i + (1 - c_3) * (w_i / 2 - y_i) == w_i / 2, name=f"w_0_[{i}]")
        model.addConstr(x_i + (1 - c_4) * (l_i / 2 - x_i) == l_i / 2, name=f"w_1_[{i}]")
        model.addConstr(y_i + (1 - c_5) * (rw - w_i / 2 - y_i) == rw - w_i / 2, name=f"w_2_[{i}]")
        model.addConstr(x_i + (1 - c_6) * (rl - l_i / 2 - x_i) == rl - l_i / 2, name=f"w_3_[{i}]")

        # model.addGenConstrPWL(theta_i, x_i, [90, 270], [o_i['width']/2, rl-o_i['width']/2], name=f"w_x_[{i}]")
        # model.addGenConstrPWL(theta_i, y_i, [0, 180], [o_i['width']/2, rw-o_i['width']/2], name=f"w_y_[{i}]")
    model.update()
    return model


def optimize_global_set_obj_function_for_other_obj(model, O_func, O_other, rl, rw, obj_func_num, rel_pos_constraint=None, room_type = 'bedroom'):
    i = 0
    up_obj = []
    for o_i in O_other:
        h_i, r_i, func_index = o_i['height'], o_i['rel_position'], o_i['func_obj_index']
        h_i_func, x_i_func, y_i_func, ori_i_func = O_func[func_index]['height'], [var for var in model.getVars() if "x_func" in var.VarName][func_index], \
                                                   [var for var in model.getVars() if "y_func" in var.VarName][func_index], \
                                                   O_func[func_index]['orientation']
        o_label, rel = o_i['label'], r_i[1]
        rel = difflib.get_close_matches(rel, ALL_RELATIONS, cutoff=0.0)[0]
        if 'rel_pos_constraint' in O_func[func_index].keys():
            rel_pos_constraint = O_func[func_index]['rel_pos_constraint']

        is_use, rel_pos_constraint_i = use_rel_pos_constraint(o_i, rel_pos_constraint, room_type)

        if rel_pos_constraint is not None and is_use:
            o_i['rel_position'][1] = rel_pos_constraint_i['relative_position'][1]
            rel_x_i = rel_pos_constraint_i['relative_coordinate'][0]
            rel_y_i = rel_pos_constraint_i['relative_coordinate'][1]
            rel_z_i = rel_pos_constraint_i['relative_coordinate'][2]
            rel_ori_i = rel_pos_constraint_i['rel_ori']

            if rel_ori_i < 0:
                rel_ori_i += 360
            o_i['bdb_ori'] = rel_ori_i+ori_i_func
            l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
            o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i

            # Create variables
            x_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_[{i}]")
            y_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_[{i}]")
            x_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world_[{i}]")
            y_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world_[{i}]")

            # Set objective function
            obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
            obj2 = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
            model.addConstr(obj1 == ((x_i * x_i) + (y_i * y_i)), name=f"set_obj1_[{i}]")
            model.addConstr(obj2 == ((x_i-rel_x_i)*(x_i-rel_x_i)+(y_i-rel_y_i)*(y_i-rel_y_i)), name=f"set_obj2_[{i}]")
            model.setObjectiveN(obj2, index=obj_func_num, priority=10)
            obj_func_num += 1
            model.setObjectiveN(obj1, index=obj_func_num, priority=5)
            obj_func_num += 1

            # x_i, y_i ->x_i_world, y_i_world
            cos_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"cos_[{i}]")
            sin_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"sin_[{i}]")
            theta_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"theta_[{i}]")
            model.addConstr(theta_i == -ori_i_func / 180 * math.pi, name=f"theta_i_[{i}]")
            model.addGenConstrCos(theta_i, cos_i, name=f"cos_theta_i_world_[{i}]")
            model.addGenConstrSin(theta_i, sin_i, name=f"sin_theta_i_world_[{i}]")
            model.addConstr(x_i_world == x_i_func + (x_i*cos_i-y_i*sin_i), name=f"x_i_world_[{i}]")
            model.addConstr(y_i_world == y_i_func + (x_i*sin_i+y_i*cos_i), name=f"y_i_world_[{i}]")
            if rel == 'above' or rel == 'on':
                o_i['on_the_ceiling'] = True
                up_obj.append(o_i)
                d_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"d_[{i}]")
                model.addConstr(d_i == rel_z_i, name=f"as_z_i_world_[{i}]")

        else:
            if 'orientation' in o_i.keys():
                rel_ori_i = ori_i_func - (O_func[func_index]['orientation'] - o_i['orientation'])
                if rel_ori_i < 0:
                    rel_ori_i += 360
                o_i['bdb_ori'] = rel_ori_i
                l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i

            else:
                o_i['bdb_ori'] = ori_i_func
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = o_i['length'], o_i['width'], o_i['height']

            # Create variables
            d_i = model.addVar(lb=0.0, ub=math.sqrt(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"d_[{i}]")
            theta_i = model.addVar(lb=0, ub=2 * math.pi, vtype=gp.GRB.CONTINUOUS, name=f"theta_[{i}]")
            theta_i_world = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"theta_world_[{i}]")
            x_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_[{i}]")
            y_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_[{i}]")
            x_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world_[{i}]")
            y_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world_[{i}]")
            if "touching" in rel:
                weight1, weight2, delta = 1, 1, 0
            else:
                # weight1, weight2, delta = 0, 1, max(l_i, w_i, l_i_func, w_i_func)
                weight1, weight2, delta = 1, 1, 0
            # Set objective function
            if rel == 'left of' or rel == "left touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = theta_i*theta_i * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == theta_i * theta_i * 1 / 2, name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel == "right of" or rel == "right touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = (math.pi-theta_i) * (math.pi-theta_i) * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == (math.pi - theta_i) * (math.pi - theta_i) * 1 / 2, name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel == 'in front of' or rel == "front touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = (math.pi/2 - theta_i) * (math.pi/2 - theta_i) * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == (math.pi / 2 - theta_i) * (math.pi / 2 - theta_i) * 1 / 2,
                                name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel == 'behind' or rel == "behind touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = (3*math.pi/2 - theta_i) * (3*math.pi/2 - theta_i) * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == (3 * math.pi / 2 - theta_i) * (3 * math.pi / 2 - theta_i) * 1 / 2,
                                name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel in 'in the right front of' or rel == "right front touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = (3*math.pi/4 - theta_i) * (3*math.pi/4 - theta_i) * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == (3 * math.pi / 4 - theta_i) * (3 * math.pi / 4 - theta_i) * 1 / 2,
                                name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel in 'in the left front of' or rel == "left front touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = (math.pi/4 - theta_i) * (math.pi/4 - theta_i) * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == (math.pi / 4 - theta_i) * (math.pi / 4 - theta_i) * 1 / 2,
                                name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel in 'in the right behind of' or rel == "right behind touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = (5*math.pi/4 - theta_i) * (5*math.pi/4 - theta_i) * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == (5 * math.pi / 4 - theta_i) * (5 * math.pi / 4 - theta_i) * 1 / 2,
                                name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel in 'in the left behind of' or rel == "left behind touching":
                # obj1 = (d_i-delta) * (d_i-delta) * 1 / 2
                # obj2 = (7*math.pi/4 - theta_i) * (7*math.pi/4 - theta_i) * 1 / 2
                # model.setObjective(weight1*obj1+weight2*obj2, gp.GRB.MINIMIZE)
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == (7 * math.pi / 4 - theta_i) * (7 * math.pi / 4 - theta_i) * 1 / 2,
                                name=f"set_obj2_[{i}]")
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1
            elif rel == 'above' or rel == 'on':
                if rel == 'on':
                    delta = 0
                else:
                    delta = min(rl, rw) / 4
                # obj1 = ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2
                # obj2 = (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (y_i_world - y_i_func)
                # model.setObjective(obj1 + obj2, gp.GRB.MINIMIZE)
                obj1 = ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2
                obj2 = (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (y_i_world - y_i_func)
                obj1_func = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"above_obj1_func_[{i}]")
                obj2_func = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"above_obj2_func_[{i}]")
                model.addConstr(obj1_func == obj1, name=f"constr_above_obj1_func_[{i}]")
                model.addConstr(obj2_func == obj2, name=f"constr_above_obj1_func_[{i}]")
                model.setObjectiveN(obj1_func + obj2_func, index=obj_func_num, priority=10)
                obj_func_num += 1
                o_i['on_the_ceiling'] = True
                up_obj.append(o_i)
                i += 1
                continue
            else:
                ipdb.set_trace()

            # d_i, theta_i->x_i,y_i
            model.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i_func / 180 * math.pi),
                            name=f"theta_i_world_[{i}]")
            model.addGenConstrCos(theta_i_world, x_i, name=f"cos_theta_i_world_[{i}]")
            model.addGenConstrSin(theta_i_world, y_i, name=f"sin_theta_i_world_[{i}]")
            model.addConstr(x_i_world == x_i_func + d_i * x_i, name=f"x_i_world_[{i}]")
            model.addConstr(y_i_world == y_i_func + d_i * y_i, name=f"y_i_world_[{i}]")
        i += 1
    model.update()
    return model, up_obj, obj_func_num


def optimize_global_add_collision_constraint(model, O_func, O_other, up_obj, rl, rw):
    # Collision
    # 1) Object on th ground
    O = O_func+O_other
    for i in range(len(O)-1):
        o_i = O[i]
        for j in range(i+1, len(O)):
            o_j = O[j]
            if (o_i != o_j) and o_i['on_the_ceiling'] == False and o_j['on_the_ceiling'] == False:
                index_i = o_i['obj_index']
                index_j = o_j['obj_index']
                if o_i in O_func:
                    x_i = [var for var in model.getVars() if "x_func" in var.VarName][index_i]
                    y_i = [var for var in model.getVars() if "y_func" in var.VarName][index_i]
                    l_i, w_i, h_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
                    # l_i, w_i, h_i = [var for var in model.getVars() if "bdb_l_func" in var.VarName][index_i], [var for var in model.getVars() if "bdb_w_func" in var.VarName][index_i], o_i['height']
                else:
                    x_i = model.getVarByName(f"x_world_[{index_i}]")
                    y_i = model.getVarByName(f"y_world_[{index_i}]")
                    l_i, w_i, h_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
                    # l_i, w_i, h_i = model.getVarByName(f"bdb_l_[{index_i}]"), model.getVarByName(f"bdb_w_[{index_i}]"), o_i['height']
                if o_j in O_func:
                    x_j = [var for var in model.getVars() if "x_func" in var.VarName][index_j]
                    y_j = [var for var in model.getVars() if "y_func" in var.VarName][index_j]
                    l_j, w_j, h_j = o_j['bdb_l'], o_j['bdb_w'], o_j['bdb_h']
                    #l_j, w_j, h_j = [var for var in model.getVars() if "bdb_l_func" in var.VarName][index_j], [var for var in model.getVars() if "bdb_w_func" in var.VarName][index_j], o_i['height']
                else:
                    x_j = model.getVarByName(f"x_world_[{index_j}]")
                    y_j = model.getVarByName(f"y_world_[{index_j}]")
                    l_j, w_j, h_j = o_j['bdb_l'], o_j['bdb_w'], o_j['bdb_h']
                    #l_j, w_j, h_j = model.getVarByName(f"bdb_l_[{index_j}]"), model.getVarByName(f"bdb_w_[{index_j}]"), o_i['height']

                x_i_x_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_ground_[{i}]")
                y_i_y_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p_ground_[{i}]")
                abs_x_i_x_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_ground_[{i}]")
                abs_y_i_y_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_ground_[{i}]")
                c_1, c_2 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_ground_[{i}]"), \
                           model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_2_ground_[{i}]")

                model.addConstr(x_i_x_p == x_i - x_j, name=f"coll_x_d_ground_[{i}]")
                model.addConstr(y_i_y_p == y_i - y_j, name=f"coll_y_d_ground_[{i}]")
                model.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_ground_[{i}]")
                model.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_ground_[{i}]")
                model.addConstr(((l_j + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_ground_[{i}]")
                model.addConstr(((w_j + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_ground_[{i}]")
                model.addConstr(c_1 + c_2 >= 1, name=f"coll_c_ground_[{i}]")
    # 2) Objects on the ceiling
    for i in range(len(up_obj)-1):
        o_i = up_obj[i]
        for j in range(i+1, len(up_obj)):
            o_j = up_obj[j]
            if (o_i != o_j) and o_i['on_the_ceiling'] == False and o_j['on_the_ceiling'] == False:
                index_i = o_i['obj_index']
                index_j = o_j['obj_index']
                x_i = model.getVarByName(f"x_world_[{index_i}]")
                y_i = model.getVarByName(f"y_world_[{index_i}]")
                x_j = model.getVarByName(f"x_world_[{index_j}]")
                y_j = model.getVarByName(f"y_world_[{index_j}]")
                # l_i, w_i, h_i = model.getVarByName(f"bdb_l_[{index_i}]"), model.getVarByName(f"bdb_w_[{index_i}]"), o_i['height']
                # l_j, w_j, h_j = model.getVarByName(f"bdb_l_[{index_j}]"), model.getVarByName(f"bdb_w_[{index_j}]"), o_i['height']
                l_i, w_i, h_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
                l_j, w_j, h_j = o_j['bdb_l'], o_j['bdb_w'], o_j['bdb_h']

                x_i_x_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_ceiling_[{i}]")
                y_i_y_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p_ceiling_[{i}]")
                abs_x_i_x_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_ceiling_[{i}]")
                abs_y_i_y_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_ceiling_[{i}]")
                c_1, c_2 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_ceiling_[{i}]"), \
                           model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_2_ceiling_[{i}]")

                model.addConstr(x_i_x_p == x_i - x_j, name=f"coll_x_d_ceiling_[{i}]")
                model.addConstr(y_i_y_p == y_i - y_j, name=f"coll_y_d_ceiling_[{i}]")
                model.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_ceiling_[{i}]")
                model.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_ceiling_[{i}]")
                model.addConstr(((l_j + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_ceiling_[{i}]")
                model.addConstr(((w_j + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_ceiling_[{i}]")
                model.addConstr(c_1 + c_2 >= 1, name=f"coll_c_ceiling_[{i}]")
                i += 1
    model.update()
    return model


def optimize_global_add_out_of_bound_constraint(model, O_func, O_other, rl, rw):
    for i in range(len(O_func)):
        # l_i, w_i, h_i = [var for var in model.getVars() if "bdb_l_func" in var.VarName][i], [var for var in model.getVars() if "bdb_w_func" in var.VarName][i], O_func[i]['height']
        l_i, w_i, h_i = O_func[i]['bdb_l'], O_func[i]['bdb_w'], O_func[i]['bdb_h']
        x_i = [var for var in model.getVars() if "x_func" in var.VarName][i]
        y_i = [var for var in model.getVars() if "y_func" in var.VarName][i]
        model.addConstr(x_i >= l_i / 2, name=f"o_f_0_[{i}]")
        model.addConstr(x_i <= rl - l_i / 2, name=f"o_f_1_[{i}]")
        model.addConstr(y_i >= w_i / 2, name=f"o_f_2_[{i}]")
        model.addConstr(y_i <= rw - w_i / 2, name=f"o_f_3_[{i}]")
    for i in range(len(O_other)):
        # l_i, w_i, h_i = model.getVarByName(f"bdb_l_[{i}]"), model.getVarByName(f"bdb_w_[{i}]"), O_other[i]['height']
        l_i, w_i, h_i = O_other[i]['bdb_l'], O_other[i]['bdb_w'], O_other[i]['bdb_h']
        x_i_world = model.getVarByName(f"x_world_[{i}]")
        y_i_world = model.getVarByName(f"y_world_[{i}]")
        model.addConstr(x_i_world >= l_i / 2, name=f"o_o_0_[{i}]")
        model.addConstr(x_i_world <= rl - l_i / 2, name=f"o_o_1_[{i}]")
        model.addConstr(y_i_world >= w_i / 2, name=f"o_o_2_[{i}]")
        model.addConstr(y_i_world <= rw - w_i / 2, name=f"o_o_3_[{i}]")
    model.update()
    return model


def optimize_global(val_id, input, room_type='bedroom'):
    # input
    rl, rw, O_func, O_other = load_gpt_output(input)  # Return the size of the room, the sequence of functional objects, and the sequence of non-functional objects

    # Create a new model
    m = gp.Model()
    obj_func_num = 0 

    # Set objective function for func_obj
    m, obj_func_num = optimize_global_set_obj_function_for_func_obj(m, O_func, rl, rw, obj_func_num)

    # Add func obj constraint
    m = optimize_global_add_func_obj_constraint(m, O_func, rl, rw)

    # Set objective function for other_obj
    m, up_obj, obj_func_num = optimize_global_set_obj_function_for_other_obj(m, O_func, O_other, rl, rw, obj_func_num, None, room_type)

    # # Add Out of bound constraint
    m = optimize_global_add_out_of_bound_constraint(m, O_func, O_other, rl, rw)

    # Add collision constraint
    m = optimize_global_add_collision_constraint(m, O_func, O_other, up_obj, rl, rw)

    # Solve it!
    m.Params.TimeLimit = 600  # 10 minutes
    m.optimize()
    # m.computeIIS()
    # m.write("delete_model1.ilp")
    if m.status == GRB.OPTIMAL:
        print("Model status: Optimal solution found")
        print("__________________________________________________________________________")
        all_P = []
        for i in range(len(O_func)):
            o_i = O_func[i]
            h_i = o_i['height']
            x_i = [var for var in m.getVars() if "x_func" in var.VarName][i]
            y_i = [var for var in m.getVars() if "y_func" in var.VarName][i]
            z_i = h_i / 2
            o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i.x, y_i.x, z_i
            # o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'], o_i['bdb_ori'] = l_i.x, w_i.x, h_i, ori_i.x
            all_P.append(o_i)
        for i in range(len(O_other)):
            o_i = O_other[i]
            # l_i, w_i, h_i, func_index = m.getVarByName(f"bdb_l_[{i}]"), m.getVarByName(f"bdb_w_[{i}]"), o_i['height'], \
            #                             o_i['func_obj_index']
            h_i = o_i['height']
            func_index = o_i['func_obj_index']
            x_i_world = m.getVarByName(f"x_world_[{i}]")
            y_i_world = m.getVarByName(f"y_world_[{i}]")
            # ori_i = O_func[func_index]['orientation']
            if o_i['on_the_ceiling']:
                h_i_func = O_func[func_index]['bdb_h']
                d_i = m.getVarByName(f"d_[{i}]")
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2
            else:
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, h_i / 2

            # o_i['bdb_ori'] = o_i['bdb_ori'] + O_func[func_index]['orientation']
            l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
            o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
            # o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'], o_i['bdb_ori'] = l_i.x, w_i.x, h_i, ori_i
            all_P.append(o_i)
        object_list = parse_gpt_output_step_by_step(all_P, type = 'global')
        # Updated hierarchy data
        for area in input['children']:
            func_obj = area['children']
            flag_func = 0
            for p in all_P:
                if p['label'] == func_obj['label']:
                    flag_func = 1
                    func_obj_infom = p
                    func_obj['bdb_x'], func_obj['bdb_y'], func_obj['bdb_z'] = func_obj_infom['bdb_x'], \
                                                                              func_obj_infom['bdb_y'], \
                                                                              func_obj_infom['bdb_z']
                    func_obj['bdb_l'], func_obj['bdb_w'], func_obj['bdb_h'] = func_obj_infom['bdb_l'], \
                                                                              func_obj_infom['bdb_w'], \
                                                                              func_obj_infom['bdb_h']
                    func_obj['left'], func_obj['top'], func_obj['depth'] = func_obj['bdb_x'], func_obj['bdb_y'], \
                                                                           func_obj[
                                                                               'bdb_z']
                    func_obj['orientation'] = func_obj_infom['orientation']
                    func_obj['bdb_ori'] = func_obj_infom['bdb_ori']
            if flag_func == 0:
                print('Functional object has been deleted')
                func_obj['is_delete'] = True
                ipdb.set_trace()

            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    flag = 0
                    for p in all_P:
                        if p['label'] == obj['label']:
                            flag = 1
                            obj_infom = p
                            obj['bdb_x'], obj['bdb_y'], obj['bdb_z'] = obj_infom['bdb_x'], \
                                                                       obj_infom['bdb_y'], \
                                                                       obj_infom['bdb_z']
                            obj['bdb_l'], obj['bdb_w'], obj['bdb_h'] = obj_infom['bdb_l'], \
                                                                       obj_infom['bdb_w'], \
                                                                       obj_infom['bdb_h']
                            obj['left'], obj['top'], obj['depth'] = obj['bdb_x'], obj['bdb_y'], obj['bdb_z']
                            obj['orientation'] = obj_infom['orientation']
                            obj['bdb_ori'] = obj_infom['bdb_ori']
                    if flag == 0:
                        print('Object has been deleted')
                        obj['is_delete'] = True
                        ipdb.set_trace()

        return input, object_list, True

    elif m.status == GRB.INFEASIBLE:
        print("Model Status: The model is not feasible")
        print(val_id)
        # 更新hierarchy data
        for area in input['children']:
            func_obj = area['children']
            func_obj['is_delete'] = True
            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    obj['is_delete'] = True
        # ipdb.set_trace()
        return input, [], False
    elif m.status == GRB.UNBOUNDED:
        print("Model Status: The model is unbounded")
        print(val_id)
        # 更新hierarchy data
        for area in input['children']:
            func_obj = area['children']
            func_obj['is_delete'] = True
            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    obj['is_delete'] = True
        # ipdb.set_trace()
        return input, [], False
    else:
        print("Model status: Optimal solution not found")
        # ipdb.set_trace()
        print(val_id)
        all_P = []
        for i in range(len(O_func)):
            o_i = O_func[i]
            # l_i, w_i, h_i = [var for var in m.getVars() if "bdb_l_func" in var.VarName][i], \
            #                 [var for var in m.getVars() if "bdb_w_func" in var.VarName][i], o_i['height']
            # ori_i = [var for var in m.getVars() if "ori_func" in var.VarName][i]
            h_i = o_i['height']
            x_i = [var for var in m.getVars() if "x_func" in var.VarName][i]
            y_i = [var for var in m.getVars() if "y_func" in var.VarName][i]
            z_i = h_i / 2
            o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i.x, y_i.x, z_i
            # o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'], o_i['bdb_ori'] = l_i.x, w_i.x, h_i, ori_i.x
            all_P.append(o_i)
        for i in range(len(O_other)):
            o_i = O_other[i]
            # l_i, w_i, h_i, func_index = m.getVarByName(f"bdb_l_[{i}]"), m.getVarByName(f"bdb_w_[{i}]"), o_i['height'], \
            #                             o_i['func_obj_index']
            h_i = o_i['height']
            func_index = o_i['func_obj_index']
            x_i_world = m.getVarByName(f"x_world_[{i}]")
            y_i_world = m.getVarByName(f"y_world_[{i}]")
            # ori_i = O_func[func_index]['bdb_ori']
            if o_i['on_the_ceiling']:
                h_i_func = O_func[func_index]['bdb_h']
                d_i = m.getVarByName(f"d_[{i}]")
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2
            else:
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, h_i / 2
            # o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'], o_i['bdb_ori'] = l_i.x, w_i.x, h_i, ori_i
            all_P.append(o_i)
        object_list = parse_gpt_output_step_by_step(all_P, type = 'global')
        # Updated hierarchy data
        for area in input['children']:
            func_obj = area['children']
            flag_func = 0
            for p in all_P:
                if p['label'] == func_obj['label']:
                    flag_func = 1
                    func_obj_infom = p
                    func_obj['bdb_x'], func_obj['bdb_y'], func_obj['bdb_z'] = func_obj_infom['bdb_x'], \
                                                                              func_obj_infom['bdb_y'], \
                                                                              func_obj_infom['bdb_z']
                    func_obj['bdb_l'], func_obj['bdb_w'], func_obj['bdb_h'] = func_obj_infom['bdb_l'], \
                                                                              func_obj_infom['bdb_w'], \
                                                                              func_obj_infom['bdb_h']
                    func_obj['left'], func_obj['top'], func_obj['depth'] = func_obj['bdb_x'], func_obj['bdb_y'], \
                                                                           func_obj['bdb_z']
                    func_obj['orientation'] = func_obj_infom['orientation']
                    func_obj['bdb_ori'] = func_obj_infom['bdb_ori']
            if flag_func == 0:
                print('Functional object has been deleted')
                func_obj['is_delete'] = True
                ipdb.set_trace()

            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    flag = 0
                    for p in all_P:
                        if p['label'] == obj['label']:
                            flag = 1
                            obj_infom = p
                            obj['bdb_x'], obj['bdb_y'], obj['bdb_z'] = obj_infom['bdb_x'], \
                                                                       obj_infom['bdb_y'], \
                                                                       obj_infom['bdb_z']
                            obj['bdb_l'], obj['bdb_w'], obj['bdb_h'] = obj_infom['bdb_l'], \
                                                                       obj_infom['bdb_w'], \
                                                                       obj_infom['bdb_h']
                            obj['left'], obj['top'], obj['depth'] = obj['bdb_x'], obj['bdb_y'], obj['bdb_z']
                            obj['orientation'] = obj_infom['orientation']
                            obj['bdb_ori'] = obj_infom['bdb_ori']
                    if flag == 0:
                        print('Object has been deleted')
                        obj['is_delete'] = True
                        ipdb.set_trace()

        return input, object_list, False


def optimize_global_set_obj_function_for_func_obj_by_obj(model, O_func, rl, rw, obj_func_num):
    func_obj_size = len(O_func)
    x_func = model.addVars(func_obj_size, lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="x_func")
    y_func = model.addVars(func_obj_size, lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="y_func")

    # Set objective function
    obj = []
    for i in range(func_obj_size - 1):
        for j in range(i + 1, func_obj_size):
            if O_func[i]['is_optimized'] and O_func[j]['is_optimized']:
                obj.append((x_func[i] - x_func[j]) * (x_func[i] - x_func[j]) + (y_func[i] - y_func[j]) * (y_func[i] - y_func[j]))
            elif O_func[i]['is_optimized'] and not O_func[j]['is_optimized']:
                    old_x_func, old_y_func = O_func[j]['bdb_x'], O_func[j]['bdb_y']
                    obj.append(
                        (x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_x_func) * (
                                    y_func[i] - old_x_func)
                    )
            elif not O_func[i]['is_optimized'] and O_func[j]['is_optimized']:
                old_x_func, old_y_func = O_func[i]['bdb_x'], O_func[i]['bdb_y']
                obj.append(
                    (old_x_func - x_func[j]) * (old_x_func - x_func[j]) + (old_y_func - y_func[j]) * (
                            old_y_func - y_func[j])
                )
            else:
                continue
    obj_func = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="obj_func")
    model.addConstr(obj_func == -quicksum(obj), name='constr_obj_func')
    model.setObjectiveN(obj_func, index=obj_func_num, weight = 0.1)
    obj_func_num += 1

    for i in range(func_obj_size):
        if not O_func[i]['is_optimized']:
            old_x_func, old_y_func = O_func[i]['bdb_x'], O_func[i]['bdb_y']
            # Set objective function
            frozen_obj_func = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"frozen_obj_func")
            model.addConstr(
                frozen_obj_func == (
                        (x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_y_func) * (
                        y_func[i] - old_y_func)), name=f"set_obj3")
            model.setObjectiveN(frozen_obj_func, index=obj_func_num, weight=5, priority=15, name='frozen_constr_obj_func')
            obj_func_num += 1

    model.update()
    return model, obj_func_num


def optimize_global_add_func_obj_constraint_by_obj(model, O_func, rl, rw):
    func_obj_size = len(O_func)
    x_func = [var for var in model.getVars() if "x_func" in var.VarName]
    y_func = [var for var in model.getVars() if "y_func" in var.VarName]

    for i in range(len(O_func)):
        o_i = O_func[i]
        if o_i['is_optimized']:
            o_i['bdb_ori'] = 0
            o_i['bdb_l'] = o_i['length']
            o_i['bdb_w'] = o_i['width']
            o_i['bdb_h'] = o_i['height']

            x_i = x_func[i]
            y_i = y_func[i]


            if 'dining_table' in o_i['label']:
                model.addConstr(x_i == rl / 2, name=f"w_0_[{i}]")
                model.addConstr(y_i == rw / 2, name=f"w_1_[{i}]")
            else:
                # model.addConstr(y_i == w_i / 2, name=f"w_0_[{i}]")
                model.addConstr(y_i == o_i['width'] / 2, name=f"w_0_[{i}]")
        else:
            x_i = x_func[i]
            y_i = y_func[i]
            old_orientation = o_i['bdb_ori']

            if 'dining_table' in o_i['label']:
                model.addConstr(x_i == rl / 2, name=f"w_0_[{i}]")
                model.addConstr(y_i == rw / 2, name=f"w_1_[{i}]")
            else:
                if old_orientation == 0:
                    model.addConstr(y_i - o_i['width'] / 2 == 0, name=f"w_0_[{i}]")
                elif old_orientation == 90:
                    model.addConstr(x_i - o_i['length'] / 2 == 0, name=f"w_1_[{i}]")
                elif old_orientation == 180:
                    model.addConstr(y_i - (rw - o_i['width'] / 2) == 0, name=f"w_2_[{i}]")
                elif old_orientation == 270:
                    model.addConstr(x_i - (rl - o_i['length'] / 2) == 0, name=f"w_3_[{i}]")
                else:
                    ipdb.set_trace()

    model.update()
    return model


def optimize_global_set_obj_function_for_other_obj_by_obj(model, O_func, O_other, rl, rw, obj_func_num, rel_pos_constraint=None, room_type = 'bedroom'):
    i = 0
    up_obj = []
    down_obj = []
    for o_i in O_other:
        if o_i['is_optimized']:
            h_i, r_i, func_index = o_i['height'], o_i['rel_position'], o_i['func_obj_index']
            h_i_func, x_i_func, y_i_func, ori_i_func = O_func[func_index]['height'], \
                                                       [var for var in model.getVars() if "x_func" in var.VarName][
                                                           func_index], \
                                                       [var for var in model.getVars() if "y_func" in var.VarName][
                                                           func_index], \
                                                       O_func[func_index]['bdb_ori']
            o_label, rel = o_i['label'], r_i[1]
            rel = difflib.get_close_matches(rel, ALL_RELATIONS, cutoff=0.0)[0]

            is_use, rel_pos_constraint_i = use_rel_pos_constraint(o_i, rel_pos_constraint, room_type)
            if rel == 'on top of':
                ipdb.set_trace()

            if rel_pos_constraint is not None and is_use:
                rel_x_i = rel_pos_constraint_i['relative_coordinate'][0]
                rel_y_i = rel_pos_constraint_i['relative_coordinate'][1]
                rel_z_i = rel_pos_constraint_i['relative_coordinate'][2]
                rel_ori_i = rel_pos_constraint_i['rel_ori']

                if rel_ori_i < 0:
                    rel_ori_i += 360
                o_i['bdb_ori'] = rel_ori_i
                l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i

                # Create variables
                x_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_[{i}]")
                y_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_[{i}]")
                x_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world_[{i}]")
                y_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world_[{i}]")

                # Set objective function
                obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                obj2 = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                model.addConstr(obj1 == ((x_i * x_i) + (y_i * y_i)), name=f"set_obj1_[{i}]")
                model.addConstr(obj2 == ((x_i - rel_x_i) * (x_i - rel_x_i) + (y_i - rel_y_i) * (y_i - rel_y_i)),
                                name=f"set_obj2_[{i}]")
                # model.addConstr(obj2 == x_i * rel_y_i - y_i * rel_x_i, name=f"set_obj2_[{i}]")
                # model.addConstr(x_i * rel_x_i >= 0, name=f"set_obj2_add_1_[{i}]")
                # model.addConstr(y_i * rel_y_i >= 0, name=f"set_obj2_add_2_[{i}]")
                # model.setObjectiveN(obj2, index=obj_func_num, weight = 2, priority=10) # priority=10
                model.setObjectiveN(obj2, index=obj_func_num, priority=10)  # priority=10
                obj_func_num += 1
                model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                obj_func_num += 1

                # x_i, y_i ->x_i_world, y_i_world
                cos_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"cos_[{i}]")
                sin_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"sin_[{i}]")
                theta_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"theta_[{i}]")
                model.addConstr(theta_i == -ori_i_func / 180 * math.pi, name=f"theta_i_[{i}]")
                model.addGenConstrCos(theta_i, cos_i, name=f"cos_theta_i_world_[{i}]")
                model.addGenConstrSin(theta_i, sin_i, name=f"sin_theta_i_world_[{i}]")
                model.addConstr(x_i_world == x_i_func + (x_i * cos_i - y_i * sin_i), name=f"x_i_world_[{i}]")
                model.addConstr(y_i_world == y_i_func + (x_i * sin_i + y_i * cos_i), name=f"y_i_world_[{i}]")

                if rel == 'above' or rel == 'on top of':
                    o_i['on_the_ceiling'] = True
                    up_obj.append(o_i)
                    d_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"d_[{i}]")
                    model.addConstr(d_i == rel_z_i, name=f"as_z_i_world_[{i}]")
                elif rel == 'below':
                    down_obj.append(o_i)
                    d_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"d_[{i}]")
                    model.addConstr(d_i == rel_z_i, name=f"as_z_i_world_[{i}]")


            else:
                if 'orientation' in o_i.keys(): 
                    rel_ori_i = -(O_func[func_index]['orientation'] - o_i['orientation'])
                    if rel_ori_i < 0:
                        rel_ori_i += 360
                    o_i['bdb_ori'] = rel_ori_i
                    l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                    o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i

                    # model.addConstr(bdb_l_i == l_i, name=f"bdb_l_other_[{i}]")
                    # model.addConstr(bdb_w_i == w_i, name=f"bdb_w_other_[{i}]")
                    # model.addConstr(bdb_ori_i == rel_ori_i, name=f"bdb_ori_other_[{i}]")
                else:
                    o_i['bdb_ori'] = ori_i_func
                    o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = o_i['length'], o_i['width'], o_i['height']
                    # model.addGenConstrPWL(ori_i_func, bdb_l_i, [0, 90, 180, 270],
                    #                       [o_i['length'], o_i['width'], o_i['length'], o_i['width']],
                    #                       name=f"bdb_l_other_[{i}]")
                    # model.addGenConstrPWL(ori_i_func, bdb_w_i, [0, 90, 180, 270],
                    #                       [o_i['width'], o_i['length'], o_i['width'], o_i['length']],
                    #                       name=f"bdb_w_other_[{i}]")
                    # model.addConstr(bdb_ori_i == ori_i_func, name=f"bdb_ori_other_[{i}]")


                # Create variables
                d_i = model.addVar(lb=0.0, ub=math.sqrt(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"d_[{i}]")
                theta_i = model.addVar(lb=0, ub=2 * math.pi, vtype=gp.GRB.CONTINUOUS, name=f"theta_[{i}]")
                theta_i_world = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"theta_world_[{i}]")
                x_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_[{i}]")
                y_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_[{i}]")
                x_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world_[{i}]")
                y_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world_[{i}]")
                if "touching" in rel:
                    weight1, weight2, delta = 1, 1, 0
                else:
                    # weight1, weight2, delta = 0, 1, max(l_i, w_i, l_i_func, w_i_func)
                    weight1, weight2, delta = 1, 1, 0
                # Set objective function
                if rel == 'left of' or rel == "left touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == theta_i * theta_i * 1 / 2, name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel == "right of" or rel == "right touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == (math.pi - theta_i) * (math.pi - theta_i) * 1 / 2, name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel == 'in front of' or rel == "front touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == (math.pi / 2 - theta_i) * (math.pi / 2 - theta_i) * 1 / 2,
                                    name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel == 'behind' or rel == "behind touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == (3 * math.pi / 2 - theta_i) * (3 * math.pi / 2 - theta_i) * 1 / 2,
                                    name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel in 'in the right front of' or rel == "right front touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == (3 * math.pi / 4 - theta_i) * (3 * math.pi / 4 - theta_i) * 1 / 2,
                                    name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel in 'in the left front of' or rel == "left front touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == (math.pi / 4 - theta_i) * (math.pi / 4 - theta_i) * 1 / 2,
                                    name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel in 'in the right behind of' or rel == "right behind touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == (5 * math.pi / 4 - theta_i) * (5 * math.pi / 4 - theta_i) * 1 / 2,
                                    name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel in 'in the left behind of' or rel == "left behind touching":
                    obj1 = model.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1_[{i}]")
                    obj2 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2_[{i}]")
                    model.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name=f"set_obj1_[{i}]")
                    model.addConstr(obj2 == (7 * math.pi / 4 - theta_i) * (7 * math.pi / 4 - theta_i) * 1 / 2,
                                    name=f"set_obj2_[{i}]")
                    model.setObjectiveN(obj2, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    model.setObjectiveN(obj1, index=obj_func_num, priority=5)
                    obj_func_num += 1
                elif rel == 'above' or rel == 'on top of':
                    if rel == 'on top of':
                        delta = 0
                    else:
                        delta = min(rl, rw) / 4
                    # obj1 = ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2
                    # obj2 = (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (y_i_world - y_i_func)
                    # model.setObjective(obj1 + obj2, gp.GRB.MINIMIZE)
                    obj1 = ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2
                    obj2 = (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (
                                y_i_world - y_i_func)
                    obj1_func = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"above_obj1_func_[{i}]")
                    obj2_func = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"above_obj2_func_[{i}]")
                    model.addConstr(obj1_func == obj1, name=f"constr_above_obj1_func_[{i}]")
                    model.addConstr(obj2_func == obj2, name=f"constr_above_obj1_func_[{i}]")
                    model.setObjectiveN(obj1_func + obj2_func, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    o_i['on_the_ceiling'] = True
                    up_obj.append(o_i)
                    i += 1
                    continue
                elif rel == 'below':
                    obj1 = (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (y_i_world - y_i_func)
                    obj1_func = model.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name=f"below_obj1_func_[{i}]")
                    model.addConstr(obj1_func == obj1, name=f"constr_below_obj1_func_[{i}]")
                    model.setObjectiveN(obj1_func, index=obj_func_num, priority=10)
                    obj_func_num += 1
                    down_obj.append(o_i)
                    i += 1
                    continue
                else:
                    ipdb.set_trace()

                # d_i, theta_i->x_i,y_i
                model.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i_func / 180 * math.pi),
                                name=f"theta_i_world_[{i}]")
                model.addGenConstrCos(theta_i_world, x_i, name=f"cos_theta_i_world_[{i}]")
                model.addGenConstrSin(theta_i_world, y_i, name=f"sin_theta_i_world_[{i}]")
                model.addConstr(x_i_world == x_i_func + d_i * x_i, name=f"x_i_world_[{i}]")
                model.addConstr(y_i_world == y_i_func + d_i * y_i, name=f"y_i_world_[{i}]")
        else:
            r_i, func_index = o_i['rel_position'], o_i['func_obj_index']
            l_i, w_i, h_i, ori_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'], o_i['bdb_ori']
            old_x_i, old_y_i, old_z_i = o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z']

            o_label, rel = o_i['label'], r_i[1]
            rel = difflib.get_close_matches(rel, ALL_RELATIONS, cutoff=0.0)[0]

            # Create variables
            x_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world_[{i}]")
            y_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world_[{i}]")
            z_i_world = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"z_world_[{i}]")

            # Set objective function
            obj3 = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3")
            model.addConstr(
                obj3 == ((x_i_world - old_x_i) * (x_i_world - old_x_i) + (y_i_world - old_y_i) * (y_i_world - old_y_i)
                         + (z_i_world - old_z_i) * (z_i_world - old_z_i)),
                name=f"set_obj3")
            model.setObjectiveN(obj3, index=obj_func_num, weight=5, priority=15)
            obj_func_num+=1

            if rel == 'above' or rel == 'on top of':
                o_i['on_the_ceiling'] = True
                up_obj.append(o_i)

            elif rel == 'below':
                down_obj.append(o_i)

        i += 1
    model.update()
    return model, up_obj, down_obj, obj_func_num


def is_not_in_down(o_i, down_obj):
    for o in down_obj:
        if o == o_i:
            return False
    return True


def optimize_global_add_collision_constraint_by_obj(model, O_func, O_other, up_obj, down_obj, rl, rw):
    # Collision
    # 1) Object on th ground
    O = O_func+O_other
    for i in range(len(O)-1):
        o_i = O[i]
        for j in range(i+1, len(O)):
            o_j = O[j]
            if (o_i != o_j) and o_i['on_the_ceiling'] == False and o_j['on_the_ceiling'] == False and is_not_in_down(o_i, down_obj) and is_not_in_down(o_j, down_obj):
                index_i = o_i['obj_index']
                index_j = o_j['obj_index']
                if o_i in O_func:
                    x_i = [var for var in model.getVars() if "x_func" in var.VarName][index_i]
                    y_i = [var for var in model.getVars() if "y_func" in var.VarName][index_i]
                    l_i, w_i, h_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
                    # l_i, w_i, h_i = [var for var in model.getVars() if "bdb_l_func" in var.VarName][index_i], [var for var in model.getVars() if "bdb_w_func" in var.VarName][index_i], o_i['height']
                else:
                    x_i = model.getVarByName(f"x_world_[{index_i}]")
                    y_i = model.getVarByName(f"y_world_[{index_i}]")
                    l_i, w_i, h_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
                    # l_i, w_i, h_i = model.getVarByName(f"bdb_l_[{index_i}]"), model.getVarByName(f"bdb_w_[{index_i}]"), o_i['height']
                if o_j in O_func:
                    x_j = [var for var in model.getVars() if "x_func" in var.VarName][index_j]
                    y_j = [var for var in model.getVars() if "y_func" in var.VarName][index_j]
                    l_j, w_j, h_j = o_j['bdb_l'], o_j['bdb_w'], o_j['bdb_h']
                    # l_j, w_j, h_j = [var for var in model.getVars() if "bdb_l_func" in var.VarName][index_j], [var for var in model.getVars() if "bdb_w_func" in var.VarName][index_j], o_i['height']
                else:
                    x_j = model.getVarByName(f"x_world_[{index_j}]")
                    y_j = model.getVarByName(f"y_world_[{index_j}]")
                    l_j, w_j, h_j = o_j['bdb_l'], o_j['bdb_w'], o_j['bdb_h']
                    # l_j, w_j, h_j = model.getVarByName(f"bdb_l_[{index_j}]"), model.getVarByName(f"bdb_w_[{index_j}]"), o_i['height']

                x_i_x_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_ground_[{i}]")
                y_i_y_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p_ground_[{i}]")
                abs_x_i_x_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_ground_[{i}]")
                abs_y_i_y_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_ground_[{i}]")
                c_1, c_2 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_ground_[{i}]"), \
                           model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_2_ground_[{i}]")

                model.addConstr(x_i_x_p == x_i - x_j, name=f"coll_x_d_ground_[{i}]")
                model.addConstr(y_i_y_p == y_i - y_j, name=f"coll_y_d_ground_[{i}]")
                model.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_ground_[{i}]")
                model.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_ground_[{i}]")
                model.addConstr(((l_j + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_ground_[{i}]")
                model.addConstr(((w_j + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_ground_[{i}]")
                model.addConstr(c_1 + c_2 >= 1, name=f"coll_c_ground_[{i}]")
    # 2)Objects on the ceiling
    for i in range(len(up_obj)-1):
        o_i = up_obj[i]
        for j in range(i+1, len(up_obj)):
            o_j = up_obj[j]
            if (o_i != o_j) and o_i['on_the_ceiling'] == True and o_j['on_the_ceiling'] == True:
                index_i = o_i['obj_index']
                index_j = o_j['obj_index']
                x_i = model.getVarByName(f"x_world_[{index_i}]")
                y_i = model.getVarByName(f"y_world_[{index_i}]")
                x_j = model.getVarByName(f"x_world_[{index_j}]")
                y_j = model.getVarByName(f"y_world_[{index_j}]")
                # l_i, w_i, h_i = model.getVarByName(f"bdb_l_[{index_i}]"), model.getVarByName(f"bdb_w_[{index_i}]"), o_i['height']
                # l_j, w_j, h_j = model.getVarByName(f"bdb_l_[{index_j}]"), model.getVarByName(f"bdb_w_[{index_j}]"), o_i['height']
                l_i, w_i, h_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
                l_j, w_j, h_j = o_j['bdb_l'], o_j['bdb_w'], o_j['bdb_h']

                x_i_x_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_ceiling_[{i}]")
                y_i_y_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p_ceiling_[{i}]")
                abs_x_i_x_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_ceiling_[{i}]")
                abs_y_i_y_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_ceiling_[{i}]")
                c_1, c_2 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_ceiling_[{i}]"), \
                           model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_2_ceiling_[{i}]")

                model.addConstr(x_i_x_p == x_i - x_j, name=f"coll_x_d_ceiling_[{i}]")
                model.addConstr(y_i_y_p == y_i - y_j, name=f"coll_y_d_ceiling_[{i}]")
                model.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_ceiling_[{i}]")
                model.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_ceiling_[{i}]")
                model.addConstr(((l_j + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_ceiling_[{i}]")
                model.addConstr(((w_j + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_ceiling_[{i}]")
                model.addConstr(c_1 + c_2 >= 1, name=f"coll_c_ceiling_[{i}]")
    model.update()
    return model


def optimize_global_add_out_of_bound_constraint_by_obj(model, O_func, O_other, rl, rw, obj_func_num, room_type):
    for i in range(len(O_func)):
        x_i = [var for var in model.getVars() if "x_func" in var.VarName][i]
        y_i = [var for var in model.getVars() if "y_func" in var.VarName][i]
        model.addConstr(x_i >= 0, name=f"o_f_0_[{i}]")
        model.addConstr(x_i <= rl, name=f"o_f_1_[{i}]")
        model.addConstr(y_i >= 0, name=f"o_f_2_[{i}]")
        model.addConstr(y_i <= rw, name=f"o_f_3_[{i}]")

    for i in range(len(O_other)):
        l_i, w_i, h_i = O_other[i]['bdb_l'], O_other[i]['bdb_w'], O_other[i]['bdb_h']
        x_i_world = model.getVarByName(f"x_world_[{i}]")
        y_i_world = model.getVarByName(f"y_world_[{i}]")

        obj3_x = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3_x_{i}")
        obj3_y = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3_y_{i}")
        l_i_2_min = min(l_i / 2, rl - l_i / 2)
        l_i_2_max = max(l_i / 2, rl - l_i / 2)
        r_i_2_min = min(w_i / 2, rw - w_i / 2)
        r_i_2_max = max(w_i / 2, rw - w_i / 2)
        try:
           model.addGenConstrPWL(x_i_world, obj3_x, [l_i_2_min - 1, l_i_2_min, l_i_2_max, l_i_2_max + 1], [1, 0, 0, 1],
                           name=f"set_obj3_x_{i}")
        except:
            ipdb.set_trace()
        model.addGenConstrPWL(y_i_world, obj3_y, [r_i_2_min - 1, r_i_2_min, r_i_2_max, r_i_2_max + 1], [1, 0, 0, 1],
                           name=f"set_obj3_y_{i}")
        if room_type == 'bedroom':
            model.setObjectiveN(obj3_x, index=obj_func_num, weight=2, priority=10)#  priority=5
            obj_func_num += 1
            model.setObjectiveN(obj3_y, index=obj_func_num, weight=2, priority=10)
            obj_func_num += 1
        else:
            model.setObjectiveN(obj3_x, index=obj_func_num, weight=2, priority=5)
            obj_func_num += 1
            model.setObjectiveN(obj3_y, index=obj_func_num, weight=2, priority=5)
            obj_func_num += 1
    model.update()
    return model, obj_func_num


def optimize_global_by_obj(rl, rw, area, rel_pos_constraint = None, room_type = 'bedroom'):
    # with gp.Env() as env, gp.Model(env=env) as m:
       O_func, O_other = load_area_gpt_output(area)
        if len(O_func) + len(O_other) >= 8:
            return None, None

        # Create a new model
        m = gp.Model()
        obj_func_num = 0

        # Set objective function for func_obj
        m, obj_func_num = optimize_global_set_obj_function_for_func_obj_by_obj(m, O_func, rl, rw, obj_func_num)

        # Add func obj constraint
        m = optimize_global_add_func_obj_constraint_by_obj(m, O_func, rl, rw)

        # Set objective function for other_obj
        m, up_obj, down_obj, obj_func_num = optimize_global_set_obj_function_for_other_obj_by_obj(m, O_func, O_other,
                                                                                                  rl, rw, obj_func_num,
                                                                                                  rel_pos_constraint,
                                                                                                  room_type)

        # # Add Out of bound constraint
        m, obj_func_num = optimize_global_add_out_of_bound_constraint_by_obj(m, O_func, O_other, rl, rw, obj_func_num,
                                                                             room_type)

        # Add collision constraint
        m = optimize_global_add_collision_constraint_by_obj(m, O_func, O_other, up_obj, down_obj, rl, rw)

        # Solve it!
        m.Params.TimeLimit = 300  # 10 minutes
        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            print("__________________________________________________________________________")
            all_P = []
            for i in range(len(O_func)):
                o_i = O_func[i]
                h_i = o_i['height']
                x_i = [var for var in m.getVars() if "x_func" in var.VarName][i]
                y_i = [var for var in m.getVars() if "y_func" in var.VarName][i]
                z_i = h_i / 2
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i.x, y_i.x, z_i
                all_P.append(o_i)
            for i in range(len(O_other)):
                o_i = O_other[i]
                h_i = o_i['height']
                func_index = o_i['func_obj_index']
                x_i_world = m.getVarByName(f"x_world_[{i}]")
                y_i_world = m.getVarByName(f"y_world_[{i}]")
                # ori_i = m.getVarByName(f"bdb_ori_[{i}]")
                if o_i['on_the_ceiling']:
                    try:
                        h_i_func = O_func[func_index]['bdb_h']
                        d_i = m.getVarByName(f"d_[{i}]")
                        o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2
                    except:
                        z_i_world = m.getVarByName(f"z_world_[{i}]")
                        ipdb.set_trace()
                        o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, z_i_world.x
                else:
                    o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, h_i / 2
                all_P.append(o_i)
            # object_list = parse_gpt_output_step_by_step(all_P)
            return all_P, True

        elif m.status == GRB.INFEASIBLE:
            print("Model Status: The model is not feasible")
            m.computeIIS()
            m.write("delete_model1.ilp")
            ipdb.set_trace()
            return [], False
        elif m.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            ipdb.set_trace()
            return [], False
        else:
            print("Model status: Optimal solution not found")
            ipdb.set_trace()
            all_P = []
            for i in range(len(O_func)):
                o_i = O_func[i]
                h_i = o_i['height']
                x_i = [var for var in m.getVars() if "x_func" in var.VarName][i]
                y_i = [var for var in m.getVars() if "y_func" in var.VarName][i]
                z_i = h_i / 2
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i.x, y_i.x, z_i
                all_P.append(o_i)
            for i in range(len(O_other)):
                o_i = O_other[i]
                h_i = o_i['height']
                func_index = o_i['func_obj_index']
                x_i_world = m.getVarByName(f"x_world_[{i}]")
                y_i_world = m.getVarByName(f"y_world_[{i}]")
                # ori_i = m.getVarByName(f"bdb_ori_[{i}]")
                if o_i['on_the_ceiling']:
                    try:
                        h_i_func = O_func[func_index]['bdb_h']
                        d_i = m.getVarByName(f"d_[{i}]")
                        o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2
                    except:
                        z_i_world = m.getVarByName(f"z_world_[{i}]")
                        o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, z_i_world.x
                else:
                    o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i_world.x, y_i_world.x, h_i / 2
                all_P.append(o_i)
            # object_list = parse_gpt_output_step_by_step(all_P)
            return all_P, False


def optimize_gola_by_area_by_every_area(object_global, area_local, area_data, type = 'one-time', interactive_num = 0):

    object_list = []  # Save the object's coordinates in the world coordinate system
    area_object_list = []  # Save the coordinates of the object in the corresponding area coordinate system
    # area_ori = area_local['bdb_ori']
    # R = roty_3d(-area_ori / 180. * np.pi)
    area_ori = area_local['bdb_ori']
    R = area_local['bdb_ori'] - area_local['start_orientation']
    R = roty_3d(-R / 180. * np.pi)
    area_center_x = area_local['old_length_width'][0]
    area_center_y = area_local['old_length_width'][1]
    area_x = area_local['bdb_x']
    area_y = area_local['bdb_y']
    area_local['left'] = area_x
    area_local['top'] = area_y
    area_local['orientation'] = area_ori

    for obj in object_global:
        # area_obj_ori = 0
        area_obj_ori = object_global[0]['bdb_ori']
        world_obj_ori = copy.deepcopy(area_ori)
        if ('chair' in obj['label'] or 'stool' in obj['label']) and 'rel_position' in obj.keys():
            # ipdb.set_trace()
            if 'front' in obj['rel_position'][1]:
                # area_obj_ori = 180
                if area_obj_ori >= 180:
                    area_obj_ori = area_obj_ori - 180
                else:
                    area_obj_ori = area_obj_ori + 180
                if area_ori >= 180:
                    world_obj_ori = world_obj_ori - 180
                else:
                    world_obj_ori = world_obj_ori + 180

        area_object_list.append(
            [obj['label'],
            {
                "length": obj['length'],
                "width": obj['width'],
                "height": obj['height'],
                "left": obj['bdb_x'],
                "top": obj['bdb_y'],
                "depth": obj['bdb_z'],
                "orientation": area_obj_ori,
                "description":obj['description']
            }
        ])

        area_coordinate = np.array([obj['bdb_x'], obj['bdb_y'], obj['bdb_z']]) - [area_center_x, area_center_y, 0]

        world_coordinate = area_coordinate @ R + [area_x, area_y, 0]
        world_obj = [obj['label'],
                    {
                        "length": obj['length'],
                        "width": obj['width'],
                        "height": obj['height'],
                        "left": world_coordinate[0],
                        "top": world_coordinate[1],
                        "depth": world_coordinate[2],
                        "orientation": world_obj_ori,
                        "description": obj['description'],
                    }]
        ipdb.set_trace()
        obj['left'], obj['top'], obj['depth'], obj['orientation'] = world_coordinate[0], world_coordinate[1], world_coordinate[2], world_obj_ori

        object_list.append(world_obj)

    try:
        area_data['bdb_x'], area_data['bdb_y'], area_data['bdb_z'] = area_local['bdb_x'], area_local['bdb_y'], area_local['bdb_z']
        area_data['bdb_l'], area_data['bdb_w'], area_data['bdb_h'] = area_local['bdb_l'], area_local['bdb_w'], area_local['bdb_h']
        area_data['bdb_ori'] = area_local['bdb_ori']
        area_data['orientation'] = area_local['orientation']
        area_data['tight_length'] = area_local['length']
        area_data['tight_width'] = area_local['width']

        func_obj = area_data['children']
        func_obj['bdb_x'], func_obj['bdb_y'], func_obj['bdb_z'] = object_global[0]['bdb_x'], object_global[0]['bdb_y'], \
                                                                  object_global[0]['bdb_z']
        func_obj['bdb_l'], func_obj['bdb_w'], func_obj['bdb_h'] = object_global[0]['bdb_l'], object_global[0]['bdb_w'], \
                                                                  object_global[0]['bdb_h']
        func_obj['left'], func_obj['top'], func_obj['depth'] = object_global[0]['left'], object_global[0]['top'], \
                                                               object_global[0]['depth']
        func_obj['orientation'] = object_global[0]['orientation']
        func_obj['bdb_ori'] = object_global[0]['bdb_ori']
        if 'children' in func_obj.keys():
            for j in range(len(func_obj['children'])):
                flag = 0
                for k in range(len(object_global)):
                    obj = func_obj['children'][j]
                    new_obj = object_global[k]
                    if obj['label'] == new_obj['label']:
                        obj = func_obj['children'][j]
                        obj['bdb_x'], obj['bdb_y'], obj['bdb_z'] = new_obj['bdb_x'], new_obj['bdb_y'], new_obj['bdb_z']
                        obj['bdb_l'], obj['bdb_w'], obj['bdb_h'] = new_obj['bdb_l'], new_obj['bdb_w'], new_obj['bdb_h']
                        obj['left'], obj['top'], obj['depth'] = new_obj['left'], new_obj['top'], new_obj['depth']
                        obj['orientation'] = new_obj['orientation']
                        obj['bdb_ori'] = new_obj['bdb_ori']
                        flag = 1
                        break
                if flag == 0:
                    print('Object has been deleted')
                    obj['is_delete'] = True
                    ipdb.set_trace()
    except:
        ipdb.set_trace()
    return area_data, object_list, area_object_list


def optimize_gola_by_area(val_id, prompt, data, rel_pos_constraints=None, type='one-time', interactive_num=0, room_type = 'bedroom'):
    rl, rw = float(data['length']), float(data['width'])
    all_area_object_global_list = []
    all_status = []
    
    for i in range(len(data['children'])):
        area = data['children'][i]
        if 'children' in area.keys():
            area_l = float(area['length']) # with_area_size
            area_w = float(area['width']) # with_area_size
            if rel_pos_constraints is not None:
                rel_pos_constraint = rel_pos_constraints[i]
            else:
                rel_pos_constraint = None
            object_list, status = optimize_global_by_obj(area_l, area_w, area, rel_pos_constraint, room_type = room_type)
            all_area_object_global_list.append(object_list)
            all_status.append(status)
        else:
            all_area_object_global_list.append([])
            all_status.append([])

    all_area_list = []
    for i in range(len(data['children'])):
        area = data['children'][i]
        if 'children' in area.keys() and len(all_area_object_global_list[i]) != 0:
            area_l, area_w, area_left, area_top = get_area_length_width(all_area_object_global_list[i])  # without_area_size
            area_data = {
                'label': area['label'].split(' ')[0],
                'length': area_l,
                'width': area_w,
                'old_length_width': [area['length'], area['width']],
                'area_coordinate_left': area_left,
                'area_coordinate_top': area_top,
                'obj_info': all_area_object_global_list[i]
            }
            if 'optimization' in area.keys():
                if len(area['optimization']) != 0 and area['optimization'][0] == 'area_global':
                    area_data['is_optimized'] = True
                else:
                    area_data['height'] = 0.0
                    area_data['bdb_x'] = area['bdb_x']
                    area_data['bdb_y'] = area['bdb_y']
                    area_data['bdb_z'] = area['bdb_z']
                    area_data['bdb_l'] = area['bdb_l']
                    area_data['bdb_w'] = area['bdb_w']
                    area_data['bdb_h'] = area['bdb_h']
                    # area_data['orientation'] = area['orientation']
                    area_data['bdb_ori'] = area['orientation']
                    area_data['is_optimized'] = False
            all_area_list.append(area_data)
        else:
            ipdb.set_trace()
            all_area_list.append({})

    area_loc_list, _ = optimize_global_by_area(rl, rw, all_area_list, room_type)

    if len(area_loc_list) == 0:
       print("Area global optimization failed. Perform local optimization !")
       area_loc_list, _ = optimize_local_by_area(rl, rw, all_area_list, room_type)

    if len(area_loc_list) == len(data['children']) and len(all_area_object_global_list) == len(area_loc_list):
        print("Finish Optimized!")
    else:
        ipdb.set_trace()

    all_object_list = []
    all_area_object_list = []
    for i in range(len(area_loc_list)):
        area = area_loc_list[i]
        area_object_list = []
        if len(area) != 0:
            area_center_x = area['area_coordinate_left']
            area_center_y = area['area_coordinate_top']
            area_x = area['bdb_x']
            area_y = area['bdb_y']
            area['orientation'] = area['bdb_ori']
            R = area['bdb_ori']
            R = roty_3d(-R / 180. * np.pi)
            for obj in all_area_object_global_list[i]:
                area_obj_ori = obj['bdb_ori']
                area_object_list.append([
                    obj['label'],
                    {
                        "length": obj['length'],
                        "width": obj['width'],
                        "height": obj['height'],
                        "left": obj['bdb_x'],
                        "top": obj['bdb_y'],
                        "depth": obj['bdb_z'],
                        "orientation": area_obj_ori,
                        "description": obj['description']
                    }
                ])
                # Translate the coordinate origin within the region to the center of the region
                area_coordinate = np.array([obj['bdb_x'], obj['bdb_y'], obj['bdb_z']]) - [area_center_x,
                                                                                          area_center_y, 0]
                # Rotate and translate the area into the room
                world_coordinate = area_coordinate @ R + [area_x, area_y, 0]
                world_obj_ori = copy.deepcopy(area['bdb_ori']) + area_obj_ori
                if world_obj_ori >= 360:
                    world_obj_ori -= 360
                world_obj = [obj['label'],
                             {
                                 "length": obj['length'],
                                 "width": obj['width'],
                                 "height": obj['height'],
                                 "left": world_coordinate[0],
                                 "top": world_coordinate[1],
                                 "depth": world_coordinate[2],
                                 "orientation": world_obj_ori,
                                 "description": obj['description']
                             }]
                obj['left'], obj['top'], obj['depth'], obj['orientation'] = world_coordinate[0], world_coordinate[
                    1], \
                                                                            world_coordinate[2], world_obj_ori

                all_object_list.append(world_obj)
            all_area_object_list.append(area_object_list)
        else:
            all_object_list.append([])
            all_area_object_list.append(area_object_list)

    # Updated hierarchy data
    for i in range(len(data['children'])):
        area = data['children'][i]
        if area_loc_list[i] != {}:
            area['bdb_x'], area['bdb_y'], area['bdb_z'] = area_loc_list[i]['bdb_x'], area_loc_list[i]['bdb_y'], \
                                                          area_loc_list[i]['bdb_z']
            area['bdb_l'], area['bdb_w'], area['bdb_h'] = area_loc_list[i]['bdb_l'], area_loc_list[i]['bdb_w'], \
                                                          area_loc_list[i]['bdb_h']
            area['bdb_ori'] = area_loc_list[i]['bdb_ori']
            area['orientation'] = area_loc_list[i]['orientation']
            area['tight_length'] = area_loc_list[i]['length']
            area['tight_width'] = area_loc_list[i]['width']

            func_obj = area['children']
            func_obj['bdb_x'], func_obj['bdb_y'], func_obj['bdb_z'] = all_area_object_global_list[i][0]['bdb_x'], \
                                                                      all_area_object_global_list[i][0]['bdb_y'], \
                                                                      all_area_object_global_list[i][0]['bdb_z']
            func_obj['bdb_l'], func_obj['bdb_w'], func_obj['bdb_h'] = all_area_object_global_list[i][0]['bdb_l'], \
                                                                      all_area_object_global_list[i][0]['bdb_w'], \
                                                                      all_area_object_global_list[i][0]['bdb_h']
            func_obj['left'], func_obj['top'], func_obj['depth'] = all_area_object_global_list[i][0]['left'], \
                                                                   all_area_object_global_list[i][0]['top'], \
                                                                   all_area_object_global_list[i][0]['depth']
            func_obj['orientation'] = all_area_object_global_list[i][0]['orientation']
            func_obj['bdb_ori'] = all_area_object_global_list[i][0]['bdb_ori']
            if 'children' in func_obj.keys():
                for j in range(len(func_obj['children'])):
                    flag = 0
                    for k in range(len(all_area_object_global_list[i])):
                        obj = func_obj['children'][j]
                        new_obj = all_area_object_global_list[i][k]
                        if obj['label'] == new_obj['label']:
                            obj = func_obj['children'][j]
                            obj['bdb_x'], obj['bdb_y'], obj['bdb_z'] = all_area_object_global_list[i][k]['bdb_x'], \
                                                                       all_area_object_global_list[i][k]['bdb_y'], \
                                                                       all_area_object_global_list[i][k]['bdb_z']
                            obj['bdb_l'], obj['bdb_w'], obj['bdb_h'] = all_area_object_global_list[i][k]['bdb_l'], \
                                                                       all_area_object_global_list[i][k]['bdb_w'], \
                                                                       all_area_object_global_list[i][k]['bdb_h']
                            obj['left'], obj['top'], obj['depth'] = all_area_object_global_list[i][k]['left'], \
                                                                    all_area_object_global_list[i][k]['top'], \
                                                                    all_area_object_global_list[i][k]['depth']
                            obj['orientation'] = all_area_object_global_list[i][k]['orientation']
                            obj['bdb_ori'] = all_area_object_global_list[i][k]['bdb_ori']
                            flag = 1
                            break
                    if flag == 0:
                        print('Object has been deleted')
                        obj['is_delete'] = True
                        ipdb.set_trace()
        else:
            print('Area has been deleted')
            area['is_delete'] = True
            ipdb.set_trace()

    return data, all_object_list, all_area_object_list, all_status

def optimize_local_other_obj_by_area(o_i, P, up_P, down_P, O_func, rl, rw, rel_pos_constraint, room_type):
    r_i, func_index = o_i['rel_position'], o_i['func_obj_index']
    l_i, w_i, h_i= o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h']
    l_i_func, w_i_func, h_i_func, x_i_func, y_i_func, z_i_func, ori_i = O_func[func_index]['bdb_l'], \
                                                                 O_func[func_index]['bdb_w'], \
                                                                 O_func[func_index]['bdb_h'], \
                                                                 O_func[func_index]['bdb_x'], \
                                                                 O_func[func_index]['bdb_y'], \
                                                                 O_func[func_index]['bdb_z'], \
                                                                 O_func[func_index]['bdb_ori']
    o_label, rel = o_i['label'], r_i[1]
    rel = difflib.get_close_matches(rel, ALL_RELATIONS, cutoff=0.0)[0]
    is_use, rel_pos_constraint_i = use_rel_pos_constraint(o_i, rel_pos_constraint, room_type)
    # Create a new model
    m1 = gp.Model()

    if rel_pos_constraint is not None and is_use:
        # o_i['rel_position'][1] = rel_pos_constraint_i['relative_position'][1]
        rel_x_i = rel_pos_constraint_i['relative_coordinate'][0]
        rel_y_i = rel_pos_constraint_i['relative_coordinate'][1]
        rel_z_i = rel_pos_constraint_i['relative_coordinate'][2]
        rel_ori_i = rel_pos_constraint_i['rel_ori']

        if rel_ori_i < 0:
            rel_ori_i += 360
        o_i['bdb_ori'] = rel_ori_i
        l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
        o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i

        # Create variables
        x_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x")
        y_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y")
        x_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world")
        y_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world")

        # Set objective function
        obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name=f"obj1")
        obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj2")
        m1.addConstr(obj1 == ((x_i * x_i) + (y_i * y_i)), name=f"set_obj1")
        m1.addConstr(obj2 == ((x_i - rel_x_i) * (x_i - rel_x_i) + (y_i - rel_y_i) * (y_i - rel_y_i)),
                        name=f"set_obj2")
        m1.setObjectiveN(obj2, index=0, priority=10)
        m1.setObjectiveN(obj1, index=1, priority=5)

        # x_i, y_i ->x_i_world, y_i_world
        cos_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"cos")
        sin_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"sin")
        theta_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"theta")
        m1.addConstr(theta_i == -ori_i / 180 * math.pi, name=f"theta_i")
        m1.addGenConstrCos(theta_i, cos_i, name=f"cos_theta_i_world")
        m1.addGenConstrSin(theta_i, sin_i, name=f"sin_theta_i_world")
        m1.addConstr(x_i_world == x_i_func + (x_i * cos_i - y_i * sin_i), name=f"x_i_world")
        m1.addConstr(y_i_world == y_i_func + (x_i * sin_i + y_i * cos_i), name=f"y_i_world")

        obj3_x = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3_x")
        obj3_y = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3_y")
        l_i_2_min = min(l_i / 2, rl - l_i / 2)
        l_i_2_max = max(l_i / 2, rl - l_i / 2)
        r_i_2_min = min(w_i / 2, rw - w_i / 2)
        r_i_2_max = max(w_i / 2, rw - w_i / 2)
        m1.addGenConstrPWL(x_i_world, obj3_x, [l_i_2_min - 1, l_i_2_min, l_i_2_max, l_i_2_max + 1], [1, 0, 0, 1],
                           name=f"set_obj3_x")
        m1.addGenConstrPWL(y_i_world, obj3_y, [r_i_2_min - 1, r_i_2_min, r_i_2_max, r_i_2_max + 1], [1, 0, 0, 1],
                           name=f"set_obj3_y")
        # m1.addGenConstrPWL(x_i_world, obj3_x, [l_i / 2 - 1, l_i / 2, rl - l_i / 2, rl - l_i / 2 + 1], [1, 0, 0, 1],
        #                    name=f"set_obj3_x")
        # m1.addGenConstrPWL(y_i_world, obj3_y, [w_i / 2 - 1, w_i / 2, rw - w_i / 2, rw - w_i / 2 + 1], [1, 0, 0, 1],
        #                    name=f"set_obj3_y")
        if room_type == 'livingroom':
            m1.setObjectiveN(obj3_x, index=2, weight=2, priority=10)
            m1.setObjectiveN(obj3_y, index=3, weight=2, priority=10)
        else:
            m1.setObjectiveN(obj3_x, index=2, weight=3, priority=5)
            m1.setObjectiveN(obj3_y, index=3, weight=3, priority=5)

        if rel == 'above' or rel == 'on':
            o_i['on_the_ceiling'] = True
            d_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"d")
            m1.addConstr(d_i == rel_z_i, name=f"as_z_i_world")

        # Add Constraints
        # Collision
        if rel == 'above' or rel == 'on top of':
            P_list = up_P
        elif rel == 'below':
            P_list = down_P
        else:
            P_list = P

        i = 0
        for o_p in P_list:

            x_i_x_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p[{i}]")
            y_i_y_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
            abs_x_i_x_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p[{i}]")
            abs_y_i_y_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p[{i}]")
            c_1, c_2 = m1.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), m1.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                          name=f"c_2[{i}]")
            m1.addConstr(x_i_x_p == x_i_world - o_p['bdb_x'], name=f"coll_x_i-x_p[{i}]")
            m1.addConstr(y_i_y_p == y_i_world - o_p['bdb_y'], name=f"coll_y_i-y_p[{i}]")
            m1.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_i-x_p[{i}]")
            m1.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_i-y_p[{i}]")
            m1.addConstr(((o_p['bdb_l'] + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
            m1.addConstr(((o_p['bdb_w'] + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
            m1.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
            i += 1



        # Solve it!
        m1.optimize()
        if m1.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            if rel == 'above' or rel == 'on top of':
                return x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2, True, False, is_use
            elif rel == 'below':
                return x_i_world.x, y_i_world.x, h_i / 2, False, True, is_use
            else:
                return x_i_world.x, y_i_world.x, h_i / 2, False, False, is_use
        elif m1.status == GRB.INFEASIBLE:
            print("Model Status: The model is not feasible")
            return None
        elif m1.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            return None
        else:
            print("Model status: Optimal solution not found")
            return None
    else:
        # Create variables
        d_i = m1.addVar(lb=0.0, ub=math.sqrt(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="d")
        theta_i = m1.addVar(lb=0, ub=2 * math.pi, vtype=gp.GRB.CONTINUOUS, name="theta")

        theta_i_world = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="theta_world")
        x_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x")
        y_i = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="y")
        x_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="x_world")
        y_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="y_world")
        if "touching" in rel:
            weight1, weight2, delta = 1, 1, 0
        else:
            weight1, weight2, delta = 1, 1, 0

        # Set objective function
        if rel == 'left of' or rel == "left touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == theta_i * theta_i * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == "right of" or rel == "right touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (math.pi - theta_i) * (math.pi - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == 'in front of' or rel == "front touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (math.pi / 2 - theta_i) * (math.pi / 2 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == 'behind' or rel == "behind touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (3 * math.pi / 2 - theta_i) * (3 * math.pi / 2 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the right front of' or rel == "right front touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (3 * math.pi / 4 - theta_i) * (3 * math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the left front of' or rel == "left front touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (math.pi / 4 - theta_i) * (math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the right behind of' or rel == "right behind touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (5 * math.pi / 4 - theta_i) * (5 * math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel in 'in the left behind of' or rel == "left behind touching":
            obj1 = m1.addVar(lb=0.0, ub=(rl * rl + rw * rw), vtype=gp.GRB.CONTINUOUS, name="obj1")
            m1.addConstr(obj1 == (d_i - delta) * (d_i - delta) * 1 / 2, name='set_obj1')
            obj2 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="obj2")
            m1.addConstr(obj2 == (7 * math.pi / 4 - theta_i) * (7 * math.pi / 4 - theta_i) * 1 / 2, name='set_obj2')
            m1.setObjectiveN(obj2, index=0, priority=10)
            m1.setObjectiveN(obj1, index=1, priority=5)

            # d_i, theta_i->x_i,y_i
            m1.addConstr(theta_i_world == theta_i + (2 * math.pi - ori_i / 180 * math.pi))
            m1.addGenConstrCos(theta_i_world, x_i)
            m1.addGenConstrSin(theta_i_world, y_i)
            m1.addConstr(x_i_world == x_i_func + d_i * x_i)
            m1.addConstr(y_i_world == y_i_func + d_i * y_i)
        elif rel == 'above' or rel == 'on top of':
            if rel == 'on top of':
                delta = 0
            else:
                delta = min(rl, rw) / 4
            if len(up_P) == 0:
                return x_i_func, y_i_func, (h_i_func + h_i) / 2 + delta + h_i_func / 2, True, False, is_use
            else:
                obj1 = m1.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="obj1")
                m1.addConstr(obj1 == ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2, name='set_obj1')
                obj2 = m1.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="obj2")
                m1.addConstr(obj2 == (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (y_i_world - y_i_func), name='set_obj2')
                # obj1 = ((h_i_func + h_i) / 2 + delta - d_i) * ((h_i_func + h_i) / 2 + delta - d_i) * 1 / 2
                # obj2 = (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (y_i_world - y_i_func)
                m1.setObjectiveN(obj1 + obj2, index=1, priority=10)
        elif rel == 'below':
            if len(down_P) == 0:
                return x_i_func, y_i_func, h_i / 2, False, True, is_use
            else:
                obj1 = m1.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name="obj1")
                m1.addConstr(obj1 == (x_i_world - x_i_func) * (x_i_world - x_i_func) + (y_i_world - y_i_func) * (
                            y_i_world - y_i_func), name='set_obj1')
                m1.setObjectiveN(obj1, index=1, priority=10)
        else:
            ipdb.set_trace()

        # Out of bound
        obj3_x = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3_x")
        obj3_y = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3_y")
        l_i_2_min = min(l_i / 2, rl - l_i / 2)
        l_i_2_max = max(l_i / 2, rl - l_i / 2)
        r_i_2_min = min(w_i / 2, rw - w_i / 2)
        r_i_2_max = max(w_i / 2, rw - w_i / 2)
        m1.addGenConstrPWL(x_i_world, obj3_x, [l_i_2_min - 1, l_i_2_min, l_i_2_max, l_i_2_max + 1], [1, 0, 0, 1],
                           name=f"set_obj3_x")
        m1.addGenConstrPWL(y_i_world, obj3_y, [r_i_2_min - 1, r_i_2_min, r_i_2_max, r_i_2_max + 1], [1, 0, 0, 1],
                           name=f"set_obj3_y")
        # m1.addGenConstrPWL(x_i_world, obj3_x, [l_i / 2 - 1, l_i / 2, rl - l_i / 2, rl - l_i / 2 + 1], [1, 0, 0, 1],
        #                        name=f"set_obj3_x")
        # m1.addGenConstrPWL(y_i_world, obj3_y, [w_i / 2 - 1, w_i / 2, rw - w_i / 2, rw - w_i / 2 + 1], [1, 0, 0, 1],
        #                        name=f"set_obj3_y")
        if room_type == 'livingroom':
            m1.setObjectiveN(obj3_x, index=2, weight=2, priority=10)
            m1.setObjectiveN(obj3_y, index=3, weight=2, priority=10)
        else:
            m1.setObjectiveN(obj3_x, index=2, weight=3, priority=5)
            m1.setObjectiveN(obj3_y, index=3, weight=3, priority=5)

        # Add constraints
        # Collision
        if rel == 'above' or rel == 'on top of':
            P_list = up_P
        else:
            P_list = P
        i = 0
        for o_p in P_list:
            x_i_x_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_[{i}]")
            y_i_y_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
            abs_x_i_x_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p[{i}]")
            abs_y_i_y_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p[{i}]")
            c_1, c_2 = m1.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), m1.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                          name=f"c_2[{i}]")
            m1.addConstr(x_i_x_p == x_i_world - o_p['bdb_x'], name=f"coll_x_i-x_p_[{i}]")
            m1.addConstr(y_i_y_p == y_i_world - o_p['bdb_y'], name=f"coll_y_i-y_p[{i}]")
            m1.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_i-x_p_[{i}]")
            m1.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_i-y_p[{i}]")
            m1.addConstr(((o_p['bdb_l'] + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
            m1.addConstr(((o_p['bdb_w'] + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
            m1.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
            i += 1
        # Out of bound
        # m1.addConstr(x_i_world >= l_i / 2, name='o_0')
        # m1.addConstr(x_i_world <= rl - l_i / 2, name='o_1')
        # m1.addConstr(y_i_world >= w_i / 2, name='o_2')
        # m1.addConstr(y_i_world <= rw - w_i / 2, name='o_3')

        # Solve it!
        m1.optimize()
        if m1.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            print("__________________________________________________________________________")
            if rel == 'above' or rel == 'on top of':
                return x_i_world.x, y_i_world.x, d_i.x + h_i_func / 2, True, False, is_use
            elif rel == 'below':
                return x_i_world.x, y_i_world.x, h_i / 2, False, True, is_use
            else:
                return x_i_world.x, y_i_world.x, h_i / 2, False, False, is_use
        elif m1.status == GRB.INFEASIBLE:
            print("Model Status: The model is not feasible")
            return None
        elif m1.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            return None
        else:
            print("Model status: Optimal solution not found")
            return None


def frozen_local_other_obj_by_area(o_i, P, up_P, down_P, rl, rw):
    r_i, func_index = o_i['rel_position'], o_i['func_obj_index']
    l_i, w_i, h_i, ori_i = o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'], o_i['bdb_ori']
    old_x_i, old_y_i, old_z_i = o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z']

    o_label, rel = o_i['label'], r_i[1]
    rel = difflib.get_close_matches(rel, ALL_RELATIONS, cutoff=0.0)[0]

    # Create a new model
    m1 = gp.Model()

    # Create variables
    x_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_world")
    y_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_world")
    z_i_world = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"z_world")

    # Set objective function
    obj3 = m1.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"obj3")
    m1.addConstr(obj3 == ((x_i_world - old_x_i) * (x_i_world - old_x_i) + (y_i_world - old_y_i) * (y_i_world - old_y_i)
                          + (z_i_world - old_z_i) * (z_i_world - old_z_i)),
                 name=f"set_obj3") 
    m1.setObjectiveN(obj3, index=0, weight=2, priority=10)

    # Add constraints
    # Collision
    if rel == 'above' or rel == 'on top of':
        P_list = up_P
    elif rel == 'below':
        P_list = down_P
    else:
        P_list = P
    i = 0
    for o_p in P_list:
        x_i_x_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_[{i}]")
        y_i_y_p = m1.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
        abs_x_i_x_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p[{i}]")
        abs_y_i_y_p = m1.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p[{i}]")
        c_1, c_2 = m1.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), m1.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                      name=f"c_2[{i}]")
        m1.addConstr(x_i_x_p == x_i_world - o_p['bdb_x'], name=f"coll_x_i-x_p_[{i}]")
        m1.addConstr(y_i_y_p == y_i_world - o_p['bdb_y'], name=f"coll_y_i-y_p[{i}]")
        m1.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_i-x_p_[{i}]")
        m1.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_i-y_p[{i}]")
        m1.addConstr(((o_p['bdb_l'] + l_i) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
        m1.addConstr(((o_p['bdb_w'] + w_i) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
        m1.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
        i += 1
    # Solve it!
    m1.optimize()
    if m1.status == GRB.OPTIMAL:
        print("Model status: Optimal solution found")
        print("__________________________________________________________________________")
        if rel == 'above' or rel == 'on top of':
            return x_i_world.x, y_i_world.x, z_i_world.x, True, False
        elif rel == 'below':
            return x_i_world.x, y_i_world.x, z_i_world.x, False, True
        else:
            return x_i_world.x, y_i_world.x, z_i_world.x, False, False
    elif m1.status == GRB.INFEASIBLE:
        print("Model Status: The model is not feasible")
        return None
    elif m1.status == GRB.UNBOUNDED:
        print("Model Status: The model is unbounded")
        return None
    else:
        print("Model status: Optimal solution not found")
        return None


def optimize_local_obj_by_area(rl, rw, area, rel_pos_constraint, room_type = 'bedroom'):
    O_func, O_other = load_area_gpt_output(area)
    all_P = []  # Record all placed objects
    P = []  # Objects recorded on the floor
    up_P = []  # Objects recorded on the ceiling
    down_P = []  # Objects under a certain object (e.g., rug)
    # First, place the functional objects (functional objects are generally on a two-dimensional plane)
    for o_i in O_func:
        if len(P) == 0:
            if o_i['is_optimized']:
                ori_list = [0]
                time = 0
                while 1:
                    time += 1
                    ori_i = random.choice(ori_list)
                    o_i['bdb_ori'] = ori_i
                    l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                    if l_i <= rl + l_i / 2 and w_i <= rw + w_i / 2: 
                        break
                    if time > 20:
                        print(ori_i)
                        ipdb.set_trace()
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
                if 'dining_table' in o_i['label']:
                    x_i = rl / 2
                    y_i = rw / 2
                else:
                    while 1:
                        if ori_i == 0:
                            y_i = w_i / 2
                            x_i = round(random.uniform(l_i / 2, rl - l_i / 2), 2)
                        elif ori_i == 90:
                            x_i = l_i / 2
                            y_i = round(random.uniform(w_i / 2, rw - w_i / 2), 2)
                        elif ori_i == 180:
                            y_i = rw - w_i / 2
                            x_i = round(random.uniform(l_i / 2, rl - l_i / 2), 2)
                        elif ori_i == 270:
                            x_i = rl - l_i / 2
                            y_i = round(random.uniform(w_i / 2, rw - w_i / 2), 2)
                        if x_i == l_i / 2 or x_i == rl - l_i / 2 or y_i == w_i / 2 or y_i == rw - w_i / 2:
                            break
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i, y_i, h_i / 2
            o_i['is_use_rel_lib'] = False 
            P.append(o_i)
            all_P.append(o_i)
        else:
            ipdb.set_trace()

    # Then place the non-functional objects
    for o_i in O_other:
        func_index = o_i['func_obj_index']
        if o_i['is_optimized'] or O_func[func_index]['is_optimized']:
            ori_i_func = O_func[func_index]['bdb_ori']
            if 'orientation' in o_i.keys(): 
                o_i['bdb_ori'] = o_i['orientation'] - O_func[func_index]['orientation']
                if o_i['bdb_ori'] < 0:
                    o_i['bdb_ori'] += 360
                l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i
            else: 
                o_i['bdb_ori'] = ori_i_func
                l_i, w_i, h_i = get_bdbox_l_w_h(o_i)
                o_i['bdb_l'], o_i['bdb_w'], o_i['bdb_h'] = l_i, w_i, h_i

            x_i, y_i, z_i, is_up, is_down, is_use = optimize_local_other_obj_by_area(o_i, P, up_P, down_P, O_func, rl,
                                                                                     rw, rel_pos_constraint, room_type)
            o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i, y_i, z_i
            o_i['is_use_rel_lib'] = is_use 
            if is_up:
                up_P.append(o_i)
            elif is_down:
                down_P.append(o_i)
            else:
                P.append(o_i)
            all_P.append(o_i)
        else:
            try:
                x_i, y_i, z_i, is_up, is_down = frozen_local_other_obj_by_area(o_i, P, up_P, down_P, rl, rw)
                o_i['bdb_x'], o_i['bdb_y'], o_i['bdb_z'] = x_i, y_i, z_i
                o_i['is_use_rel_lib'] = False
                if is_up:
                    up_P.append(o_i)
                elif is_down:
                    down_P.append(o_i)
                else:
                     P.append(o_i)
                all_P.append(o_i)
            except:
                ipdb.set_trace()
                all_P.append({})

    return all_P, None


def optimize_global_set_area_function(model, all_area_list, rl, rw, obj_func_num):
    area_size = len(all_area_list)
    x_func = model.addVars(area_size, lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="x_func")
    y_func = model.addVars(area_size, lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="y_func")

    # Set objective function
    # 1.1 The distance between regions should be as large as possible.
    for i in range(area_size - 1):
        for j in range(i + 1, area_size):
            if all_area_list[i] != {} and all_area_list[j] != {}:
                if all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                    obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                    model.addConstr(obj_func_i_j == -((x_func[i] - x_func[j]) * (x_func[i] - x_func[j]) + (y_func[i] - y_func[j]) * (
                                y_func[i] - y_func[j])), name=f'constr_obj_func_{i}_{j}')
                    model.setObjectiveN(obj_func_i_j, index = obj_func_num, weight = 2, priority=5) #weight = 2
                    obj_func_num += 1
                elif all_area_list[i]['is_optimized'] and not all_area_list[j]['is_optimized']:
                    old_x_func, old_y_func = all_area_list[j]['bdb_x'], all_area_list[j]['bdb_y']
                    obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                    model.addConstr(obj_func_i_j == -((x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_y_func) * (
                                y_func[i] - old_y_func)), name=f'constr_obj_func_{i}_{j}')
                    model.setObjectiveN(obj_func_i_j, index=obj_func_num, weight = 2, priority=5)#weight = 2
                    obj_func_num += 1
                elif not all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                    old_x_func, old_y_func = all_area_list[i]['bdb_x'], all_area_list[i]['bdb_y']
                    obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                    model.addConstr(obj_func_i_j == -((old_x_func - x_func[j]) * (old_x_func - x_func[j]) + (old_y_func - y_func[j]) * (
                                old_y_func - y_func[j])), name=f'constr_obj_func_{i}_{j}')
                    model.setObjectiveN(obj_func_i_j, index=obj_func_num, weight = 2, priority=5)#weight = 2
                    obj_func_num += 1
                else:
                    continue
            else:
                continue

    # The frozen area should be as close as possible to the original coordinates.
    for i in range(area_size):
        if all_area_list[i] != {}:
            if not all_area_list[i]['is_optimized']:
                old_x_func, old_y_func = all_area_list[i]['bdb_x'], all_area_list[i]['bdb_y']
                # Set objective function
                frozen_obj_func = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"frozen_obj_func_{i}")
                model.addConstr(
                    frozen_obj_func == (
                                (x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_y_func) * (
                                    y_func[i] - old_y_func)), name=f"set_obj3_{i}")
                model.setObjectiveN(frozen_obj_func, weight = 3, index=obj_func_num, priority=5, name=f'frozen_constr_obj_func_{i}')#weight = 3
                obj_func_num += 1
        else:
            continue
            # ipdb.set_trace()

    model.update()
    return model, obj_func_num


def optimize_global_set_area_function_for_livingroom(model, all_area_list, rl, rw, obj_func_num):
    area_size = len(all_area_list)
    x_func = model.addVars(area_size, lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="x_func")
    y_func = model.addVars(area_size, lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="y_func")

    # Set objective function
    # 1.1 The distance between regions should be as large as possible.
    living_tv_exist = False
    for i in range(area_size - 1):
        for j in range(i + 1, area_size):
            if all_area_list[i] != {} and all_area_list[j] != {}:
                if (all_area_list[i]['label'] == 'living_area' and all_area_list[j]['label'] == 'tv_area') or \
                        (all_area_list[j]['label'] == 'living_area' and all_area_list[i]['label'] == 'tv_area'):
                    living_tv_exist = True
                    if all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                        obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                        model.addConstr(obj_func_i_j == (
                                    (x_func[i] - x_func[j]) * (x_func[i] - x_func[j]) + (y_func[i] - y_func[j]) * (
                                    y_func[i] - y_func[j])), name=f'constr_obj_func_{i}_{j}')
                        model.setObjectiveN(obj_func_i_j, priority=10, weight=5, index=obj_func_num)
                        obj_func_num += 1

                    elif all_area_list[i]['is_optimized'] and not all_area_list[j]['is_optimized']:
                        old_x_func, old_y_func = all_area_list[j]['bdb_x'], all_area_list[j]['bdb_y']
                        obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                        model.addConstr(obj_func_i_j == (
                                    (x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_y_func) * (
                                    y_func[i] - old_y_func)), name=f'constr_obj_func_{i}_{j}')
                        model.setObjectiveN(obj_func_i_j, priority=10, weight=5, index=obj_func_num)
                        obj_func_num += 1

                    elif not all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                        old_x_func, old_y_func = all_area_list[i]['bdb_x'], all_area_list[i]['bdb_y']
                        obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                        model.addConstr(obj_func_i_j == (
                                    (old_x_func - x_func[j]) * (old_x_func - x_func[j]) + (old_y_func - y_func[j]) * (
                                    old_y_func - y_func[j])), name=f'constr_obj_func_{i}_{j}')
                        model.setObjectiveN(obj_func_i_j, priority=10, weight=5, index=obj_func_num)
                        obj_func_num += 1
                    else:
                        continue
                else:
                    if all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                        obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                        model.addConstr(obj_func_i_j == -(
                                (x_func[i] - x_func[j]) * (x_func[i] - x_func[j]) + (y_func[i] - y_func[j]) * (
                                y_func[i] - y_func[j])), name=f'constr_obj_func_{i}_{j}')
                        model.setObjectiveN(obj_func_i_j, priority=5, weight=5, index=obj_func_num) #priority = 5, weight=5
                        obj_func_num += 1
                    elif all_area_list[i]['is_optimized'] and not all_area_list[j]['is_optimized']:
                        old_x_func, old_y_func = all_area_list[j]['bdb_x'], all_area_list[j]['bdb_y']
                        obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                        model.addConstr(obj_func_i_j == -(
                                (x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_y_func) * (
                                y_func[i] - old_y_func)), name=f'constr_obj_func_{i}_{j}')
                        model.setObjectiveN(obj_func_i_j,  priority=5, weight=5, index=obj_func_num)
                        obj_func_num += 1
                    elif not all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                        old_x_func, old_y_func = all_area_list[i]['bdb_x'], all_area_list[i]['bdb_y']
                        obj_func_i_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_{i}_{j}")
                        model.addConstr(obj_func_i_j == -(
                                (old_x_func - x_func[j]) * (old_x_func - x_func[j]) + (old_y_func - y_func[j]) * (
                                old_y_func - y_func[j])), name=f'constr_obj_func_{i}_{j}')
                        model.setObjectiveN(obj_func_i_j,  priority=5, weight=5, index=obj_func_num)
                        obj_func_num += 1
                    else:
                        continue
            else:
                # ipdb.set_trace()
                continue
    if living_tv_exist:
        living_tv_other_obj = []
        for i in range(area_size - 1):
            for j in range(i + 1, area_size):
                if all_area_list[i] != {} and all_area_list[j] != {}:
                    if (all_area_list[i]['label'] in ['living_area', 'tv_area'] and all_area_list[j]['label'] not in [
                        'living_area', 'tv_area']) or \
                            (all_area_list[j]['label'] in ['living_area', 'tv_area'] and all_area_list[i][
                                'label'] not in ['living_area', 'tv_area']):
                        if all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                            living_tv_other_obj.append(
                                (x_func[i] - x_func[j]) * (x_func[i] - x_func[j]) + (y_func[i] - y_func[j]) * (
                                        y_func[i] - y_func[j])
                            )
                        elif all_area_list[i]['is_optimized'] and not all_area_list[j]['is_optimized']:
                            old_x_func, old_y_func = all_area_list[j]['bdb_x'], all_area_list[j]['bdb_y']
                            living_tv_other_obj.append(
                                (x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_x_func) * (
                                        y_func[i] - old_x_func)
                            )
                        elif not all_area_list[i]['is_optimized'] and all_area_list[j]['is_optimized']:
                            old_x_func, old_y_func = all_area_list[i]['bdb_x'], all_area_list[i]['bdb_y']
                            living_tv_other_obj.append(
                                (old_x_func - x_func[j]) * (old_x_func - x_func[j]) + (old_y_func - y_func[j]) * (
                                        old_y_func - y_func[j])
                            )
                        else:
                            continue
                else:
                    # ipdb.set_trace()
                    continue
        living_tv_other_obj_func = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"obj_func_l_tv_other")
        model.addConstr(living_tv_other_obj_func == -quicksum(living_tv_other_obj), name='constr_obj_func_l_tv_other')
        model.setObjectiveN(living_tv_other_obj_func, priority=10, weight=2, index=obj_func_num) # 10 2
        obj_func_num += 1


    # The frozen area should be as close as possible to the original coordinates.
    for i in range(area_size):
        if all_area_list[i] != {}:
            if not all_area_list[i]['is_optimized']:
                old_x_func, old_y_func = all_area_list[i]['bdb_x'], all_area_list[i]['bdb_y']
                # Set objective function
                frozen_obj_func = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name=f"frozen_obj_func_{i}")
                model.addConstr(
                    frozen_obj_func == (
                                (x_func[i] - old_x_func) * (x_func[i] - old_x_func) + (y_func[i] - old_y_func) * (
                                    y_func[i] - old_y_func)), name=f"set_obj3_{i}")
                model.setObjectiveN(frozen_obj_func, weight = 3, index=obj_func_num, name=f'frozen_constr_obj_func_{i}')
                obj_func_num += 1
        else:
            continue
            # ipdb.set_trace()

    model.update()
    return model, obj_func_num


def optimize_global_add_wall_constraint_by_area(model, all_area_list, rl, rw):
    area_size = len(all_area_list)
    x_func = [var for var in model.getVars() if "x_func" in var.VarName]
    y_func = [var for var in model.getVars() if "y_func" in var.VarName]
    ori_func = model.addVars(area_size, lb = 0, vtype=gp.GRB.INTEGER, name="ori_func")
    bdb_l_func = model.addVars(area_size, lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="bdb_l_func")
    bdb_w_func = model.addVars(area_size, lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="bdb_w_func")

    for i in range(area_size):
        if all_area_list[i] != {}:
            o_i = all_area_list[i]
            if o_i['is_optimized']:
                ori_i = ori_func[i]
                l_i = bdb_l_func[i]
                w_i = bdb_w_func[i]
                x_i = x_func[i]
                y_i = y_func[i]
                c_3, c_4, c_5, c_6 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_3_[{i}]"), \
                                     model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_4_[{i}]"), \
                                     model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_5_[{i}]"), \
                                     model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_6_[{i}]")
                model.addConstr(ori_i + (1 - c_3) * (0 - ori_i) == 0, name=f"t_0_[{i}]")
                model.addConstr(ori_i + (1 - c_4) * (90 - ori_i) == 90, name=f"t_1_[{i}]")
                model.addConstr(ori_i + (1 - c_5) * (180 - ori_i) == 180, name=f"t_2_[{i}]")
                model.addConstr(ori_i + (1 - c_6) * (270 - ori_i) == 270, name=f"t_3_[{i}]")
                model.addConstr(c_3 + c_4 + c_5 + c_6 == 1, name=f"t_4_[{i}]")

                model.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270],
                                      [o_i['length'], o_i['width'], o_i['length'], o_i['width']],
                                      name=f"bdb_l_func_[{i}]")
                model.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270],
                                      [o_i['width'], o_i['length'], o_i['width'], o_i['length']],
                                      name=f"bdb_w_func_[{i}]")

                model.addConstr(y_i + (1 - c_3) * (w_i / 2 - y_i) == w_i / 2, name=f"w_0_[{i}]")
                model.addConstr(x_i + (1 - c_4) * (l_i / 2 - x_i) == l_i / 2, name=f"w_1_[{i}]")
                model.addConstr(y_i + (1 - c_5) * (rw - w_i / 2 - y_i) == rw - w_i / 2, name=f"w_2_[{i}]")
                model.addConstr(x_i + (1 - c_6) * (rl - l_i / 2 - x_i) == rl - l_i / 2, name=f"w_3_[{i}]")
            else:
                ori_i = ori_func[i]
                l_i = bdb_l_func[i]
                w_i = bdb_w_func[i]
                x_i = x_func[i]
                y_i = y_func[i]
                old_orientation = o_i['bdb_ori']
                model.addConstr(ori_i - old_orientation == 0, name=f"t_[{i}]")
                model.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270],
                                      [o_i['length'], o_i['width'], o_i['length'], o_i['width']],
                                      name=f"bdb_l_func_[{i}]")
                model.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270],
                                      [o_i['width'], o_i['length'], o_i['width'], o_i['length']],
                                      name=f"bdb_w_func_[{i}]")
                if old_orientation == 0:
                    model.addConstr(y_i - w_i / 2 == 0, name=f"w_0_[{i}]")
                elif old_orientation == 90:
                    model.addConstr(x_i - l_i / 2 == 0, name=f"w_1_[{i}]")
                elif old_orientation == 180:
                    model.addConstr(y_i - (rw - w_i / 2) == 0, name=f"w_2_[{i}]")
                elif old_orientation == 270:
                    model.addConstr(x_i - (rl - l_i / 2) == 0, name=f"w_3_[{i}]")
                else:
                    ipdb.set_trace()
        else:
            continue
            # ipdb.set_trace()

    model.update()
    return model


def optimize_global_add_wall_constraint_by_area_for_livingroom(model, all_area_list, rl, rw):
    area_size = len(all_area_list)
    x_func = [var for var in model.getVars() if "x_func" in var.VarName]
    y_func = [var for var in model.getVars() if "y_func" in var.VarName]
    ori_func = model.addVars(area_size, lb = 0, vtype=gp.GRB.INTEGER, name="ori_func")
    bdb_l_func = model.addVars(area_size, lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="bdb_l_func")
    bdb_w_func = model.addVars(area_size, lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="bdb_w_func")

    living_areas = []
    tv_areas = []
    for i in range(area_size):
        if all_area_list[i] != {}:
            o_i = all_area_list[i]
            if 'living_area' in o_i['label']:
                living_areas.append([o_i,i])
            if 'tv_area' in o_i['label']:
                tv_areas.append([o_i, i])
    # ipdb.set_trace()
    if len(living_areas)>1 or len(tv_areas)>1:
        print('Too many living areas or tv areas')
        ipdb.set_trace()
    if len(living_areas) == 1 and len(tv_areas) == 1:
        l_o_i, l_i = living_areas[0][0], living_areas[0][1]
        t_o_i, t_i = tv_areas[0][0], tv_areas[0][1]
        if l_o_i['is_optimized'] and t_o_i['is_optimized']:
            # living_area
            l_ori_i = ori_func[l_i]
            # tv_area
            t_ori_i = ori_func[t_i]
            if rl > rw:
                model.addConstr(l_ori_i == 0, name=f"l_ori_t_0_[{l_i}]")
                model.addConstr(t_ori_i == 180, name=f"t_ori_t_0_[{t_i}]")
            else:
                model.addConstr(l_ori_i == 90, name=f"l_ori_t_0_[{l_i}]")
                model.addConstr(t_ori_i == 270, name=f"t_ori_t_0_[{t_i}]")
        elif l_o_i['is_optimized'] and not t_o_i['is_optimized']:
            # living_area
            l_ori_i = ori_func[l_i]
            # tv_area
            t_ori_i = t_o_i['bdb_ori']
            if t_ori_i > 180:
                model.addConstr(l_ori_i == t_ori_i - 180, name=f"l_ori_t_0_[{l_i}]")
            else:
                model.addConstr(l_ori_i == t_ori_i + 180, name=f"l_ori_t_0_[{t_i}]")
        else:
            # living_area
            l_ori_i = l_o_i['bdb_ori']
            # tv_area
            t_ori_i = ori_func[t_i]
            if l_ori_i > 180:
                model.addConstr(t_ori_i == l_ori_i - 180, name=f"t_ori_t_0_[{l_i}]")
            else:
                model.addConstr(t_ori_i == l_ori_i + 180, name=f"t_ori_t_0_[{t_i}]")

    for i in range(area_size):
        if all_area_list[i] != {}:
            o_i = all_area_list[i]
            if o_i['is_optimized']:
                ori_i = ori_func[i]
                l_i = bdb_l_func[i]
                w_i = bdb_w_func[i]
                x_i = x_func[i]
                y_i = y_func[i]
                c_3, c_4, c_5, c_6 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_3_[{i}]"), \
                                     model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_4_[{i}]"), \
                                     model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_5_[{i}]"), \
                                     model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_6_[{i}]")
                model.addConstr(ori_i + (1 - c_3) * (0 - ori_i) == 0, name=f"t_0_[{i}]")
                model.addConstr(ori_i + (1 - c_4) * (90 - ori_i) == 90, name=f"t_1_[{i}]")
                model.addConstr(ori_i + (1 - c_5) * (180 - ori_i) == 180, name=f"t_2_[{i}]")
                model.addConstr(ori_i + (1 - c_6) * (270 - ori_i) == 270, name=f"t_3_[{i}]")
                model.addConstr(c_3 + c_4 + c_5 + c_6 == 1, name=f"t_4_[{i}]")

                model.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270],
                                      [o_i['length'], o_i['width'], o_i['length'], o_i['width']],
                                      name=f"bdb_l_func_[{i}]")
                model.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270],
                                      [o_i['width'], o_i['length'], o_i['width'], o_i['length']],
                                      name=f"bdb_w_func_[{i}]")

                model.addConstr(y_i + (1 - c_3) * (w_i / 2 - y_i) == w_i / 2, name=f"w_0_[{i}]")
                model.addConstr(x_i + (1 - c_4) * (l_i / 2 - x_i) == l_i / 2, name=f"w_1_[{i}]")
                model.addConstr(y_i + (1 - c_5) * (rw - w_i / 2 - y_i) == rw - w_i / 2, name=f"w_2_[{i}]")
                model.addConstr(x_i + (1 - c_6) * (rl - l_i / 2 - x_i) == rl - l_i / 2, name=f"w_3_[{i}]")
            else:
                ori_i = ori_func[i]
                l_i = bdb_l_func[i]
                w_i = bdb_w_func[i]
                x_i = x_func[i]
                y_i = y_func[i]
                old_orientation = o_i['bdb_ori']
                model.addConstr(ori_i - old_orientation == 0, name=f"t_[{i}]")
                model.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270],
                                      [o_i['length'], o_i['width'], o_i['length'], o_i['width']],
                                      name=f"bdb_l_func_[{i}]")
                model.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270],
                                      [o_i['width'], o_i['length'], o_i['width'], o_i['length']],
                                      name=f"bdb_w_func_[{i}]")
                if old_orientation == 0:
                    model.addConstr(y_i - w_i / 2 == 0, name=f"w_0_[{i}]")
                elif old_orientation == 90:
                    model.addConstr(x_i - l_i / 2 == 0, name=f"w_1_[{i}]")
                elif old_orientation == 180:
                    model.addConstr(y_i - (rw - w_i / 2) == 0, name=f"w_2_[{i}]")
                elif old_orientation == 270:
                    model.addConstr(x_i - (rl - l_i / 2) == 0, name=f"w_3_[{i}]")
                else:
                    ipdb.set_trace()
        else:
            continue
            # ipdb.set_trace()



    model.update()
    return model


def optimize_global_add_out_of_bound_constraint_by_area(model, all_area_list, rl, rw):
    area_size = len(all_area_list)
    for i in range(area_size):
        if all_area_list[i] != {}:
            # o_i = all_area_list[i]
            # if o_i['is_optimized']:
            l_i = [var for var in model.getVars() if "bdb_l_func" in var.VarName][i]
            w_i = [var for var in model.getVars() if "bdb_w_func" in var.VarName][i]
            x_i = [var for var in model.getVars() if "x_func" in var.VarName][i]
            y_i = [var for var in model.getVars() if "y_func" in var.VarName][i]
            model.addConstr(x_i >= l_i / 2, name=f"o_f_0_[{i}]")
            model.addConstr(x_i <= rl - l_i / 2, name=f"o_f_1_[{i}]")
            model.addConstr(y_i >= w_i / 2, name=f"o_f_2_[{i}]")
            model.addConstr(y_i <= rw - w_i / 2, name=f"o_f_3_[{i}]")
    model.update()
    return model


def optimize_global_add_collision_constraint_by_area(model, all_area_list, rl, rw, obj_func_num):

    area_size = len(all_area_list)
    n = 0
    area_obj = []
    for i in range(area_size - 1):
        if all_area_list[i] != {}:
            x_i = [var for var in model.getVars() if "x_func" in var.VarName][i]
            y_i = [var for var in model.getVars() if "y_func" in var.VarName][i]
            ori_i = [var for var in model.getVars() if "ori_func" in var.VarName][i]

            for obj_i in all_area_list[i]['obj_info']:
                i_obj_l = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_l_[{n}]")
                i_obj_w = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_w_[{n}]")
                model.addGenConstrPWL(ori_i, i_obj_l, [0, 90, 180, 270],[obj_i['bdb_l'], obj_i['bdb_w'], obj_i['bdb_l'], obj_i['bdb_w']],
                                      name=f"ori_i_obj_l_[{n}]")
                model.addGenConstrPWL(ori_i, i_obj_w, [0, 90, 180, 270],[obj_i['bdb_w'], obj_i['bdb_l'], obj_i['bdb_w'], obj_i['bdb_l']],
                                      name=f"ori_i_obj_w_[{n}]")
                i_obj_x, i_obj_y, i_obj_z = copy.deepcopy(obj_i['bdb_x']), copy.deepcopy(obj_i['bdb_y']), copy.deepcopy(obj_i['bdb_z'])
                i_area_center_x, i_area_center_y = all_area_list[i]['area_coordinate_left'], all_area_list[i]['area_coordinate_top']
                # Translate the coordinate origin within the region to the center of the region
                i_area_coordinate = np.array([i_obj_x, i_obj_y, i_obj_z]) - [i_area_center_x, i_area_center_y, 0]

                rad_ori_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"rad_ori_i_[{n}]")
                cos_rad_ori_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"cos_rad_ori_i_[{n}]")
                sin_rad_ori_i = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"sin_rad_ori_i_[{n}]")
                model.addConstr(rad_ori_i == -ori_i / 180 * np.pi, name=f"coll_rad_ori_i_[{n}]")
                model.addGenConstrCos(rad_ori_i, cos_rad_ori_i, name=f"coll_cos_rad_ori_i_[{n}]")
                model.addGenConstrSin(rad_ori_i, sin_rad_ori_i, name=f"coll_sin_rad_ori_i_[{n}]")

                # Rotate and translate the area into the room
                i_obj_world_x = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_x_[{n}]")
                i_obj_world_y = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_y_[{n}]")
                i_obj_world_z = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_z_[{n}]")
                model.addConstr(i_obj_world_x == i_area_coordinate[0] * cos_rad_ori_i - i_area_coordinate[1] * sin_rad_ori_i + x_i,
                                name=f"coll_i_obj_world_x_[{n}]")
                model.addConstr(i_obj_world_y == i_area_coordinate[0] * sin_rad_ori_i + i_area_coordinate[1] * cos_rad_ori_i + y_i,
                                name=f"coll_i_obj_world_y_[{n}]")
                model.addConstr(i_obj_world_z == i_area_coordinate[2], name=f"coll_i_obj_world_z_[{n}]")

                for j in range(i + 1, area_size):
                    if all_area_list[j] != {}:
                        x_j = [var for var in model.getVars() if "x_func" in var.VarName][j]
                        y_j = [var for var in model.getVars() if "y_func" in var.VarName][j]
                        ori_j = [var for var in model.getVars() if "ori_func" in var.VarName][j]

                        for obj_j in all_area_list[j]['obj_info']:
                            j_obj_l = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"j_obj_l_[{n}]")
                            j_obj_w = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"j_obj_w_[{n}]")
                            model.addGenConstrPWL(ori_j, j_obj_l, [0, 90, 180, 270],[obj_j['bdb_l'], obj_j['bdb_w'], obj_j['bdb_l'], obj_j['bdb_w']],
                                                  name=f"ori_j_obj_l_[{n}]")
                            model.addGenConstrPWL(ori_j, j_obj_w, [0, 90, 180, 270],[obj_j['bdb_w'], obj_j['bdb_l'], obj_j['bdb_w'], obj_j['bdb_l']],
                                                  name=f"ori_j_obj_w_[{n}]")
                            j_obj_x, j_obj_y, j_obj_z = copy.deepcopy(obj_j['bdb_x']), copy.deepcopy(
                                obj_j['bdb_y']), copy.deepcopy(obj_j['bdb_z'])
                            j_area_center_x, j_area_center_y = all_area_list[j]['area_coordinate_left'], \
                                                               all_area_list[j]['area_coordinate_top']
                            # Translate the coordinate origin within the region to the center of the region
                            j_area_coordinate = np.array([j_obj_x, j_obj_y, j_obj_z]) - [j_area_center_x, j_area_center_y, 0]

                            rad_ori_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"rad_ori_j_[{n}]")
                            cos_rad_ori_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                                         name=f"cos_rad_ori_j_[{n}]")
                            sin_rad_ori_j = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                                         name=f"sin_rad_ori_j_[{n}]")
                            model.addConstr(rad_ori_j == -ori_j / 180 * np.pi, name=f"coll_rad_ori_j_[{n}]")
                            model.addGenConstrCos(rad_ori_j, cos_rad_ori_j, name=f"coll_cos_rad_ori_j_[{n}]")
                            model.addGenConstrSin(rad_ori_j, sin_rad_ori_j, name=f"coll_sin_rad_ori_j_[{n}]")

                            # Rotate and translate the area into the room
                            j_obj_world_x = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                                         name=f"j_obj_world_x_[{n}]")
                            j_obj_world_y = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                                         name=f"j_obj_world_y_[{n}]")
                            j_obj_world_z = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                                         name=f"j_obj_world_z_[{n}]")
                            model.addConstr(j_obj_world_x == j_area_coordinate[0] * cos_rad_ori_j - j_area_coordinate[1] * sin_rad_ori_j + x_j,
                                            name=f"coll_j_obj_world_x_[{n}]")
                            model.addConstr(j_obj_world_y == j_area_coordinate[0] * sin_rad_ori_j + j_area_coordinate[1] * cos_rad_ori_j + y_j,
                                            name=f"coll_j_obj_world_y_[{n}]")
                            model.addConstr(j_obj_world_z == j_area_coordinate[2], name=f"coll_j_obj_world_z_[{n}]")

                            x_i_x_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                                   name=f"x_i-x_p_ground_[{n}]")
                            y_i_y_p = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                                                   name=f"y_i-y_p_ground_[{n}]")
                            abs_x_i_x_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_ground_[{n}]")
                            abs_y_i_y_p = model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_ground_[{n}]")
                            c_1, c_2 = model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_ground_[{n}]"), \
                                       model.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_2_ground_[{n}]")

                            model.addConstr(x_i_x_p == i_obj_world_x - j_obj_world_x, name=f"coll_x_d_ground_[{n}]")
                            model.addConstr(y_i_y_p == i_obj_world_y - j_obj_world_y, name=f"coll_y_d_ground_[{n}]")
                            model.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_ground_[{n}]")
                            model.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_ground_[{n}]")
                            model.addConstr(((j_obj_l + i_obj_l) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1),
                                            name=f"coll_x_ground_[{n}]")
                            model.addConstr(((j_obj_w + i_obj_w) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2),
                                            name=f"coll_y_ground_[{n}]")
                            model.addConstr(c_1 + c_2 >= 1, name=f"coll_c_ground_[{n}]")
                            area_obj.append((i_obj_world_x - j_obj_world_x) * (i_obj_world_x - j_obj_world_x) + (i_obj_world_y - j_obj_world_y) * (
                                    i_obj_world_y - j_obj_world_y))

                            n += 1
                    else:
                        continue
                        ipdb.set_trace()
        else:
            continue
            ipdb.set_trace()

    area_obj_func = model.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name="area_obj_func")
    model.addConstr(area_obj_func == -quicksum(area_obj), name='constr_area_obj_func')
    model.setObjectiveN(area_obj_func, index=obj_func_num, weight = 2, priority=5) # 新增weight = 2, priority=5
    obj_func_num += 1

    model.update()
    return model


def optimize_global_by_area(rl, rw, all_area_list, room_type = 'bedroom'):
    # if len(all_area_list) >= 8:
    #     ipdb.set_trace()
    #     return None, None
    # Create a new model
    #with gp.Env() as env, gp.Model(env=env) as m:
        m = gp.Model()
        obj_func_num = 0 

        # Set objective function for Area
        if room_type == 'livingroom':
            m, obj_func_num = optimize_global_set_area_function_for_livingroom(m, all_area_list, rl, rw, obj_func_num)
        else:
            m, obj_func_num = optimize_global_set_area_function(m, all_area_list, rl, rw, obj_func_num)

        # Add wall constraint
        if room_type == 'livingroom':
            m = optimize_global_add_wall_constraint_by_area_for_livingroom(m, all_area_list, rl, rw)
        else:
            m = optimize_global_add_wall_constraint_by_area(m, all_area_list, rl, rw)

        # Add Out of bound constraint
        m = optimize_global_add_out_of_bound_constraint_by_area(m, all_area_list, rl, rw)

        # Add collision constraint
        m = optimize_global_add_collision_constraint_by_area(m, all_area_list, rl, rw, obj_func_num)

        # Solve it!
        m.Params.TimeLimit = 300  # 5 minutes
        m.optimize()
        # m.computeIIS()
        # m.write("delete_model1.ilp")
        if m.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            print("__________________________________________________________________________")
            all_P = []
            for i in range(len(all_area_list)):
                if all_area_list[i] != {}:
                    area_i = all_area_list[i]
                    l_i, w_i = [var for var in m.getVars() if "bdb_l_func" in var.VarName][i], \
                               [var for var in m.getVars() if "bdb_w_func" in var.VarName][i]
                    ori_i = [var for var in m.getVars() if "ori_func" in var.VarName][i]
                    x_i = [var for var in m.getVars() if "x_func" in var.VarName][i]
                    y_i = [var for var in m.getVars() if "y_func" in var.VarName][i]
                    area_i['bdb_x'], area_i['bdb_y'], area_i['bdb_z'] = x_i.x, y_i.x, 0.0
                    area_i['bdb_l'], area_i['bdb_w'], area_i['bdb_h'] = l_i.x, w_i.x, 0.0
                    area_i['bdb_ori'] = ori_i.x
                    area_i['height'] = 0.0
                    all_P.append(area_i)
                else:
                    all_P.append({})
            if type == 'modify':
                ipdb.set_trace()
            # object_list = parse_gpt_output_step_by_step(all_P)
            return all_P, True

        elif m.status == GRB.INFEASIBLE:
            print("Model Status: The model is not feasible")
            return [], False
        elif m.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            return [], False
        else:
            print("Model status: Optimal solution not found")
            # m.computeIIS()
            # m.write("delete_model1.ilp")
            # ipdb.set_trace()
            all_P = []
            try:
                for i in range(len(all_area_list)):
                    if all_area_list[i] != {}:
                        area_i = all_area_list[i]
                        # if area_i['is_optimized']:
                        l_i, w_i = [var for var in m.getVars() if "bdb_l_func" in var.VarName][i], \
                                   [var for var in m.getVars() if "bdb_w_func" in var.VarName][i]
                        ori_i = [var for var in m.getVars() if "ori_func" in var.VarName][i]
                        x_i = [var for var in m.getVars() if "x_func" in var.VarName][i]
                        y_i = [var for var in m.getVars() if "y_func" in var.VarName][i]
                        area_i['bdb_x'], area_i['bdb_y'], area_i['bdb_z'] = x_i.x, y_i.x, 0.0
                        area_i['bdb_l'], area_i['bdb_w'], area_i['bdb_h'] = l_i.x, w_i.x, 0.0
                        area_i['bdb_ori'] = ori_i.x
                        area_i['height'] = 0.0
                        all_P.append(area_i)
                    else:
                        all_P.append({})
            except:
                # ipdb.set_trace()
                return all_P, False
            # object_list = parse_gpt_output_step_by_step(all_P)
            return all_P, False


def optimize_local_area_by_area(area, P, rl, rw):
    # with gp.Env() as env, gp.Model(env=env) as m:
        # Create a new model
        m = gp.Model()
        # Create variables
        x_i = m.addVar(lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="x")
        y_i = m.addVar(lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="y")
        ori_i = m.addVar(vtype=gp.GRB.INTEGER, name="ori_func")
        l_i = m.addVar(lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="bdb_l_func")
        w_i = m.addVar(lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="bdb_w_func")
        # Set objective function
        obj = sum(
            (x_i - area_p['bdb_x']) * (x_i - area_p['bdb_x']) + (y_i - area_p['bdb_y']) * (y_i - area_p['bdb_y']) for
            area_p in P)
        m.setObjective(obj, gp.GRB.MAXIMIZE)
        # Add constraints
        c_3, c_4, c_5, c_6 = m.addVar(lb=0, vtype=gp.GRB.BINARY, name="c_3"), m.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                       name="c_4"), \
                             m.addVar(lb=0, vtype=gp.GRB.BINARY, name="c_5"), m.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                       name="c_6")
        # Orientation
        m.addConstr(ori_i + (1 - c_3) * (0 - ori_i) == 0, name="t_0")
        m.addConstr(ori_i + (1 - c_4) * (90 - ori_i) == 90, name="t_1")
        m.addConstr(ori_i + (1 - c_5) * (180 - ori_i) == 180, name="t_2")
        m.addConstr(ori_i + (1 - c_6) * (270 - ori_i) == 270, name="t_3")
        m.addConstr(c_3 + c_4 + c_5 + c_6 == 1, name="t_4")

        m.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270], [area['length'], area['width'], area['length'], area['width']],
                          name="bdb_l_func")
        m.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270], [area['width'], area['length'], area['width'], area['length']],
                          name="bdb_w_func")

        # Wall
        m.addConstr(y_i + (1 - c_3) * (w_i / 2 - y_i) == w_i / 2, name="w_0")
        m.addConstr(x_i + (1 - c_4) * (l_i / 2 - x_i) == l_i / 2, name="w_1")
        m.addConstr(y_i + (1 - c_5) * (rw - w_i / 2 - y_i) == rw - w_i / 2, name="w_2")
        m.addConstr(x_i + (1 - c_6) * (rl - l_i / 2 - x_i) == rl - l_i / 2, name="w_3")
        # m.addConstr(c_3 + c_4 + c_5 + c_6 >= 1)
        # Collision
        i = 0
        for o_p in P:
            area_ori = o_p['bdb_ori']
            R = roty_3d(-area_ori / 180. * np.pi)
            area_center_x, area_center_y = o_p['area_coordinate_left'], o_p['area_coordinate_top']
            area_x, area_y = o_p['bdb_x'], o_p['bdb_y']
            for o_p_obj in o_p['obj_info']:
                if area_ori in [0, 180]:
                    obj_l, obj_w = o_p_obj['bdb_l'], o_p_obj['bdb_w']
                else:
                    obj_l, obj_w = o_p_obj['bdb_w'], o_p_obj['bdb_l']
                obj_x, obj_y, obj_z = copy.deepcopy(o_p_obj['bdb_x']), copy.deepcopy(o_p_obj['bdb_y']), copy.deepcopy(
                    o_p_obj['bdb_z'])
                # Translate the coordinate origin within the region to the center of the region
                area_coordinate = np.array([obj_x, obj_y, obj_z]) - [area_center_x, area_center_y, 0]
                # Rotate and translate the area into the room
                world_coordinate = area_coordinate @ R + [area_x, area_y, 0]
                obj_world_x, obj_world_y, obj_world_z = world_coordinate[0], world_coordinate[1], world_coordinate[2]

                for i_obj in area['obj_info']:
                    i_obj_l = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_l_[{i}]")
                    i_obj_w = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_w_[{i}]")
                    m.addGenConstrPWL(ori_i, i_obj_l, [0, 90, 180, 270],
                                      [i_obj['bdb_l'], i_obj['bdb_w'], i_obj['bdb_l'], i_obj['bdb_w']],
                                      name=f"ori_i_obj_l_[{i}]")
                    m.addGenConstrPWL(ori_i, i_obj_w, [0, 90, 180, 270],
                                      [i_obj['bdb_w'], i_obj['bdb_l'], i_obj['bdb_w'], i_obj['bdb_l']],
                                      name=f"ori_i_obj_w_[{i}]")
                    i_obj_x, i_obj_y, i_obj_z = copy.deepcopy(i_obj['bdb_x']), copy.deepcopy(
                        i_obj['bdb_y']), copy.deepcopy(i_obj['bdb_z'])
                    i_area_center_x, i_area_center_y = area['area_coordinate_left'], area['area_coordinate_top']
                    # Translate the coordinate origin within the region to the center of the region
                    i_area_coordinate = np.array([i_obj_x, i_obj_y, i_obj_z]) - [i_area_center_x, i_area_center_y, 0]

                    rad_ori = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"rad_ori_[{i}]")
                    cos_rad_ori = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"cos_rad_ori_[{i}]")
                    sin_rad_ori = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"sin_rad_ori_[{i}]")
                    m.addConstr(rad_ori == -ori_i / 180 * np.pi, name=f"coll_rad_ori_[{i}]")
                    m.addGenConstrCos(rad_ori, cos_rad_ori, name=f"coll_cos_rad_ori_[{i}]")
                    m.addGenConstrSin(rad_ori, sin_rad_ori, name=f"coll_sin_rad_ori_[{i}]")

                    # Rotate and translate the area into the room
                    i_obj_world_x = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_x_[{i}]")
                    i_obj_world_y = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_y_[{i}]")
                    i_obj_world_z = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_z_[{i}]")
                    m.addConstr(
                        i_obj_world_x == i_area_coordinate[0] * cos_rad_ori - i_area_coordinate[1] * sin_rad_ori + x_i,
                        name=f"coll_i_obj_world_x_[{i}]")
                    m.addConstr(
                        i_obj_world_y == i_area_coordinate[0] * sin_rad_ori + i_area_coordinate[1] * cos_rad_ori + y_i,
                        name=f"coll_i_obj_world_y_[{i}]")
                    m.addConstr(i_obj_world_z == i_area_coordinate[2], name=f"coll_i_obj_world_z_[{i}]")

                    x_i_x_p = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_[{i}]")
                    y_i_y_p = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
                    abs_x_i_x_p = m.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_[{i}]")
                    abs_y_i_y_p = m.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_[{i}]")
                    c_1, c_2 = m.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), \
                               m.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_2_[{i}]")

                    m.addConstr(x_i_x_p == i_obj_world_x - obj_world_x, name=f"coll_x_d_[{i}]")
                    m.addConstr(y_i_y_p == i_obj_world_y - obj_world_y, name=f"coll_y_d_[{i}]")
                    m.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_[{i}]")
                    m.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_[{i}]")
                    m.addConstr(((obj_l + i_obj_l) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
                    m.addConstr(((obj_w + i_obj_w) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
                    m.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
                    i += 1
        # Out of bound
        m.addConstr(x_i >= l_i / 2, name='o_0')
        m.addConstr(x_i <= rl - l_i / 2, name='o_1')
        m.addConstr(y_i >= w_i / 2, name='o_2')
        m.addConstr(y_i <= rw - w_i / 2, name='o_3')
        # Solve it!
        m.Params.TimeLimit = 600  # 10 minutes
        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            return x_i.x, y_i.x, l_i.x, w_i.x, ori_i.x
        elif m.status == GRB.INFEASIBLE:
            print("Model Status: The model is not feasible")
            return None
        elif m.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            return None
        else:
            print("Model status: Optimal solution not found")
            return None


def optimize_local_area_by_area_for_livingroom(area, partner_area, P, rl, rw):
    # with gp.Env() as env, gp.Model(env=env) as m:
        # Create a new model
        m = gp.Model()
        # Create variables
        x_i = m.addVar(lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="x")
        y_i = m.addVar(lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="y")
        ori_i = m.addVar(vtype=gp.GRB.INTEGER, name="ori_func")
        l_i = m.addVar(lb=0.0, ub=rl, vtype=gp.GRB.CONTINUOUS, name="bdb_l_func")
        w_i = m.addVar(lb=0.0, ub=rw, vtype=gp.GRB.CONTINUOUS, name="bdb_w_func")
        # Set objective function
        # obj = sum((x_i - area_p['bdb_x']) * (x_i - area_p['bdb_x']) + (y_i - area_p['bdb_y']) * (y_i - area_p['bdb_y']) for area_p in P if area_p['label'] != partner_area['label'])
        # m.setObjective(obj, gp.GRB.MAXIMIZE)
        l_t_obj = (x_i - partner_area['bdb_x']) * (x_i - partner_area['bdb_x']) + (y_i - partner_area['bdb_y']) * (
                    y_i - partner_area['bdb_y'])
        m.setObjective(l_t_obj, gp.GRB.MINIMIZE)
        # Add constraints
        c_3, c_4, c_5, c_6 = m.addVar(lb=0, vtype=gp.GRB.BINARY, name="c_3"), m.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                       name="c_4"), \
                             m.addVar(lb=0, vtype=gp.GRB.BINARY, name="c_5"), m.addVar(lb=0, vtype=gp.GRB.BINARY,
                                                                                       name="c_6")
        # Orientation
        if partner_area['bdb_ori'] >= 180:
            m.addConstr(ori_i == partner_area['bdb_ori'] - 180, name="l_t_0")
        else:
            m.addConstr(ori_i == partner_area['bdb_ori'] + 180, name="l_t_0")

        m.addConstr(ori_i + (1 - c_3) * (0 - ori_i) == 0, name="t_0")
        m.addConstr(ori_i + (1 - c_4) * (90 - ori_i) == 90, name="t_1")
        m.addConstr(ori_i + (1 - c_5) * (180 - ori_i) == 180, name="t_2")
        m.addConstr(ori_i + (1 - c_6) * (270 - ori_i) == 270, name="t_3")
        m.addConstr(c_3 + c_4 + c_5 + c_6 == 1, name="t_4")

        m.addGenConstrPWL(ori_i, l_i, [0, 90, 180, 270], [area['length'], area['width'], area['length'], area['width']],
                          name="bdb_l_func")
        m.addGenConstrPWL(ori_i, w_i, [0, 90, 180, 270], [area['width'], area['length'], area['width'], area['length']],
                          name="bdb_w_func")

        # Wall
        m.addConstr(y_i + (1 - c_3) * (w_i / 2 - y_i) == w_i / 2, name="w_0")
        m.addConstr(x_i + (1 - c_4) * (l_i / 2 - x_i) == l_i / 2, name="w_1")
        m.addConstr(y_i + (1 - c_5) * (rw - w_i / 2 - y_i) == rw - w_i / 2, name="w_2")
        m.addConstr(x_i + (1 - c_6) * (rl - l_i / 2 - x_i) == rl - l_i / 2, name="w_3")
        # m.addConstr(c_3 + c_4 + c_5 + c_6 >= 1)
        # Collision
        i = 0
        for o_p in P:
            area_ori = o_p['bdb_ori']
            R = roty_3d(-area_ori / 180. * np.pi)
            area_center_x, area_center_y = o_p['area_coordinate_left'], o_p['area_coordinate_top']
            area_x, area_y = o_p['bdb_x'], o_p['bdb_y']
            for o_p_obj in o_p['obj_info']:
                if area_ori in [0, 180]:
                    obj_l, obj_w = o_p_obj['bdb_l'], o_p_obj['bdb_w']
                else:
                    obj_l, obj_w = o_p_obj['bdb_w'], o_p_obj['bdb_l']
                obj_x, obj_y, obj_z = copy.deepcopy(o_p_obj['bdb_x']), copy.deepcopy(o_p_obj['bdb_y']), copy.deepcopy(
                    o_p_obj['bdb_z'])
                # Translate the coordinate origin within the region to the center of the region
                area_coordinate = np.array([obj_x, obj_y, obj_z]) - [area_center_x, area_center_y, 0]
                # Rotate and translate the area into the room
                world_coordinate = area_coordinate @ R + [area_x, area_y, 0]
                obj_world_x, obj_world_y, obj_world_z = world_coordinate[0], world_coordinate[1], world_coordinate[2]

                for i_obj in area['obj_info']:
                    i_obj_l = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_l_[{i}]")
                    i_obj_w = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_w_[{i}]")
                    m.addGenConstrPWL(ori_i, i_obj_l, [0, 90, 180, 270],
                                      [i_obj['bdb_l'], i_obj['bdb_w'], i_obj['bdb_l'], i_obj['bdb_w']],
                                      name=f"ori_i_obj_l_[{i}]")
                    m.addGenConstrPWL(ori_i, i_obj_w, [0, 90, 180, 270],
                                      [i_obj['bdb_w'], i_obj['bdb_l'], i_obj['bdb_w'], i_obj['bdb_l']],
                                      name=f"ori_i_obj_w_[{i}]")
                    i_obj_x, i_obj_y, i_obj_z = copy.deepcopy(i_obj['bdb_x']), copy.deepcopy(
                        i_obj['bdb_y']), copy.deepcopy(
                        i_obj['bdb_z'])
                    i_area_center_x, i_area_center_y = area['area_coordinate_left'], area['area_coordinate_top']
                    # Translate the coordinate origin within the region to the center of the region
                    i_area_coordinate = np.array([i_obj_x, i_obj_y, i_obj_z]) - [i_area_center_x, i_area_center_y, 0]

                    rad_ori = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"rad_ori_[{i}]")
                    cos_rad_ori = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"cos_rad_ori_[{i}]")
                    sin_rad_ori = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"sin_rad_ori_[{i}]")
                    m.addConstr(rad_ori == -ori_i / 180 * np.pi, name=f"coll_rad_ori_[{i}]")
                    m.addGenConstrCos(rad_ori, cos_rad_ori, name=f"coll_cos_rad_ori_[{i}]")
                    m.addGenConstrSin(rad_ori, sin_rad_ori, name=f"coll_sin_rad_ori_[{i}]")

                    # Rotate and translate the area into the room
                    i_obj_world_x = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_x_[{i}]")
                    i_obj_world_y = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_y_[{i}]")
                    i_obj_world_z = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"i_obj_world_z_[{i}]")
                    m.addConstr(
                        i_obj_world_x == i_area_coordinate[0] * cos_rad_ori - i_area_coordinate[1] * sin_rad_ori + x_i,
                        name=f"coll_i_obj_world_x_[{i}]")
                    m.addConstr(
                        i_obj_world_y == i_area_coordinate[0] * sin_rad_ori + i_area_coordinate[1] * cos_rad_ori + y_i,
                        name=f"coll_i_obj_world_y_[{i}]")
                    m.addConstr(i_obj_world_z == i_area_coordinate[2], name=f"coll_i_obj_world_z_[{i}]")

                    x_i_x_p = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"x_i-x_p_[{i}]")
                    y_i_y_p = m.addVar(lb=-GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name=f"y_i-y_p[{i}]")
                    abs_x_i_x_p = m.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_x_i-x_p_[{i}]")
                    abs_y_i_y_p = m.addVar(vtype=gp.GRB.CONTINUOUS, name=f"abs_y_i-y_p_[{i}]")
                    c_1, c_2 = m.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_1_[{i}]"), \
                               m.addVar(lb=0, vtype=gp.GRB.BINARY, name=f"c_2_[{i}]")

                    m.addConstr(x_i_x_p == i_obj_world_x - obj_world_x, name=f"coll_x_d_[{i}]")
                    m.addConstr(y_i_y_p == i_obj_world_y - obj_world_y, name=f"coll_y_d_[{i}]")
                    m.addConstr(abs_x_i_x_p == abs_(x_i_x_p), name=f"coll_abs_x_d_[{i}]")
                    m.addConstr(abs_y_i_y_p == abs_(y_i_y_p), name=f"coll_abs_y_d_[{i}]")
                    m.addConstr(((obj_l + i_obj_l) / 2 - abs_x_i_x_p) <= rl * rl * (1 - c_1), name=f"coll_x_[{i}]")
                    m.addConstr(((obj_w + i_obj_w) / 2 - abs_y_i_y_p) <= rw * rw * (1 - c_2), name=f"coll_y_[{i}]")
                    m.addConstr(c_1 + c_2 >= 1, name=f"coll_c_[{i}]")
                    i += 1

        # Out of bound
        m.addConstr(x_i >= l_i / 2, name='o_0')
        m.addConstr(x_i <= rl - l_i / 2, name='o_1')
        m.addConstr(y_i >= w_i / 2, name='o_2')
        m.addConstr(y_i <= rw - w_i / 2, name='o_3')
        # Solve it!
        m.Params.TimeLimit = 600  # 10 minutes
        m.optimize()
        if m.status == GRB.OPTIMAL:
            print("Model status: Optimal solution found")
            return x_i.x, y_i.x, l_i.x, w_i.x, ori_i.x
        elif m.status == GRB.INFEASIBLE:
            m.computeIIS()
            m.write("delete_model1.ilp")
            print("Model Status: The model is not feasible")
            return None
        elif m.status == GRB.UNBOUNDED:
            print("Model Status: The model is unbounded")
            return None
        else:
            print("Model status: Optimal solution not found")
            return None


def optimize_local_by_area(rl, rw, data, room_type = 'bedroom'):
    all_area = []
    rl += 0.01
    rw += 0.01

    for area in data:
        if len(area) != 0:
            if not area['is_optimized']:
                all_area.append(area)
                ipdb.set_trace()
    for area in data:
        if len(area) != 0 :
            if len(all_area) == 0:
                if area['is_optimized']:
                    ori_list = [0, 90, 180, 270]
                    time = 0
                    while 1:
                        time += 1
                        ori_i = random.choice(ori_list)
                        area['bdb_ori'] = ori_i
                        l_i, w_i = get_2dbdbox_l_w(area)
                        if l_i <= rl and w_i <= rw:
                            break
                        if time > 20:
                            print(ori_i)
                            ipdb.set_trace()
                            break
                    if time > 20:
                        ipdb.set_trace()
                        all_area.append({})
                        continue
                    while 1:
                        if ori_i == 0:
                            y_i = w_i / 2
                            x_i = round(random.uniform(l_i / 2, rl - l_i / 2), 2)
                        elif ori_i == 90:
                            x_i = l_i / 2
                            y_i = round(random.uniform(w_i / 2, rw - w_i / 2), 2)
                        elif ori_i == 180:
                            y_i = rw - w_i / 2
                            x_i = round(random.uniform(l_i / 2, rl - l_i / 2), 2)
                        elif ori_i == 270:
                            x_i = rl - l_i / 2
                            y_i = round(random.uniform(w_i / 2, rw - w_i / 2), 2)
                        if x_i == l_i / 2 or x_i == rl - l_i / 2 or y_i == w_i / 2 or y_i == rw - w_i / 2:
                            break
                    area['bdb_l'], area['bdb_w'], area['bdb_h'] = l_i, w_i, 0.0
                    area['bdb_x'], area['bdb_y'], area['bdb_z'] = x_i, y_i, 0.0
                    all_area.append(area)
            else:
                if area['is_optimized']:
                    try:
                        if room_type == 'livingroom':
                            if 'living_area' in area['label']:
                                tv_area = None
                                for p_area in all_area:
                                    if 'tv_area' in p_area['label']:
                                        tv_area = p_area
                                if tv_area is not None:
                                    x_i, y_i, l_i, w_i, ori_i = optimize_local_area_by_area_for_livingroom(area, tv_area, all_area, rl,rw)
                                else:
                                    x_i, y_i, l_i, w_i, ori_i = optimize_local_area_by_area(area, all_area, rl, rw)
                            elif 'tv_area' in area['label']:
                                living_area = None
                                for p_area in all_area:
                                    if 'living_area' in p_area['label']:
                                        living_area = p_area
                                if living_area is not None:
                                    x_i, y_i, l_i, w_i, ori_i = optimize_local_area_by_area_for_livingroom(area, living_area, all_area, rl, rw)
                                else:
                                    x_i, y_i, l_i, w_i, ori_i = optimize_local_area_by_area(area, all_area, rl, rw)
                            else:
                                x_i, y_i, l_i, w_i, ori_i = optimize_local_area_by_area(area, all_area, rl, rw)
                        else:
                            x_i, y_i, l_i, w_i, ori_i = optimize_local_area_by_area(area, all_area, rl, rw)
                        area['bdb_x'], area['bdb_y'], area['bdb_z'] = x_i, y_i, 0.0
                        area['bdb_l'], area['bdb_w'], area['bdb_h'] = l_i, w_i, 0.0
                        area['bdb_ori'] = ori_i
                        all_area.append(area)
                    except:
                        all_area.append({})

        else:
            ipdb.set_trace()
            all_area.append({})

    return all_area, None


def optimize_galo_by_area(val_id, prompt, data, rel_pos_constraints = None, type = 'one-time', interactive_num = 0, room_type = 'bedroom'):
    rl, rw = float(data['length']), float(data['width'])
    all_area_object = []
    all_status = []
    for i in range(len(data['children'])):
        area = data['children'][i]
        if 'children' in area.keys():
            area_l = float(area['length'])  # with_area_size
            area_w = float(area['width'])  # with_area_size
            if rel_pos_constraints is not None:
                rel_pos_constraint = rel_pos_constraints[i]
            else:
                rel_pos_constraint = None
            object_local, status = optimize_local_obj_by_area(area_l, area_w, area, rel_pos_constraint, room_type)
            all_area_object.append(object_local)
            all_status.append(status)
        else:
            all_area_object.append([])
            all_status.append([])

    
    all_area_list = []
    for i in range(len(data['children'])):
        area = data['children'][i]
        if 'children' in area.keys() and len(area['children']) != 0:
            area_l, area_w, area_left, area_top = get_area_length_width(all_area_object[i])
            area_data = {
                'label': area['label'].split(' ')[0],
                'length': area_l,
                'width': area_w,
                'old_length_width': [area['length'], area['width']],
                'area_coordinate_left': area_left,
                'area_coordinate_top': area_top,
                'obj_info': all_area_object[i]
            }
            if 'optimization' in area.keys():
                if len(area['optimization']) != 0 and area['optimization'][0] == 'area_global':
                    area_data['is_optimized'] = True
                else:
                    area_data['height'] = 0.0
                    area_data['bdb_x'] = area['bdb_x']
                    area_data['bdb_y'] = area['bdb_y']
                    area_data['bdb_z'] = area['bdb_z']
                    area_data['bdb_l'] = area['bdb_l']
                    area_data['bdb_w'] = area['bdb_w']
                    area_data['bdb_h'] = area['bdb_h']
                    # area_data['orientation'] = area['orientation']
                    area_data['bdb_ori'] = area['orientation']
                    area_data['is_optimized'] = False
            all_area_list.append(area_data)
        else:
            ipdb.set_trace()
            all_area_list.append({})

    area_global, _ = optimize_global_by_area(rl, rw, all_area_list, room_type)


    if len(area_global) == 0:
        print("Area global optimization failed. Perform local optimization !")
        ipdb.set_trace()
        area_global, _ = optimize_local_by_area(rl, rw, all_area_list, room_type)


    if len(all_area_object) == len(area_global) and len(area_global) == len(data['children']):
        print("Finish Optimized!")

    # Based on the results of regional optimization, convert the object's coordinates to world coordinate system coordinates.
    all_object_list = [] # Save the object's coordinates in the world coordinate system
    all_area_object_list = [] # Save the object's coordinates in the corresponding regional coordinate system
    for i in range(len(all_area_object)):
        area_object_list = []
        area_inform = area_global[i]
        if area_inform != {}:
            area_ori = area_inform['bdb_ori']
            R = roty_3d(-area_ori / 180. * np.pi)
            area_center_x = area_inform['area_coordinate_left']
            area_center_y = area_inform['area_coordinate_top']
            area_x = area_inform['bdb_x']
            area_y = area_inform['bdb_y']
            area_inform['left'] = area_x
            area_inform['top'] = area_y
            area_inform['orientation'] = area_ori

            for obj in all_area_object[i]:
                area_obj_ori = obj['bdb_ori']

                area_object_list.append([
                    obj['label'],
                    {
                        "length": obj['length'],
                        "width": obj['width'],
                        "height": obj['height'],
                        "left": obj['bdb_x'],
                        "top": obj['bdb_y'],
                        "depth": obj['bdb_z'],
                        "orientation": area_obj_ori,
                        "is_use_rel_lib": obj['is_use_rel_lib'],
                        "description": obj['description']
                    }
                ])
                # Translate the coordinate origin within the region to the center of the region
                area_coordinate = np.array([obj['bdb_x'], obj['bdb_y'], obj['bdb_z']]) - [area_center_x, area_center_y, 0]
                # Rotate and translate the area into the room
                world_coordinate = area_coordinate @ R + [area_x, area_y, 0]
                world_obj_ori = copy.deepcopy(area_ori)+area_obj_ori
                if world_obj_ori >= 360:
                    world_obj_ori -= 360
                world_obj = [obj['label'],
                             {
                                 "length": obj['length'],
                                 "width": obj['width'],
                                 "height": obj['height'],
                                 "left": world_coordinate[0],
                                 "top": world_coordinate[1],
                                 "depth": world_coordinate[2],
                                 "orientation": world_obj_ori,
                                 "description": obj['description']
                             }]
                obj['left'], obj['top'], obj['depth'], obj['orientation'] = world_coordinate[0], world_coordinate[1], \
                                                                            world_coordinate[2], world_obj_ori

                all_object_list.append(world_obj)
            all_area_object_list.append(area_object_list)
        else:
            all_area_object_list.append(area_object_list)
            all_object_list.append([])


    for i in range(len(data['children'])):
        area = data['children'][i]
        if area_global[i] != {}:
            area['bdb_x'], area['bdb_y'], area['bdb_z'] = area_global[i]['bdb_x'], area_global[i]['bdb_y'], \
                                                          area_global[i]['bdb_z']
            area['bdb_l'], area['bdb_w'], area['bdb_h'] = area_global[i]['bdb_l'], area_global[i]['bdb_w'], \
                                                          area_global[i]['bdb_h']
            area['bdb_ori'] = area_global[i]['bdb_ori']
            area['orientation'] = area_global[i]['orientation']
            area['tight_length'] = area_global[i]['length']
            area['tight_width'] = area_global[i]['width']

            func_obj = area['children']
            func_obj['bdb_x'], func_obj['bdb_y'], func_obj['bdb_z'] = all_area_object[i][0]['bdb_x'], \
                                                                      all_area_object[i][0]['bdb_y'], \
                                                                      all_area_object[i][0]['bdb_z']
            func_obj['bdb_l'], func_obj['bdb_w'], func_obj['bdb_h'] = all_area_object[i][0]['bdb_l'], \
                                                                      all_area_object[i][0]['bdb_w'], \
                                                                      all_area_object[i][0]['bdb_h']
            func_obj['left'], func_obj['top'], func_obj['depth'] = all_area_object[i][0]['left'], \
                                                                   all_area_object[i][0]['top'], \
                                                                   all_area_object[i][0]['depth']
            func_obj['orientation'] = all_area_object[i][0]['orientation']
            func_obj['bdb_ori'] = all_area_object[i][0]['bdb_ori']
            if 'children' in func_obj.keys():
                for j in range(len(func_obj['children'])):
                    flag = 0
                    for k in range(len(all_area_object[i])):
                        obj = func_obj['children'][j]
                        new_obj = all_area_object[i][k]
                        if obj['label'] == new_obj['label']:
                            obj = func_obj['children'][j]
                            obj['bdb_x'], obj['bdb_y'], obj['bdb_z'] = all_area_object[i][k]['bdb_x'], \
                                                                       all_area_object[i][k]['bdb_y'], \
                                                                       all_area_object[i][k]['bdb_z']
                            obj['bdb_l'], obj['bdb_w'], obj['bdb_h'] = all_area_object[i][k]['bdb_l'], \
                                                                       all_area_object[i][k]['bdb_w'], \
                                                                       all_area_object[i][k]['bdb_h']
                            obj['left'], obj['top'], obj['depth'] = all_area_object[i][k]['left'], \
                                                                    all_area_object[i][k]['top'], \
                                                                    all_area_object[i][k]['depth']
                            obj['orientation'] = all_area_object[i][k]['orientation']
                            obj['bdb_ori'] = all_area_object[i][k]['bdb_ori']
                            flag = 1
                            break
                    if flag == 0:
                        print('Object has been deleted')
                        obj['is_delete'] = True
                        ipdb.set_trace()
        else:
            print('Area has been deleted')
            area['is_delete'] = True

    return data, all_object_list, all_area_object_list, all_status


def optimize_lola_by_area(val_id, prompt, data, rel_pos_constraints=None, type='one-time', interactive_num=0):
    rl, rw = float(data['length']), float(data['width'])
    all_area_object = []
    all_status = []
    for i in range(len(data['children'])):
        area = data['children'][i]
        if 'children' in area.keys():
            area_l = float(area['length'])  # with_area_size
            area_w = float(area['width'])  # with_area_size
            if rel_pos_constraints is not None:
                rel_pos_constraint = rel_pos_constraints[i]
            else:
                rel_pos_constraint = None
            # if type != 'one-time':
            #     visualize_area_obj_topview(area['label'], area_l, area_w, [], name = f'area_{i}', type = type, interactive_num = interactive_num)
            object_local, status = optimize_local_obj_by_area(area_l, area_w, area, rel_pos_constraint)
            # if type != 'one-time':
            #     visualize_area_obj_topview(area['label'], area_l, area_w, object_local, name = f'area_{i}_add_obj', type = type, interactive_num = interactive_num)
            all_area_object.append(object_local)
            all_status.append(status)
        else:
            all_area_object.append([])
            all_status.append([])

    all_area_list = []
    for i in range(len(data['children'])):
        area = data['children'][i]
        if 'children' in area.keys() and len(all_area_object[i]) != 0:
            func_obj = all_area_object[i][0]
            old_ori = func_obj['bdb_ori']
            area_l, area_w, area_left, area_top = get_area_length_width(all_area_object[i])  # without_area_size
            area_data = {  # without_area_size
                'label': area['label'].split(' ')[0],
                'length': area_l,
                'width': area_w,
                'bdb_ori': old_ori
            }
            l, w = get_2dbdbox_l_w(area_data)  # without_area_size
            start_area_data = {  # without_area_size
                'label': area['label'],
                'length': l,
                'width': w,
                'old_length_width': [area['length'], area['width']],
                'area_coordinate_left': area_left,
                'area_coordinate_top': area_top,
                'start_orientation': old_ori
            }
           
            if 'optimization' in area.keys():
                if len(area['optimization']) != 0 and area['optimization'][0] == 'area_global':
                    start_area_data['is_optimized'] = True
                else:
                    start_area_data['height'] = 0.0
                    start_area_data['bdb_x'] = area['bdb_x']
                    start_area_data['bdb_y'] = area['bdb_y']
                    start_area_data['bdb_z'] = area['bdb_z']
                    start_area_data['bdb_l'] = area['bdb_l']
                    start_area_data['bdb_w'] = area['bdb_w']
                    start_area_data['bdb_h'] = area['bdb_h']
                    # area_data['orientation'] = area['orientation']
                    start_area_data['bdb_ori'] = area['orientation']
                    start_area_data['is_optimized'] = False
            all_area_list.append(start_area_data)
        else:
            ipdb.set_trace()
            all_area_list.append({})
    area_loc_list, _ = optimize_local_by_area(rl, rw, all_area_list)
    if len(area_loc_list) != len(data['children']):
        ipdb.set_trace()

    # ipdb.set_trace()
    all_object_list = []
    all_area_object_list = []
    for i in range(len(area_loc_list)):
        area = area_loc_list[i]
        area_object_list = []
        if len(area) != 0:
            area_center_x = area['area_coordinate_left']
            area_center_y = area['area_coordinate_top']
            # area_center_x = area['old_length_width'][0]/2 # with_area_size
            # area_center_y = area['old_length_width'][1]/2 # with_area_size
            area_x = area['bdb_x']
            area_y = area['bdb_y']
            area['orientation'] = area['bdb_ori']
            R = area['bdb_ori'] - area['start_orientation']
            R = roty_3d(-R / 180. * np.pi)
            for obj in all_area_object[i]:
                area_obj_ori = area['start_orientation']
                if ('chair' in obj['label'] or 'stool' in obj['label']) and 'rel_position' in obj.keys():
                    f_x, f_y = all_area_object[i][0]['bdb_x'], all_area_object[i][0]['bdb_y']
                    o_x, o_y = obj['bdb_x'], obj['bdb_y']
                    angle = 3 * math.pi / 2 - math.atan2((o_y - f_y), (o_x - f_x))
                    degree = math.degrees(angle)
                    diff = [abs(degree - 0), abs(degree - 90), abs(degree - 180), abs(degree - 270), abs(degree - 360),
                            abs(degree - 450)]
                    index = diff.index(min(diff))
                    if index == 0 or index == 4:
                        area_obj_ori = 0
                    elif index == 1 or index == 5:
                        area_obj_ori = 90
                    elif index == 2:
                        area_obj_ori = 180
                    elif index == 3:
                        area_obj_ori = 270

                area_object_list.append([
                    obj['label'],
                    {
                        "length": obj['length'],
                        "width": obj['width'],
                        "height": obj['height'],
                        "left": obj['bdb_x'],
                        "top": obj['bdb_y'],
                        "depth": obj['bdb_z'],
                        "orientation": area_obj_ori,
                        "description": obj['description']
                    }
                ])
                # Translate the coordinate origin within the region to the center of the region
                area_coordinate = np.array([obj['bdb_x'], obj['bdb_y'], obj['bdb_z']]) - [area_center_x,
                                                                                                area_center_y, 0]
                # Rotate and translate the area into the room
                world_coordinate = area_coordinate @ R + [area_x, area_y, 0]
                world_obj_ori = copy.deepcopy(area['bdb_ori'])
                if ('chair' in obj['label'] or 'stool' in obj['label']) and 'rel_position' in obj.keys():
                    f_x, f_y = all_area_object[i][0]['left'], all_area_object[i][0]['top']
                    o_x, o_y = world_coordinate[0], world_coordinate[1]
                    angle = 3 * math.pi / 2 - math.atan2((o_y - f_y), (o_x - f_x))
                    degree = math.degrees(angle)
                    diff = [abs(degree - 0), abs(degree - 90), abs(degree - 180), abs(degree - 270), abs(degree - 360),
                            abs(degree - 450)]
                    index = diff.index(min(diff))
                    if index == 0 or index == 4:
                        world_obj_ori = 0
                    elif index == 1 or index == 5:
                        world_obj_ori = 90
                    elif index == 2:
                        world_obj_ori = 180
                    elif index == 3:
                        world_obj_ori = 270
                world_obj = [obj['label'],
                             {
                                 "length": obj['length'],
                                 "width": obj['width'],
                                 "height": obj['height'],
                                 "left": world_coordinate[0],
                                 "top": world_coordinate[1],
                                 "depth": world_coordinate[2],
                                 "orientation": world_obj_ori,
                                 "description": obj['description']
                             }]
                obj['left'], obj['top'], obj['depth'], obj['orientation'] = world_coordinate[0], world_coordinate[1], \
                                                                            world_coordinate[2], world_obj_ori

                all_object_list.append(world_obj)
            all_area_object_list.append(area_object_list)
        else:
            all_object_list.append([])
            all_area_object_list.append(area_object_list)
    # Updated hierarchy data(all_area_object, area_global, hierarchy_data)
    for i in range(len(data['children'])):
        area = data['children'][i]
        if area_loc_list[i] != {}:
            area['bdb_x'], area['bdb_y'], area['bdb_z'] = area_loc_list[i]['bdb_x'], area_loc_list[i]['bdb_y'], \
                                                          area_loc_list[i]['bdb_z']
            area['bdb_l'], area['bdb_w'], area['bdb_h'] = area_loc_list[i]['bdb_l'], area_loc_list[i]['bdb_w'], \
                                                          area_loc_list[i]['bdb_h']
            area['bdb_ori'] = area_loc_list[i]['bdb_ori']
            area['orientation'] = area_loc_list[i]['orientation']
            area['tight_length'] = area_loc_list[i]['length']
            area['tight_width'] = area_loc_list[i]['width']

            func_obj = area['children']
            func_obj['bdb_x'], func_obj['bdb_y'], func_obj['bdb_z'] = all_area_object[i][0]['bdb_x'], \
                                                                      all_area_object[i][0]['bdb_y'], \
                                                                      all_area_object[i][0]['bdb_z']
            func_obj['bdb_l'], func_obj['bdb_w'], func_obj['bdb_h'] = all_area_object[i][0]['bdb_l'], \
                                                                      all_area_object[i][0]['bdb_w'], \
                                                                      all_area_object[i][0]['bdb_h']
            func_obj['left'], func_obj['top'], func_obj['depth'] = all_area_object[i][0]['left'], \
                                                                   all_area_object[i][0]['top'], \
                                                                   all_area_object[i][0]['depth']
            func_obj['orientation'] = all_area_object[i][0]['orientation']
            func_obj['bdb_ori'] = all_area_object[i][0]['bdb_ori']
            if 'children' in func_obj.keys():
                for j in range(len(func_obj['children'])):
                    flag = 0
                    for k in range(len(all_area_object[i])):
                        obj = func_obj['children'][j]
                        new_obj = all_area_object[i][k]
                        if obj['label'] == new_obj['label']:
                            obj = func_obj['children'][j]
                            obj['bdb_x'], obj['bdb_y'], obj['bdb_z'] = all_area_object[i][k]['bdb_x'], \
                                                                       all_area_object[i][k]['bdb_y'], \
                                                                       all_area_object[i][k]['bdb_z']
                            obj['bdb_l'], obj['bdb_w'], obj['bdb_h'] = all_area_object[i][k]['bdb_l'], \
                                                                       all_area_object[i][k]['bdb_w'], \
                                                                       all_area_object[i][k]['bdb_h']
                            obj['left'], obj['top'], obj['depth'] = all_area_object[i][k]['left'], \
                                                                    all_area_object[i][k]['top'], \
                                                                    all_area_object[i][k]['depth']
                            obj['orientation'] = all_area_object[i][k]['orientation']
                            obj['bdb_ori'] = all_area_object[i][k]['bdb_ori']
                            flag = 1
                            break
                    if flag == 0:
                        print('Object has been deleted')
                        obj['is_delete'] = True
                        ipdb.set_trace()
        else:
            print('Area has been deleted')
            area['is_delete'] = True
            ipdb.set_trace()

    return data, all_object_list, all_area_object_list, all_status
