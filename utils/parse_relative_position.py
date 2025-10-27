import copy
import difflib
import json

import ipdb
import math

import numpy as np

from utils.utils import get_3d_bdbox, roty_3d, roty, area_to_graph, OBJ_LIST
import torch.nn.functional as F
from utils.get_new_hierarchical_scene_tree import draw_id_scene
from utils.get_lib import draw_bdb2d_point, draw_area_graph

OLD_ALL_RELATIONS = ["right of", "right touching", "left of", "left touching", "behind", "behind touching", "in front of",
                 "front touching", "in the right front of", "right front touching", "in the left front of",
                 "left front touching", "in the right behind of", "right behind touching", "in the left behind of",
                 "left behind touching", "above", "on"]

ALL_RELATIONS = ["right of", "left of", "behind", "in front of", "right front of", "left front of", "right behind",
                 "left behind","above", "on top of", "below"]

def trans_sys(obj, R = None):
    # 1. trans to world sys
    new_obj = {}
    box3d_world_vertices = get_3d_bdbox(obj)
    box3d_world = [[min(box3d_world_vertices[:, 0]), min(box3d_world_vertices[:, 1]), min(box3d_world_vertices[:, 2])],
            [max(box3d_world_vertices[:, 0]), max(box3d_world_vertices[:, 1]), max(box3d_world_vertices[:, 2])]]
    new_obj['label'] = obj['label']
    new_obj['length'] = float(abs(box3d_world[0][0] - box3d_world[1][0]))
    new_obj['width'] = float(abs(box3d_world[0][1] - box3d_world[1][1]))
    new_obj['height'] = float(abs(box3d_world[0][2] - box3d_world[1][2]))
    if 'left' in obj.keys():
        new_obj['left'] = obj['left']
        new_obj['top'] = obj['top']
        new_obj['depth'] = obj['depth']
    new_obj['orientation'] = obj['orientation']
    if 'relative position' in obj.keys():
        new_obj['relative position'] = obj['relative position']
    # 2. trans to R_sys
    if R:
        new_obj['orientation'] = R
        new_R_obj = {}
        box3d_R_vertices = get_3d_bdbox(new_obj)
        box3d_R = [
            [min(box3d_R_vertices[:, 0]), min(box3d_R_vertices[:, 1]), min(box3d_R_vertices[:, 2])],
            [max(box3d_R_vertices[:, 0]), max(box3d_R_vertices[:, 1]), max(box3d_R_vertices[:, 2])]]
        new_R_obj['label'] = obj['label']
        new_R_obj['length'] = float(abs(box3d_R[0][0] - box3d_R[1][0]))
        new_R_obj['width'] = float(abs(box3d_R[0][1] - box3d_R[1][1]))
        new_R_obj['height'] = float(abs(box3d_R[0][2] - box3d_R[1][2]))
        if 'left' in obj.keys():
            new_R_obj['left'] = obj['left']
            new_R_obj['top'] = obj['top']
            new_R_obj['depth'] = obj['depth']
        new_R_obj['orientation'] = R
        if 'relative position' in obj.keys():
            new_R_obj['relative position'] = obj['relative position']
        return new_R_obj
    else:
        return new_obj


def bias_ab(a, b):
    # return min(a,b)/4
    return 0


def get_other_obj_position_from_func_obj_old(other_obj, func_obj):
    position = {}
    try:
        obj_name, rel, func_obj_name = other_obj['relative position']
    except:
        obj_name, rel, func_obj_name = other_obj['relative_position']
    if other_obj['orientation'] != func_obj['orientation']:
        other_obj = trans_sys(other_obj, func_obj['orientation'])
    else:
        other_obj = other_obj

    if rel == None:
        rel = "above"
        print(obj_name, rel, func_obj_name)
    else:
        rel = difflib.get_close_matches(rel, ALL_RELATIONS, cutoff=0.0)[0]

    if rel == 'surrounding' or rel == 'inside':
        position['left'] = func_obj['left']
        position['top'] = func_obj['top']
        position['depth'] = func_obj['depth']
        return position
    elif rel == 'right of' or rel == 'right touching':
        if 'touching' in rel:
            dis = (other_obj['length'] + func_obj['length']) / 2
        else:
            dis = (other_obj['length'] + func_obj['length'])/2 + bias_ab(other_obj['length'],  func_obj['length'])
        dis_x, dis_y = dis*np.cos(func_obj['orientation']/ 180 * math.pi), dis*np.sin(func_obj['orientation']/ 180 * math.pi)
        pos = np.asarray([[func_obj['left']+dis_x, func_obj['top']-dis_y, other_obj['height']/2]])
    elif rel == 'left of' or rel == 'left touching':
        if 'touching' in rel:
            dis = (other_obj['length'] + func_obj['length']) / 2
        else:
            dis = (other_obj['length'] + func_obj['length'])/2 + bias_ab(other_obj['length'],  func_obj['length'])
        dis_x, dis_y = dis * np.cos(func_obj['orientation'] / 180 * math.pi), dis * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left']-dis_x, func_obj['top']+dis_y, other_obj['height']/2]])
    elif rel == 'behind' or rel == 'behind touching':
        if 'touching' in rel:
            dis = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_y, dis_x = dis * np.cos(func_obj['orientation'] / 180 * math.pi), dis * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left']-dis_x, func_obj['top']-dis_y, other_obj['height']/2]])
    elif rel == 'in front of' or rel == 'front touching':
        if 'touching' in rel:
            dis = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_y, dis_x = dis * np.cos(func_obj['orientation'] / 180 * math.pi), dis * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left']+dis_x, func_obj['top']+dis_y, other_obj['height']/2]])
    elif rel in 'in the right front of' or rel in 'right front touching':
        if 'touching' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            dis_y = (-other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'], func_obj['length'])
            dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x*np.cos(func_obj['orientation']/ 180 * math.pi), dis_x*np.sin(func_obj['orientation']/ 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] + dis_x_x + dis_y_x, func_obj['top'] - dis_x_y + dis_y_y, other_obj['height'] / 2]])
    elif rel in 'in the left front of' or rel in 'left front touching':
        if 'touching' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            dis_y = (-other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'],  func_obj['length'])
            dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x * np.cos(func_obj['orientation'] / 180 * math.pi), dis_x * np.sin(func_obj['orientation'] / 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] - dis_x_x + dis_y_x, func_obj['top'] + dis_x_y + dis_y_y, other_obj['height'] / 2]])
    elif rel in 'in the right behind of' or rel in 'right behind touching':
        if 'touching' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            dis_y = (-other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'],  func_obj['length'])
            dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x * np.cos(func_obj['orientation'] / 180 * math.pi), dis_x * np.sin(
            func_obj['orientation'] / 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(
            func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] + dis_x_x - dis_y_x, func_obj['top'] - dis_x_y - dis_y_y, other_obj['height'] / 2]])
        # ipdb.set_trace()
    elif rel in 'in the left behind of' or rel in 'left behind touching':
        if 'touching' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            dis_y = (-other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'],  func_obj['length'])
            dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x * np.cos(func_obj['orientation'] / 180 * math.pi), dis_x * np.sin(func_obj['orientation'] / 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] - dis_x_x - dis_y_x, func_obj['top'] + dis_x_y - dis_y_y, other_obj['height'] / 2]])
        # ipdb.set_trace()
    elif rel == 'above' or rel == 'on':
        if 'on' in rel:
            dis = (other_obj['height'] + func_obj['height']) / 2
        else:
            dis = (other_obj['height'] + func_obj['height']) / 2 + bias_ab(other_obj['height'],  func_obj['height'])
        pos = np.asarray([[func_obj['left'], func_obj['top'], func_obj['depth']+dis]])
        position['left'] = pos[0][0]
        position['top'] = pos[0][1]
        position['depth'] = pos[0][2]
        return position
    else:
        ipdb.set_trace()

    new_pos = pos
    #     # position['left'] = max(0,new_pos[0][0])
    #     # position['top'] = max(0,new_pos[0][1])
    #     # position['depth'] = max(0,new_pos[0][2])
    position['left'] = new_pos[0][0]
    position['top'] = new_pos[0][1]
    position['depth'] = new_pos[0][2]
    return position


def get_other_obj_position_from_func_obj(other_obj, func_obj):
    position = {}
    try:
        obj_name, rel, func_obj_name = other_obj['relative position']
    except:
        obj_name, rel, func_obj_name = other_obj['relative_position']
    if other_obj['orientation'] != func_obj['orientation']:
        other_obj = trans_sys(other_obj, func_obj['orientation'])
    else:
        other_obj = other_obj
    if rel == None:
        rel = "above"
        rel_no_near = "above"
        print(obj_name, rel, func_obj_name)
    else:
        rel_no_near = rel.split('(near)')[0]
        rel_no_near = difflib.get_close_matches(rel_no_near, ALL_RELATIONS, cutoff=0.0)[0]

    if rel_no_near == 'surrounding' or rel_no_near == 'inside':
        position['left'] = func_obj['left']
        position['top'] = func_obj['top']
        position['depth'] = func_obj['depth']
        return position
    elif rel_no_near == 'left of':
        if '(near)' in rel:
            dis = (other_obj['length'] + func_obj['length']) / 2
        else:
            dis = (other_obj['length'] + func_obj['length'])/2 + bias_ab(other_obj['length'],  func_obj['length'])
        dis_x, dis_y = dis*np.cos(func_obj['orientation']/ 180 * math.pi), dis*np.sin(func_obj['orientation']/ 180 * math.pi)
        pos = np.asarray([[func_obj['left']+dis_x, func_obj['top']-dis_y, other_obj['height']/2]])
    elif rel_no_near == 'right of':
        if '(near)' in rel:
            dis = (other_obj['length'] + func_obj['length']) / 2
        else:
            dis = (other_obj['length'] + func_obj['length'])/2 + bias_ab(other_obj['length'],  func_obj['length'])
        dis_x, dis_y = dis * np.cos(func_obj['orientation'] / 180 * math.pi), dis * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left']-dis_x, func_obj['top']+dis_y, other_obj['height']/2]])
    elif rel_no_near == 'behind':
        if '(near)' in rel:
            dis = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_y, dis_x = dis * np.cos(func_obj['orientation'] / 180 * math.pi), dis * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left']-dis_x, func_obj['top']-dis_y, other_obj['height']/2]])
    elif rel_no_near == 'in front of':
        if '(near)' in rel:
            dis = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_y, dis_x = dis * np.cos(func_obj['orientation'] / 180 * math.pi), dis * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left']+dis_x, func_obj['top']+dis_y, other_obj['height']/2]])
    elif rel_no_near in 'left front of':
        if '(near)' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2
            dis_y = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'], func_obj['length'])
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
            dis_y = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x*np.cos(func_obj['orientation']/ 180 * math.pi), dis_x*np.sin(func_obj['orientation']/ 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] + dis_x_x + dis_y_x, func_obj['top'] - dis_x_y + dis_y_y, other_obj['height'] / 2]])
    elif rel_no_near in 'right front of':
        if '(near)' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2
            dis_y = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'],  func_obj['length'])
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
            dis_y = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x * np.cos(func_obj['orientation'] / 180 * math.pi), dis_x * np.sin(func_obj['orientation'] / 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] - dis_x_x + dis_y_x, func_obj['top'] + dis_x_y + dis_y_y, other_obj['height'] / 2]])
    elif rel_no_near in 'left behind':
        if '(near)' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2
            dis_y = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'],  func_obj['length'])
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
            dis_y = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x * np.cos(func_obj['orientation'] / 180 * math.pi), dis_x * np.sin(
            func_obj['orientation'] / 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(
            func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] + dis_x_x - dis_y_x, func_obj['top'] - dis_x_y - dis_y_y, other_obj['height'] / 2]])
        # ipdb.set_trace()
    elif rel_no_near in 'right behind':
        if '(near)' in rel:
            dis_x = (other_obj['length'] + func_obj['length']) / 2
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2
            dis_y = (other_obj['width'] + func_obj['width']) / 2
        else:
            dis_x = (other_obj['length'] + func_obj['length']) / 2 + bias_ab(other_obj['length'],  func_obj['length'])
            # dis_y = (-other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
            dis_y = (other_obj['width'] + func_obj['width']) / 2 + bias_ab(other_obj['width'], func_obj['width'])
        dis_x_x, dis_x_y = dis_x * np.cos(func_obj['orientation'] / 180 * math.pi), dis_x * np.sin(func_obj['orientation'] / 180 * math.pi)
        dis_y_y, dis_y_x = dis_y * np.cos(func_obj['orientation'] / 180 * math.pi), dis_y * np.sin(func_obj['orientation'] / 180 * math.pi)
        pos = np.asarray([[func_obj['left'] - dis_x_x - dis_y_x, func_obj['top'] + dis_x_y - dis_y_y, other_obj['height'] / 2]])
        # ipdb.set_trace()
    elif rel_no_near == 'above' or rel_no_near == 'on top of':
        if 'on top of' in rel_no_near:
            dis = (other_obj['height'] + func_obj['height']) / 2
        else:
            dis = (other_obj['height'] + func_obj['height']) / 2 + bias_ab(other_obj['height'],  func_obj['height'])
        pos = np.asarray([[func_obj['left'], func_obj['top'], func_obj['depth']+dis]])
        position['left'] = pos[0][0]
        position['top'] = pos[0][1]
        position['depth'] = pos[0][2]
        return position
    elif rel_no_near == 'below':
        pos = np.asarray([[func_obj['left'], func_obj['top'], 0]])
    else:
        ipdb.set_trace()

    new_pos = pos
    #     # position['left'] = max(0,new_pos[0][0])
    #     # position['top'] = max(0,new_pos[0][1])
    #     # position['depth'] = max(0,new_pos[0][2])
    position['left'] = new_pos[0][0]
    position['top'] = new_pos[0][1]
    position['depth'] = new_pos[0][2]
    return position


def parse_relative_position(data, is_update_structure = False):
    for area in data['children']:
        if is_update_structure:
            for func_obj in area['children']:
                if 'children' in func_obj.keys():
                    for other_obj in func_obj['children']:
                        posstion = get_other_obj_position_from_func_obj(other_obj, func_obj)
                        other_obj['left'] = posstion['left']
                        other_obj['top'] = posstion['top']
                        other_obj['depth'] = posstion['depth']
        else:
            func_obj = area['children']
            if 'children' in func_obj.keys():
                for other_obj in func_obj['children']:
                    posstion = get_other_obj_position_from_func_obj(other_obj, func_obj)
                    other_obj['left'] = posstion['left']
                    other_obj['top'] = posstion['top']
                    other_obj['depth'] = posstion['depth']

    return data


def get_close_matches(word, possibilities, ratio, n=3, cutoff=0.6):
    '''
    Optional arg n (default 3) is the maximum number of close matches to
    return.  n must be > 0.

    Optional arg cutoff (default 0.6) is a float in [0, 1].  Possibilities
    that don't score at least that similar to word are ignored.
    '''
    if not n >  0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    result = {}
    s = difflib.SequenceMatcher()
    s.set_seq2(word)
    for id in possibilities:
        x = possibilities[id]
        s.set_seq1(x)
        if s.real_quick_ratio() >= cutoff and \
           s.quick_ratio() >= cutoff and \
           s.ratio() >= cutoff:
            result[id] = (s.ratio()*ratio, x)
    return result


def retrieve_similar_rel_cood(pre_rel_pos, lib, is_area_retrieval = False):
    # compute obj1, rel, obj2 score
    if is_area_retrieval:
        pre_area_type, pre_area_length, pre_area_width, pre_rel = pre_rel_pos
        # pre_area = pre_area_type + ": (" + str(pre_area_length) + " ," + str(pre_area_length) + ")"
        pre_area_size = "(" + str(pre_area_length) + " ," + str(pre_area_width) + ")"
        pre_obj1, pre_rel, pre_obj2 = pre_rel

        # poss_area = {id: eval(id)[0] + ": (" + str(eval(id)[1]) + " ," + str(eval(id)[2]) + ")" for id in lib}
        poss_area_type = {id: eval(id)[0] for id in lib}
        poss_area_size = {id: "(" + str(eval(id)[1]) + " ," + str(eval(id)[2]) + ")" for id in lib}
        poss_obj1s = {id: eval(id)[3][0] for id in lib}
        poss_rels = {id: eval(id)[3][1] for id in lib}
        poss_obj2s = {id: eval(id)[3][2] for id in lib}

        score_area_type = get_close_matches(pre_area_type, poss_area_type, ratio=1.0, cutoff=0.0)
        score_area_size = get_close_matches(pre_area_size, poss_area_size, ratio=1.0, cutoff=0.0)
        score_obj1s = get_close_matches(pre_obj1, poss_obj1s, ratio=1.0, cutoff=0.0)  # {id: (score, pre_obj1)}
        score_rels = get_close_matches(pre_rel, poss_rels, ratio=1.0, cutoff=0.0)
        score_obj2s = get_close_matches(pre_obj2, poss_obj2s, ratio=1.0, cutoff=0.0)
    else:
        pre_obj1, pre_rel, pre_obj2 = pre_rel_pos
        poss_obj1s = {id: eval(id)[0] for id in lib}
        poss_rels = {id: eval(id)[1] for id in lib}
        poss_obj2s = {id: eval(id)[2] for id in lib}
        score_obj1s = get_close_matches(pre_obj1, poss_obj1s, ratio=1.0, cutoff=0.0)  # {id: (score, pre_obj1)}
        score_rels = get_close_matches(pre_rel, poss_rels, ratio=1.0, cutoff=0.0)
        score_obj2s = get_close_matches(pre_obj2, poss_obj2s, ratio=1.0, cutoff=0.0)

    # sum score
    id_score = []
    for id in lib:
        score = 0
        if is_area_retrieval:
            score += score_area_type[id][0]
            score += score_area_size[id][0]
            score += score_obj1s[id][0]
            score += score_rels[id][0]
            score += score_obj2s[id][0]
        else:
            score += score_obj1s[id][0]
            score += score_rels[id][0]
            score += score_obj2s[id][0]
        id_score.append((id,score))
    id_score_sorted = sorted(id_score, key=lambda x: x[1], reverse = True)
    most_similar_obj1 = lib[id_score_sorted[0][0]]['obj']
    most_similar_obj2 = lib[id_score_sorted[0][0]]['func_obj']
    most_similar_rel_pos = lib[id_score_sorted[0][0]]['rel_coordinates']
    most_similar_rel_ori = lib[id_score_sorted[0][0]]['rel_ori']

    return id_score_sorted[0][0], most_similar_obj1, most_similar_obj2, most_similar_rel_pos, most_similar_rel_ori


def get_other_obj_position_from_lib(area_type, area_length, area_width, other_obj, func_obj, lib, is_area_retrieval = False):
    # ipdb.set_trace()
    position = {}
    if is_area_retrieval:
        pre_rel_pos = [area_type, area_length, area_width, other_obj['relative position']]
    else:
        pre_rel_pos = other_obj['relative position']
    rel_id, rel_obj, rel_func, rel_coordinates, rel_ori = retrieve_similar_rel_cood(pre_rel_pos, lib, is_area_retrieval)


    # transfer to world system
    world_ori = rel_ori + func_obj['orientation']

    rel_rel_func_2dbdb = np.asarray([[-rel_func['length'] / 2, -rel_func['width'] / 2],
                                     [rel_func['length'] / 2, -rel_func['width'] / 2],
                                     [-rel_func['length'] / 2, rel_func['width'] / 2],
                                     [rel_func['length'] / 2, rel_func['width'] / 2]])
    draw_bdb2d_point(rel_rel_func_2dbdb, rel_coordinates, str(pre_rel_pos))



    rel_func_2dbdb = np.asarray([[-func_obj['length'] / 2, -func_obj['width'] / 2],
                                 [func_obj['length'] / 2, -func_obj['width'] / 2],
                                 [-func_obj['length'] / 2, func_obj['width'] / 2],
                                 [func_obj['length'] / 2, func_obj['width'] / 2]])


    delta_x_1 = float(rel_func['length'] / 2 - rel_coordinates[0])
    delta_x_2 = float(-rel_func['length'] / 2 - rel_coordinates[0])
    delta_y_1 = float(rel_func['width'] / 2 - rel_coordinates[1])
    delta_y_2 = float(-rel_func['width'] / 2 - rel_coordinates[1])
    delta_z_1 = float(rel_func['height'] / 2 - rel_coordinates[2])
    delta_z_2 = float(-rel_func['height'] / 2 - rel_coordinates[2])

    delta_x = float(delta_x_1 / delta_x_2) if delta_x_2 != 0.0 else None
    delta_y = float(delta_y_1 / delta_y_2) if delta_y_2 != 0.0 else None
    delta_z = float(delta_z_1 / delta_z_2) if delta_z_2 != 0.0 else None

    if delta_x is not None:
        rel_coordinates_x = (func_obj['length'] / 2) * (1 + delta_x) / (1 - delta_x) if delta_x != 1.0 else 0.0
    else:
        rel_coordinates_x = -1 * (func_obj['length'] / 2)
    if delta_y is not None:
        rel_coordinates_y = (func_obj['width'] / 2) * (1 + delta_y) / (1 - delta_y) if delta_y != 1.0 else 0.0
    else:
        rel_coordinates_y = -1 * (func_obj['width'] / 2)
    if delta_z is not None:
        rel_coordinates_z = (func_obj['height'] / 2) * (1 + delta_z) / (1 - delta_z) if delta_z != 1.0 else 0.0
    else:
        rel_coordinates_z = -1 * (func_obj['height'] / 2)


    draw_bdb2d_point(rel_func_2dbdb, [rel_coordinates_x, rel_coordinates_y, rel_coordinates_z], str(pre_rel_pos))
    ipdb.set_trace()

    R = roty_3d(-func_obj['orientation'] / 180. * np.pi)
    offset = [func_obj['left'], func_obj['top'], func_obj['depth']]
    world_coordinates = [rel_coordinates_x, rel_coordinates_y, rel_coordinates_z] @ R + [offset[0], offset[1], offset[2]]
    world_func_2dbdb = [[func_obj['3d bounding box'][0], func_obj['3d bounding box'][1]],
                  [func_obj['3d bounding box'][0], func_obj['3d bounding box'][4]],
                  [func_obj['3d bounding box'][3], func_obj['3d bounding box'][1]],
                  [func_obj['3d bounding box'][3], func_obj['3d bounding box'][4]]]
    # draw_bdb2d_point(world_func_2dbdb, world_coordinates, str(pre_rel_pos))
    print(pre_rel_pos, rel_id)
    # if world_coordinates[0] >= 600 or world_coordinates[1] >= 600:
    #  ipdb.set_trace()
    # if 'MasterBedroom-391' in val_id:
    #     ipdb.set_trace()



    position['left'] = world_coordinates[0]
    position['top'] = world_coordinates[1]
    position['depth'] = world_coordinates[2]
    position['ori'] = world_ori

    return position


def parse_relative_position_from_lib(data, lib, is_update_structure = False, is_area_retrieval = False):
    for area in data['children']:
        if is_update_structure:
            for func_obj in area['children']:
                if 'children' in func_obj.keys():
                    for other_obj in func_obj['children']:
                        area_length = area['2d bounding box'][2] - area['2d bounding box'][0]
                        area_width = area['2d bounding box'][3] - area['2d bounding box'][1]
                        posstion = get_other_obj_position_from_lib(area['label'], area_length, area_width, other_obj, func_obj, lib, is_area_retrieval)
                        other_obj['left'] = posstion['left']
                        other_obj['top'] = posstion['top']
                        other_obj['depth'] = posstion['depth']
                        other_obj['orientation'] = posstion['ori']
        else:
            func_obj = area['children']
            if 'children' in func_obj.keys():
                for other_obj in func_obj['children']:
                    area_length = area['2d bounding box'][2]-area['2d bounding box'][0]
                    area_width = area['2d bounding box'][3]-area['2d bounding box'][1]
                    posstion = get_other_obj_position_from_lib(area['label'], area_length, area_width, other_obj, func_obj, lib, is_area_retrieval)
                    other_obj['left'] = posstion['left']
                    other_obj['top'] = posstion['top']
                    other_obj['depth'] = posstion['depth']
                    other_obj['orientation'] = posstion['ori']

    return data


def query_freq(feat_ni):
    #freq_ri = feat_ri['freq']
    #freq_si = feat_si['freq']
    return 1


def get_close_scores(word1, word2, ratio, n=3,cutoff=0.6):
    '''
            Optional arg n (default 3) is the maximum number of close matches to
            return.  n must be > 0.

            Optional arg cutoff (default 0.6) is a float in [0, 1].  Possibilities
            that don't score at least that similar to word are ignored.
            '''
    if not n > 0:
        raise ValueError("n must be > 0: %r" % (n,))
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
    s = difflib.SequenceMatcher()
    s.set_seq2(word1)
    s.set_seq1(word2)
    if s.real_quick_ratio() >= cutoff and \
            s.quick_ratio() >= cutoff and \
            s.ratio() >= cutoff:
        return s.ratio() * ratio
    return 0


def nodekernel(graph_A, graph_B, ri, si, min_g=1, use_vector = False, encoder_model = None, room_type='bedroom'):
    # we don't have model identity and geometry
    feat_ri = graph_A.nodes[list(graph_A.nodes)[ri]] # ******
    feat_si = graph_B.nodes[list(graph_B.nodes)[si]] # ******

    # identity kernel
    # assume every room to be "identical"
    k_identity = 1

    # tag kernel
    label_ri = feat_ri['tag']
    label_si = feat_si['tag']

    k_tag = get_close_scores(label_ri, label_si, 1, cutoff=0) 


    # geometry kernel
    size_ri = feat_ri['geometry']
    size_si = feat_si['geometry']
    dist = np.linalg.norm(np.array(size_ri)-np.array(size_si))
    freq_ri = query_freq(feat_ri)
    freq_si = query_freq(feat_si)
    k_geo = freq_ri*freq_si*np.exp(-1*(2*dist/min_g)**2) 

    # feature kernel
    if use_vector:
        feature_ri = copy.deepcopy(feat_ri['features'])
        feature_si = copy.deepcopy(feat_si['features'])
        similarity_1 = F.cosine_similarity(feature_ri, feature_si, dim=-1)
        k_feature = ((similarity_1+1)/2).cpu().numpy() 
        if room_type == 'bedroom':
            nodesim = 0.3 * k_tag + 0.1 * k_geo + 0.6 * k_feature
        else:
            nodesim = 0.7 * k_tag + 0.1 * k_geo + 0.2 * k_feature
        # if ('dressing_table' in label_ri and 'dressing_chair' in label_si) or ('dressing_table' in label_ri and 'dressing_table' in label_si):
        #     # print('label_ri:', label_ri)
        #     # print('label_si:', label_si)
        #     # print('k_feature:', k_feature)
        #     print('nodesim:', nodesim)

    else:
        # we don't consider the node frequency normalization
        # just assume all the categories are identical
        nodesim = 0.1*k_identity + 0.6*k_tag + 0.3*k_geo


    # ipdb.set_trace()
    if(nodesim<10e-6):
        nodesim = 0
    # print(nodesim)
    return nodesim


def edgekernel(graph_A, graph_B, ei, fi):
    feat_ei = graph_A.edges[(ei[0], ei[1])]
    feat_fi = graph_B.edges[(fi[0], fi[1])]

    # label kernel
    label_ei = feat_ei['tag']
    label_fi = feat_fi['tag']

    if ei[0] == fi[0] and ei[1] == fi[1]: 
        k_tag = 1
    else:
        k_tag = 0

    edgesim = k_tag
    # print(edgesim)
    return edgesim


def graphkernel1(graph_A, graph_B, p, use_vector=False, encoder_model = None, room_type='bedroom'):
    def update_kernel_matrix(graph_A, graph_B, ri, si, p, kernel_matrix):
        node_rs = kernel_matrix[ri, si, 0]
        neighbor_A = graph_A.neighbors(list(graph_A.nodes)[ri]) # ******
        neighbor_B = graph_B.neighbors(list(graph_B.nodes)[si]) # ******
        sim_edge_ef = 0
        num_r_prime = 0
        num_s_prime = 0
        for r_prime in neighbor_A:
            for s_prime in neighbor_B:
                e = (list(graph_A.nodes)[ri], r_prime)
                f = (list(graph_B.nodes)[si], s_prime)
                edge_ef = edgekernel(graph_A, graph_B, e, f)
                sim_edge_ef += edge_ef
                num_s_prime+=1
            num_r_prime += 1
        # ipdb.set_trace()
        k_prev = kernel_matrix[ri, si, p - 1]
        if num_r_prime != 0 and num_s_prime != 0:
            output = k_prev / num_r_prime * sim_edge_ef
        else:
            output = node_rs


        return output

    kernel_matrix = np.zeros((len(graph_A.nodes), len(graph_B.nodes), p + 1))

    # init the kernel values when p=0
    for rid in range(len(graph_A.nodes)):
        for sid in range(len(graph_B.nodes)):
            kernel_matrix[rid, sid, 0] = nodekernel(graph_A, graph_B, rid, sid, use_vector = use_vector, encoder_model= encoder_model, room_type=room_type)
    # ipdb.set_trace()

    for pid in range(1, p + 1):
        for rid in range(len(graph_A.nodes)):
            for sid in range(len(graph_B.nodes)):
                kernel_matrix[rid, sid, pid] = update_kernel_matrix(graph_A, graph_B, rid, sid, pid, kernel_matrix)
    # ipdb.set_trace()
    return np.sum(kernel_matrix[:, :, p]), kernel_matrix[:, :, 0]


def graphkernel_dist_metric(graph_A, graph_B, p, use_vector=False, encoder_model = None, room_type='bedroom'):
    # d(Ga,Gb)=sqrt(KpG(Ga,Ga)-2KpG(Ga,Gb)+KpG(Gb,Gb))
    GK_AA,_ = graphkernel1(graph_A, graph_A, p, use_vector=use_vector, encoder_model = encoder_model, room_type=room_type)
    GK_BB,_ = graphkernel1(graph_B, graph_B, p, use_vector=use_vector, encoder_model = encoder_model, room_type=room_type)
    GK_AB, node_matrix_AB = graphkernel1(graph_A, graph_B, p, use_vector=use_vector, encoder_model = encoder_model, room_type=room_type)
    GK_AB_prime = GK_AB/max([GK_AA,GK_BB])
    dist = np.sqrt(1 - 2*GK_AB_prime + 1)
    return dist, node_matrix_AB


def retrieval_similar_graph(pred_graph, pred_area, lib, use_vector=False, encoder_model = None, top10 = True, room_type='bedroom'):
    sim_list = []
    for id in lib:
        for area in lib[id]:
            candidate_graph = area['graph']

            similarity, similarity_matrix = graphkernel_dist_metric(pred_graph, candidate_graph, 1, use_vector=use_vector, encoder_model = encoder_model, room_type=room_type)
            # print(similarity)
            sim_list.append((id,candidate_graph,similarity,similarity_matrix))
            # ipdb.set_trace()
    sim_list_sorted = sorted(sim_list, key=lambda x: x[2], reverse=False)

    if top10:
        sim_list_similarity = [] 
        sim_list_info = []
        for sim_list in sim_list_sorted[:10]:
            for data in lib[sim_list[0]]:
                graph = data['graph']
                area = data['area']
                if graph == sim_list[1]:
                    matrix = copy.deepcopy(sim_list[3])  # array(len(pre_graph.nodes), len(graph.nodes),p+1)
                    sim_area = copy.deepcopy(area)
                    similarity = 0
                    info = {}
                    mark_pred_graph = np.zeros(len(pred_graph.nodes))  
                    mark_sim_graph = np.zeros(len(graph.nodes))
                    # 1) Root nodes correspond one-to-one
                    root_similarity = matrix[0][0]
                    similarity+=root_similarity
                    info[0] = 0
                    mark_pred_graph[0] = 1
                    mark_sim_graph[0] = 1
                    matrix[0][0] = -1
                    matrix[0, :].fill(-1)
                    matrix[:, 0].fill(-1)
                    # 2) The remaining nodes select the node with the largest corresponding value in the matrix that has not yet been matched.
                    time = 0
                    while 1:
                        for i in range(len(mark_pred_graph)):
                            if mark_pred_graph[i] == 0:
                                pred_max_i_index = np.argsort(-matrix[i, :])[0]
                                j = pred_max_i_index
                                sim_max_j_index = np.argsort(-matrix[:, j])[0]
                                if sim_max_j_index == i:
                                    similarity += matrix[i,j]
                                    info[j] = i
                                    mark_pred_graph[i] = 1
                                    mark_sim_graph[j] = 1
                                    matrix[i][j] = -1
                                    matrix[i, :].fill(-1)
                                    matrix[:, j].fill(-1)
                        time += 1
                        if (mark_pred_graph==1).all() or (mark_sim_graph==1).all():
                            break
                        if time > 20:
                            ipdb.set_trace()
                    sim_list_similarity.append(similarity)
                    sim_list_info.append(info)
                    break

        sim_index = np.argsort(-np.array(sim_list_similarity))[0]
        sim_info = sim_list_info[sim_index]
        sim_list = sim_list_sorted[sim_index]
        for data in lib[sim_list[0]]:
            graph = data['graph']
            area = data['area']
            if graph == sim_list[1]:
                index = lib[sim_list[0]].index(data)
                sim_area = copy.deepcopy(area)
                sim_func_obj = sim_area['children']
                sim_func_obj['cor_pred_label'] = pred_graph.nodes[0]['tag']
                for i in range(1, len(graph.nodes)):
                    node_tag = graph.nodes[i]['tag']
                    pred_node_tag = None
                    if i in sim_info.keys():
                        pred_node_tag = pred_graph.nodes[sim_info[i]]['tag']
                    if 'children' in sim_func_obj.keys():
                        for sim_obj in sim_func_obj['children']:
                            if node_tag == sim_obj['label']:
                                sim_obj['cor_pred_label'] = pred_node_tag
                break
    else:
        for data in lib[sim_list_sorted[0][0]]:
            graph = data['graph']
            area = data['area']
            if graph == sim_list_sorted[0][1]:
                index = lib[sim_list_sorted[0][0]].index(data)
                matrix = sim_list_sorted[0][3]  # array(len(pre_graph.nodes), len(graph.nodes),p+1)
                sim_area = copy.deepcopy(area)

                # Determine which object in sim_area corresponds to which object in pre_area
                mark_pred_graph = np.zeros(len(pred_graph.nodes)) 
                # 1) Root nodes correspond one-to-one
                sim_func_obj = sim_area['children']
                if graph.nodes[0]['tag'] == sim_func_obj['label']:
                    sim_func_obj['cor_pred_label'] = pred_graph.nodes[0]['tag']
                    mark_pred_graph[0] = 1
                else:
                    ipdb.set_trace()
                # 2) The remaining nodes select the node with the largest corresponding value in the matrix that has not yet been matched.
                for i in range(1, len(graph.nodes)):
                    node_tag = graph.nodes[i]['tag']
                    matrix_order_index = np.argsort(-matrix[:, i])
                    pred_node_tag = None
                    for index in matrix_order_index:
                        if mark_pred_graph[index] == 0:
                            pred_node_tag = pred_graph.nodes[index]['tag']
                            mark_pred_graph[index] = 1
                            break
                    if 'children' in sim_func_obj.keys():
                        for sim_obj in sim_func_obj['children']:
                            if node_tag == sim_obj['label']:
                                sim_obj['cor_pred_label'] = pred_node_tag
                break

        sim_list = sim_list_sorted[0]

    return sim_list[0], index, sim_list[1], sim_area


def get_area_rel_pos_constraints(pred_graph, pred_area, lib, use_vector=False, encoder_model = None, room_type='bedroom'):
    id, index, sim_graph, sim_area = retrieval_similar_graph(pred_graph, pred_area, lib, use_vector = use_vector, encoder_model = encoder_model, room_type = room_type)
    rel_pos_constraints = []
    sim_func_obj = sim_area['children']
    pred_func_obj = pred_area['children']
    sim_pred_label_similarity = get_close_scores(sim_func_obj['label'].split(' ')[0], pred_func_obj['label'].split(' ')[0], 1, cutoff=0)
    if sim_pred_label_similarity >= 0.6 and (sim_func_obj['length'] - sim_func_obj['width']) * (pred_func_obj['length'] - pred_func_obj['width']) < 0:
        o = pred_func_obj['length']
        pred_func_obj['length'] = pred_func_obj['width']
        pred_func_obj['width'] = o

        a = pred_area['length']
        pred_area['length'] = pred_area['width']
        pred_area['width'] = a


    if 'children' in sim_func_obj.keys():
        for sim_obj in sim_func_obj['children']:
            # 1. Obtain the relative coordinates in sim_area
            rel_coordinates = sim_obj['rel_coordinates']
            # 1.1 Visualize the effect of sim_area under a similar coordinate system
            rel_rel_func_2dbdb = np.asarray([[-sim_func_obj['length'] / 2, -sim_func_obj['width'] / 2],
                                             [sim_func_obj['length'] / 2, -sim_func_obj['width'] / 2],
                                             [-sim_func_obj['length'] / 2, sim_func_obj['width'] / 2],
                                             [sim_func_obj['length'] / 2, sim_func_obj['width'] / 2]])
            # draw_bdb2d_point(rel_rel_func_2dbdb, rel_coordinates, str(sim_obj['relative position']))

            # 2. Correct the relative coordinates in pred_area based on the relative coordinates of sim_area.
            # 2.1 Calculate the distance from the relative coordinates to the boundary of the sim_func object
            delta_x_1 = float(sim_func_obj['length'] / 2 - rel_coordinates[0])
            delta_x_2 = float(-sim_func_obj['length'] / 2 - rel_coordinates[0])
            delta_y_1 = float(sim_func_obj['width'] / 2 - rel_coordinates[1])
            delta_y_2 = float(-sim_func_obj['width'] / 2 - rel_coordinates[1])
            delta_z_1 = float(sim_func_obj['height'] / 2 - rel_coordinates[2])
            delta_z_2 = float(-sim_func_obj['height'] / 2 - rel_coordinates[2])

            delta_x = float(delta_x_1 / delta_x_2) if delta_x_2 != 0.0 else None
            delta_y = float(delta_y_1 / delta_y_2) if delta_y_2 != 0.0 else None
            delta_z = float(delta_z_1 / delta_z_2) if delta_z_2 != 0.0 else None

            if delta_x is not None:
                rel_coordinates_x = (pred_func_obj['length'] / 2) * (1 + delta_x) / (1 - delta_x) if delta_x != 1.0 else 0.0
            else:
                rel_coordinates_x = -1 * (pred_func_obj['length'] / 2)
            if delta_y is not None:
                rel_coordinates_y = (pred_func_obj['width'] / 2) * (1 + delta_y) / (1 - delta_y) if delta_y != 1.0 else 0.0
            else:
                rel_coordinates_y = -1 * (pred_func_obj['width'] / 2)
            if delta_z is not None:
                rel_coordinates_z = (pred_func_obj['height'] / 2) * (1 + delta_z) / (1 - delta_z) if delta_z != 1.0 else 0.0
            else:
                rel_coordinates_z = -1 * (pred_func_obj['height'] / 2)

            # 2.2 Visualizing the effect of pred_area in the relative coordinate system
            rel_func_2dbdb = np.asarray([[-pred_func_obj['length'] / 2, -pred_func_obj['width'] / 2],
                                         [pred_func_obj['length'] / 2, -pred_func_obj['width'] / 2],
                                         [-pred_func_obj['length'] / 2, pred_func_obj['width'] / 2],
                                         [pred_func_obj['length'] / 2, pred_func_obj['width'] / 2]])
            # draw_bdb2d_point(rel_func_2dbdb, [rel_coordinates_x, rel_coordinates_y, rel_coordinates_z], str(sim_obj['relative position']))

            rel_pos_constraint = {
                'obj': sim_obj['label'],
                'func_obj': sim_func_obj['label'],
                'cor_pred_obj': sim_obj['cor_pred_label'],
                'cor_pred_func_obj': sim_func_obj['cor_pred_label'],
                'relative_position': sim_obj['relative position'],
                'relative_coordinate': [rel_coordinates_x, rel_coordinates_y, rel_coordinates_z],
                'rel_ori': sim_obj['rel_ori'],
                'object_list':[[sim_func_obj['label'],{'length': sim_func_obj['length'],
                                                       'width': sim_func_obj['width'],
                                                       'height': sim_func_obj['height'],
                                                       'top': sim_func_obj['top'],
                                                       'left': sim_func_obj['left'],
                                                       'depth': sim_func_obj['depth'],
                                                       'orientation': sim_func_obj['orientation']}],
                               [sim_obj['label'], {'length': sim_obj['length'],
                                                        'width': sim_obj['width'],
                                                        'height': sim_obj['height'],
                                                        'top': sim_obj['top'],
                                                        'left': sim_obj['left'],
                                                        'depth': sim_obj['depth'],
                                                        'orientation': sim_obj['orientation']}]
                               ],
                'id': id
            }
            rel_pos_constraints.append(rel_pos_constraint)

            # Adjust the length and width corresponding data of other objects based on the aspect ratio of the retrieved object.
            if 'children' in pred_func_obj.keys():
                for pred_obj in pred_func_obj['children']:
                    o_1, o_2 = pred_obj['relative position'][0], pred_obj['relative position'][2]
                    if sim_obj['cor_pred_label'] == o_1 and sim_func_obj['cor_pred_label'] == o_2:
                        sim_pred_label_similarity = get_close_scores(sim_obj['label'].split(' ')[0], pred_obj['label'].split(' ')[0], 1, cutoff=0)
                        if sim_pred_label_similarity >= 0.6 and (sim_obj['length'] - sim_obj['width']) * (
                                pred_obj['length'] - pred_obj['width']) < 0:
                            o = pred_obj['length']
                            pred_obj['length'] = pred_obj['width']
                            pred_obj['width'] = o


    return rel_pos_constraints


def has_modify_in_area(area):
    modify = False
    if 'children' in area.keys():
        if area['modify_type'] == '(#)':
            modify = True
        elif area['children']['modify_type'] == '(#)':
            modify = True
        else:
            if 'children' in area['children'].keys():
                for obj in area['children']['children']:
                    if obj['modify_type'] == '(#)':
                        modify = True
    return modify


def get_rel_pos_constraints_by_retrieval_graph_for_every_area(area, lib, use_vector = False, encoder_model = None, type = 'one-time', is_update_structure = False, room_type='bedroom'):
    if type == 'one-time' or type == 'generate':
        # print(area)
        if 'children' in area.keys():
            if is_update_structure:
                objs = []
                rel_positions = []
                for func_obj in area['children']:
                    objs.append(func_obj)
                    if 'children' in func_obj.keys():
                        for obj in func_obj['children']:
                            objs.append(obj)
                            rel_positions.append(obj['relative position'])
                graph = area_to_graph(rel_positions)
            else:
                objs = []
                rel_positions = []
                func_obj = area['children']
                objs.append(func_obj)
                if 'children' in func_obj.keys():
                    for obj in func_obj['children']:
                        objs.append(obj)
                        rel_positions.append(obj['relative position'])

                graph = area_to_graph(objs, rel_positions, use_vector=use_vector, encoder_model=encoder_model)
            # draw_area_graph(graph)
            area_rel_pos_constraints = get_area_rel_pos_constraints(graph, area, lib, use_vector=use_vector,
                                                                    encoder_model=encoder_model, room_type=room_type)
            # rel_pos_constraints.append(area_rel_pos_constraints)
            area['rel_pos_constraints'] = [area_rel_pos_constraints]
            # area['rel_pos_constraints'].append(area_rel_pos_constraints)
        else:
            area_rel_pos_constraints = []
            # rel_pos_constraints.append([])
            area['rel_pos_constraints'] = []
            # area['rel_pos_constraints'].append([])
        # ipdb.set_trace()
    elif type == 'modify':
        # print(area)
        if has_modify_in_area(area):
            if is_update_structure:
                objs = []
                rel_positions = []
                for func_obj in area['children']:
                    objs.append(func_obj)
                    if 'children' in func_obj.keys():
                        for obj in func_obj['children']:
                            objs.append(obj)
                            rel_positions.append(obj['relative position'])
                graph = area_to_graph(rel_positions)
            else:
                objs = []
                rel_positions = []
                func_obj = area['children']
                objs.append(func_obj)
                if 'children' in func_obj.keys():
                    for obj in func_obj['children']:
                        objs.append(obj)
                        rel_positions.append(obj['relative position'])
                graph = area_to_graph(objs, rel_positions, use_vector=use_vector, encoder_model=encoder_model)

            # draw_area_graph(graph)
            area_rel_pos_constraints = get_area_rel_pos_constraints(graph, area, lib, use_vector, room_type=room_type)
            # rel_pos_constraints.append(area_rel_pos_constraints)
            area['rel_pos_constraints'] = [area_rel_pos_constraints]
            # area['rel_pos_constraints'].append(area_rel_pos_constraints)
        else:
            area_rel_pos_constraints = []
            # rel_pos_constraints.append([])
            area['rel_pos_constraints'] = []
            # area['rel_pos_constraints'].append([])
    return area, area_rel_pos_constraints


def get_rel_pos_constraints_by_retrieval_graph(data, lib, use_vector = False, encoder_model = None, type = 'one-time', is_update_structure = False, room_type='bedroom'):
    rel_pos_constraints = []
    for area in data['children']:
        _,area_rel_pos_constraints = get_rel_pos_constraints_by_retrieval_graph_for_every_area(area, lib, use_vector=use_vector, encoder_model=encoder_model, type=type, is_update_structure=is_update_structure, room_type = room_type)
        rel_pos_constraints.append(area_rel_pos_constraints)
        # if area_rel_pos_constraints == []:
        #     area['rel_pos_constraints'] = []
        # else:
        #     area['rel_pos_constraints'] = [area_rel_pos_constraints]
        # ipdb.set_trace()

    return data, rel_pos_constraints


def process_obj_x(lib):
    for obj in lib:
        obj['relative_coordinate'][0] = -obj['relative_coordinate'][0]
    return lib


def get_rel_pos_constraints_by_GVAE_for_every_area(area, lib, type = 'one-time', is_update_structure = False, room_type='bedroom'):
    if type == 'one-time' or type == 'generate':
        if 'children' in area.keys():
            area_rel_pos_constraints = process_obj_x(lib)
            area['rel_pos_constraints'] = [area_rel_pos_constraints]
        else:
            area_rel_pos_constraints = []
            area['rel_pos_constraints'] = []
    elif type == 'modify':
        if has_modify_in_area(area):
            area_rel_pos_constraints = process_obj_x(lib)
            area['rel_pos_constraints'] = [area_rel_pos_constraints]
        else:
            area_rel_pos_constraints = []
            area['rel_pos_constraints'] = []
    return area, area_rel_pos_constraints


def get_rel_pos_constraints_by_GVAE(val_id, data, lib, type = 'one-time', is_update_structure = False, room_type='bedroom'):
    rel_pos_constraints = []
    try:
        gvae_val_id = val_id.split('_')[1]
    except:
        gvae_val_id = val_id
    for i in range(len(data['children'])):
        area = data['children'][i]
        scene_lib = copy.deepcopy(lib[gvae_val_id])
        area, area_rel_pos_constraints = get_rel_pos_constraints_by_GVAE_for_every_area(area, scene_lib,
                                                                                                type=type,
                                                                                                is_update_structure=is_update_structure,
                                                                                                room_type=room_type)
        rel_pos_constraints.append(area_rel_pos_constraints)
    return data, rel_pos_constraints