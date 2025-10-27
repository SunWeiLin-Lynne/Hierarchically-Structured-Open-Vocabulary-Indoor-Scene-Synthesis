import json
import math

import ipdb
import numpy as np

from utils.info import AREA_FUNCTION_OBJ_FROM_BEDROOM, AREA_FUNCTION_OBJ_FROM_LIVINGROOM, COLOR, \
    HIER_AREA_FUNCTION_OBJ_FROM_BEDROOM, HIER_AREA_FUNCTION_OBJ_FROM_LIVINGROOM, HIER_AREA_OTHER_OBJ_FROM_BEDROOM, \
    HIER_AREA_OTHER_OBJ_FROM_LIVINGROOM
from utils.utils import compute_rel, get_2d_bdbox, actual_distance_of_2d_boxes, get_3d_bdbox, get_room_info, get_layout_info
import matplotlib.pyplot as plt
import open3d as o3d

from utils.get_prompt_3d import get_prompt


def get_xy_distance(obj1, obj2):
    box1_vertices = get_2d_bdbox(obj1)
    box2_vertices = get_2d_bdbox(obj2)
    box1_center = (box1_vertices[0]+box1_vertices[3])/2
    box1_size = abs(box1_vertices[0]-box1_vertices[3])
    box2_center = (box2_vertices[0]+box2_vertices[3])/2
    box2_size = abs(box2_vertices[0] - box2_vertices[3])
    box1 = [box1_center[0], box1_center[1], box1_size[0], box1_size[1]]
    box2 = [box2_center[0], box2_center[1], box2_size[0], box2_size[1]]
    return actual_distance_of_2d_boxes(box1, box2)


def is_same_area(area1, area2):
    box1_vertices = np.asarray([[-area1['length'] / 2, -area1['width'] / 2],
                               [area1['length'] / 2, -area1['width'] / 2],
                               [-area1['length'] / 2, area1['width'] / 2],
                               [area1['length'] / 2, area1['width'] / 2]])
    box1_vertices += np.asarray([[area1['left'], area1['top']]])

    box2_vertices = np.asarray([[-area2['length'] / 2, -area2['width'] / 2],
                                [area2['length'] / 2, -area2['width'] / 2],
                                [-area2['length'] / 2, area2['width'] / 2],
                                [area2['length'] / 2, area2['width'] / 2]])
    box2_vertices += np.asarray([[area2['left'], area2['top']]])

    box1_center = (box1_vertices[0] + box1_vertices[3]) / 2
    box1_size = abs(box1_vertices[0] - box1_vertices[3])
    box2_center = (box2_vertices[0] + box2_vertices[3]) / 2
    box2_size = abs(box2_vertices[0] - box2_vertices[3])
    box1 = [box1_center[0], box1_center[1], box1_size[0], box1_size[1]]
    box2 = [box2_center[0], box2_center[1], box2_size[0], box2_size[1]]
    # offset = min(area1['length'] / 2, area1['width'] / 2, area2['length'] / 2, area2['width'] / 2)
    offset = min(area1['length'], area1['width'], area2['length'], area2['width'])
    distance = actual_distance_of_2d_boxes(box1, box2)
    if distance < offset:
        return True
    else:
        return False


def is_same_area_no_merge(obj1, obj2): 
    offset = min(obj1['length'], obj1['width'], obj1['height'], obj2['length'], obj2['width'], obj2['height'])
    distance = get_xy_distance(obj1, obj2)
    if distance < offset:
        return True
    else:
        return False


def get_objects_relative_position(obj1, obj2):
    box1_vertices = get_3d_bdbox(obj1)
    box2_vertices = get_3d_bdbox(obj2)
    box1 = [min(box1_vertices[:, 0]), min(box1_vertices[:, 1]), min(box1_vertices[:, 2]),
            max(box1_vertices[:, 0]), max(box1_vertices[:, 1]), max(box1_vertices[:, 2])]
    box2 = [min(box2_vertices[:, 0]), min(box2_vertices[:, 1]), min(box2_vertices[:, 2]),
            max(box2_vertices[:, 0]), max(box2_vertices[:, 1]), max(box2_vertices[:, 2])]

    try:
        relation_str, distance = compute_rel(box1, box2, obj2['orientation']/ 180 * math.pi)
    except:
        ipdb.set_trace()
    if relation_str == None:
        ipdb.set_trace()
    relation = [obj1['label'], relation_str, obj2['label']]
    return relation


def update_area_size_and_position(function_area, is_update_structure = False):
    if not is_update_structure:
        for area in function_area:
            area_object = [] 
            area_object.append(area['children'])
            if 'children' in area['children'].keys():
                for obj in area['children']['children']:
                    area_object.append(obj)
            obj_bdbs = []
            for obj in area_object:
                bdbox = get_2d_bdbox(obj)
                obj_bdbs.append(bdbox)
            # ipdb.set_trace()
            area_bdb = [[round(min(np.array(obj_bdbs)[:, 0, 0].tolist() + np.array(obj_bdbs)[:, 3, 0].tolist()),2),
                         round(min(np.array(obj_bdbs)[:, 0, 1].tolist() + np.array(obj_bdbs)[:, 3, 1].tolist()),2)],
                        [round(max(np.array(obj_bdbs)[:, 0, 0].tolist() + np.array(obj_bdbs)[:, 3, 0].tolist()),2),
                         round(max(np.array(obj_bdbs)[:, 0, 1].tolist() + np.array(obj_bdbs)[:, 3, 1].tolist()),2)]]
            area['length'] = round(abs(area_bdb[0][0] - area_bdb[1][0]),2)
            area['width'] = round(abs(area_bdb[0][1] - area_bdb[1][1]),2)
            area['left'] = round((area_bdb[0][0] + area_bdb[1][0]) / 2,2)
            area['top'] = round((area_bdb[0][1] + area_bdb[1][1]) / 2,2)
            area['2d_bounding_box'] = area_bdb[0]+area_bdb[1]
    else:
        for area in function_area:
            area_object = []  # 记录当前area所包含的物体
            for area_obj in area['children']:
                area_object.append(area_obj)
                # ipdb.set_trace()
                if 'children' in area_obj.keys():
                    for obj in area_obj['children']:
                        area_object.append(obj)
            obj_bdbs = []
            for obj in area_object:
                bdbox = get_2d_bdbox(obj)
                obj_bdbs.append(bdbox)
            area_bdb = [[round(min(np.array(obj_bdbs)[:, 0, 0].tolist() + np.array(obj_bdbs)[:, 3, 0].tolist()), 2),
                         round(min(np.array(obj_bdbs)[:, 0, 1].tolist() + np.array(obj_bdbs)[:, 3, 1].tolist()), 2)],
                        [round(max(np.array(obj_bdbs)[:, 0, 0].tolist() + np.array(obj_bdbs)[:, 3, 0].tolist()), 2),
                         round(max(np.array(obj_bdbs)[:, 0, 1].tolist() + np.array(obj_bdbs)[:, 3, 1].tolist()), 2)]]
            area['length'] = round(abs(area_bdb[0][0] - area_bdb[1][0]), 2)
            area['width'] = round(abs(area_bdb[0][1] - area_bdb[1][1]), 2)
            area['left'] = round((area_bdb[0][0] + area_bdb[1][0]) / 2, 2)
            area['top'] = round((area_bdb[0][1] + area_bdb[1][1]) / 2, 2)
            area['2d_bounding_box'] = area_bdb[0] + area_bdb[1]

    return function_area


def draw_id_scene(right_function_area, is_update_structure = False):
    area_color = []
    area_objs = [] # list(6)
    areas = [] # list(6)
    i = 0
    for area in right_function_area:
        this_area = [area['2d_bounding_box'][0], area['2d_bounding_box'][1], 0, abs(area['2d_bounding_box'][0]-area['2d_bounding_box'][2]), abs(area['2d_bounding_box'][1]-area['2d_bounding_box'][3]), 0]
        this_area_obj = []
        bdb_fun_obj_vertices = get_3d_bdbox(area['children'])
        bdbox = [min(bdb_fun_obj_vertices[:, 0]), min(bdb_fun_obj_vertices[:, 1]), min(bdb_fun_obj_vertices[:, 2]),
                max(bdb_fun_obj_vertices[:, 0]), max(bdb_fun_obj_vertices[:, 1]), max(bdb_fun_obj_vertices[:, 2])]
        bdb_fun_obj = [min(bdb_fun_obj_vertices[:, 0]), min(bdb_fun_obj_vertices[:, 1]), min(bdb_fun_obj_vertices[:, 2]), abs(bdbox[0]-bdbox[3]), abs(bdbox[1]-bdbox[4]), abs(bdbox[2]-bdbox[5])]
        # ipdb.set_trace()
        this_area_obj.append(bdb_fun_obj)
        if 'children' in area['children'].keys():
            for obj in area['children']['children']:
                bdb_obj_vertices = get_3d_bdbox(obj)
                bdbox_obj = [min(bdb_obj_vertices[:, 0]), min(bdb_obj_vertices[:, 1]), min(bdb_obj_vertices[:, 2]),
                             max(bdb_obj_vertices[:, 0]), max(bdb_obj_vertices[:, 1]), max(bdb_obj_vertices[:, 2])]
                bdb_obj = [min(bdb_obj_vertices[:, 0]), min(bdb_obj_vertices[:, 1]), min(bdb_obj_vertices[:, 2]), abs(bdbox_obj[0]-bdbox_obj[3]), abs(bdbox_obj[1]-bdbox_obj[4]), abs(bdbox_obj[2]-bdbox_obj[5])]
                this_area_obj.append(bdb_obj)
        areas.append(this_area)
        area_objs.append(this_area_obj)
        area_color.append(COLOR[area['label']])
        i+=1
    fig = plt.figure("3D Box") 
    ax = fig.add_subplot(projection='3d')
    for i in range(len(areas)):
        area = areas[i]
        a_x, a_y, a_z = area[0], area[1], area[2]
        a_dx, a_dy, a_dz = area[3], area[4], area[5]
        ax.bar3d(a_x, a_y, a_z, a_dx, a_dy, a_dz, edgecolor=area_color[i], linewidth=0.5, alpha=0)
        for obj in area_objs[i]:
            x, y, z = obj[0], obj[1], obj[2]
            dx, dy, dz = obj[3], obj[4], obj[5]
            ax.bar3d(x, y, z, dx, dy, dz,edgecolor=area_color[i], linewidth=0.5, alpha=0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def get_hierarchical_tree_content(data, unit, is_update_structure = False, meta_data = None, room_type = 'bedrooom'):
    hierarchical_trees = {}
    for id in data:
        hierarchical_tree = {}
        condition = data[id][0]
        layout = data[id][1]
        room_name, room_max_length, room_max_width = get_room_info(condition, unit = unit)
        layout_information_list = get_layout_info(layout, unit=unit)
        layout_information = {}
        object_information = []  # [{label, length, width, height, left,top,depth,orientation},...] for object
        for i in range(len(layout_information_list)):
            object_info_list = layout_information_list[i]
            if meta_data is not None:
                obj_jid = meta_data[id]['jids'][i]
            else:
                obj_jid = None
        # for object_info_list in layout_information_list:
            object_name, object_length, object_width, object_height, object_left, object_top, object_depth, object_orientation = object_info_list

            object_info_dir_layout = {"length": float(object_length),
                                      "width": float(object_width),
                                      "height": float(object_height),
                                      "left": float(object_left),
                                      "top": float(object_top),
                                      "depth": float(object_depth),
                                      "orientation": int(object_orientation),
                                      "obj_jid": obj_jid}

            if object_name in layout_information.keys():
                o_id = len(layout_information[object_name]) + 1
                layout_information[object_name][o_id] = object_info_dir_layout
            else:
                o_id = 1
                layout_information[object_name] = {}
                layout_information[object_name][o_id] = object_info_dir_layout

            object_info_dir = {"label": object_name + " " + str(o_id),
                               "length": float(object_length),
                               "width": float(object_width),
                               "height": float(object_height),
                               "left": float(object_left),
                               "top": float(object_top),
                               "depth": float(object_depth),
                               "orientation": int(object_orientation),
                               "obj_jid": obj_jid}

            obj_box_vertices = get_3d_bdbox(object_info_dir)
            obj_box = [round(min(obj_box_vertices[:, 0]), 2), round(min(obj_box_vertices[:, 1]), 2),
                       round(min(obj_box_vertices[:, 2]), 2),
                       round(max(obj_box_vertices[:, 0]), 2), round(max(obj_box_vertices[:, 1]), 2),
                       round(max(obj_box_vertices[:, 2]), 2)]
            object_info_dir['3d_bounding_box'] = obj_box
            object_information.append(object_info_dir)

        if not is_update_structure:
            # 1. Find the objects in the scene that best reflect the functions, and determine the designated functional areas.
            function_area = []  # [{label, length, width, left, top, children:{label, length, width, height, left,top,depth,orientation}},...] for area
            other_object_information = []  # [{label, length, width, height, left,top,depth,orientation},...] for other object without obvious function
            if room_type == 'bedroom':
                # AREA_FUNCTION_OBJ = AREA_FUNCTION_OBJ_FROM_BEDROOM
                AREA_FUNCTION_OBJ = HIER_AREA_FUNCTION_OBJ_FROM_BEDROOM
                AREA_OTHER_OBJ = HIER_AREA_OTHER_OBJ_FROM_BEDROOM
            elif room_type == 'livingroom':
                AREA_FUNCTION_OBJ = HIER_AREA_FUNCTION_OBJ_FROM_LIVINGROOM
                AREA_OTHER_OBJ = HIER_AREA_OTHER_OBJ_FROM_LIVINGROOM
            for object in object_information:
                area = {}
                for key, item in AREA_FUNCTION_OBJ.items():
                    label = object['label'].split(" ")[0]
                    if label in item:
                        area['label'] = key
                        area['length'] = object['length']
                        area['width'] = object['width']
                        area['left'] = object['left']
                        area['top'] = object['top']
                        area['children'] = object
                        break
                if len(area) == 0:
                    other_object_information.append(object)
                else:
                    function_area.append(area)
            # When function_area is empty, the current area is considered as other areas. 
            # The first object is selected as the functional object, and the other objects are its child objects.
            if function_area == []:
                other_object_information = []
                for i in range(len(object_information)):
                    object = object_information[i]
                    if i == 0:
                        area = {}
                        area['label'] = 'other_area'
                        area['length'] = object['length']
                        area['width'] = object['width']
                        area['left'] = object['left']
                        area['top'] = object['top']
                        area['children'] = object
                        function_area.append(area)
                    else:
                        other_object_information.append(object)

            # 2. Cluster according to distance based on functional objects
            other_object_area = []  # Record which functional object each other non-functional object was assigned to in each functional area. ["area_index"+"_"+"function_object_index"]
            for o_object in other_object_information:
                # Determine whether an object must exist in a certain type of area: 
                # if so, assign it directly; 
                # otherwise, calculate the distance between the object and all functional objects in the areas, and select the nearest area as the area to which the current object belongs.
                area_key = 'any_area'
                is_found = False
                found_area_id = 0
                if room_type == 'livingroom' or room_type == 'bedroom':
                    for key, item in AREA_OTHER_OBJ.items():
                        label = o_object['label'].split(" ")[0]
                        if label in item:
                            area_key = key
                            break
                    if area_key != 'any_area':
                        for f_area_id in range(len(function_area)):
                            if function_area[f_area_id]['label'] == area_key:
                                found_area_id = f_area_id
                                is_found = True
                                break
                if area_key != 'any_area' and is_found:
                    area_id = found_area_id
                else:
                    area_dis = []
                    for area in function_area:
                        area_obj = area['children']
                        area_dis.append(get_xy_distance(o_object, area_obj))
                    area_id = area_dis.index(min(area_dis))
                other_object_area.append(str(area_id))
                if 'children' in function_area[area_id]['children'].keys():
                    function_area[area_id]['children']['children'].append(o_object)
                else:
                    function_area[area_id]['children']['children'] = [o_object]

            # 3. Update the area size and position
            function_area = update_area_size_and_position(function_area, is_update_structure)
            # 4. Attempt to cluster regions that are very similar and have the same functions
            right_function_area = []  # [{label, length, width, left, top, children:{label, length, width, height, left,top,depth,orientation,children:[]}},...] for area
            for area in function_area:
                sign = 0
                for right_area in right_function_area:
                    if area['label'] == right_area['label']:  # # Condition 1: Same category
                        if is_same_area(area, right_area):  # Condition 2: Close proximity
                            this_function_obj = {'label': area['children']['label'],
                                                 'length': area['children']['length'],
                                                 'width': area['children']['width'],
                                                 'height': area['children']['height'],
                                                 'left': area['children']['left'], 'top': area['children']['top'],
                                                 'depth': area['children']['depth'], 'orientation': area['children'][
                                    'orientation'],'3d_bounding_box': area['children']['3d_bounding_box'], 'obj_jid': area['children']['obj_jid']}  # {label, length, width, height, left, top, depth, orientation}
                            other_object_information.append(this_function_obj)
                            other_object_area.append(str(right_function_area.index(right_area)))
                            if 'children' in right_area['children'].keys():
                                right_area['children']['children'].append(this_function_obj)
                            else:
                                right_area['children']['children'] = [this_function_obj]
                            if 'children' in area['children'].keys():
                                this_function_obj_children = area['children']['children']
                                for k in range(len(other_object_area)):
                                    if other_object_area[k] == str(function_area.index(area)):
                                        other_object_area[k] = str(right_function_area.index(right_area))
                                for obj in this_function_obj_children:
                                    right_area['children']['children'].append(obj)
                            sign = 1
                            break
                if sign == 0:
                    if 'children' in area['children'].keys():
                        for k in range(len(other_object_area)):
                            if other_object_area[k] == str(function_area.index(area)):
                                other_object_area[k] = str(len(right_function_area))
                    right_function_area.append(area)
                else:
                    right_function_area = update_area_size_and_position(right_function_area,is_update_structure)

            # 5. Add the relative position relationship between parent and child nodes
            for i in range(len(other_object_information)):
                area_index = other_object_area[i]
                other_object = other_object_information[i]
                try:
                    parent_object = right_function_area[int(area_index)]['children']
                except:
                    ipdb.set_trace()
                relations = get_objects_relative_position(other_object, parent_object)
                for obj in right_function_area[int(area_index)]['children']['children']:
                    if obj["label"] == other_object["label"]:
                        obj["relative position"] = relations

            hierarchical_tree["label"] = room_name
            hierarchical_tree["length"] = float(room_max_length)
            hierarchical_tree["width"] = float(room_max_width)
            hierarchical_tree['2d_bounding_box'] = [0, 0, float(room_max_length), float(room_max_width)]
            hierarchical_tree["children"] = right_function_area
            hierarchical_trees[id] = hierarchical_tree
            # print(id)
            # print(hierarchical_tree)
            # draw_id_scene(right_function_area)
        else:
            # 1. Find the objects in the scene that best reflect the functions, and determine the designated functional areas.
            function_area = []  # [{label, children:{label, length, width, height, left,top,depth,orientation}},...] for area
            other_object_information = []  # [{label, length, width, height, left,top,depth,orientation},...] for other object without obvious function
            for object in object_information:
                area = {}
                for key, item in AREA_FUNCTION_OBJ.items():
                    label = object['label'].split(" ")[0]
                    if label in item:
                        area['label'] = key
                        area['length'] = object['length']
                        area['width'] = object['width']
                        area['left'] = object['left']
                        area['top'] = object['top']
                        area['children'] = [object]
                        break
                if len(area) == 0:
                    other_object_information.append(object)
                else:
                    function_area.append(area)

            # 2. Cluster according to distance based on functional objects
            other_object_area = [] 
            for o_object in other_object_information:
                area_dis = []
                for area in function_area:
                    area_obj = area['children'][0]
                    area_dis.append(get_xy_distance(o_object, area_obj))
                area_id = area_dis.index(min(area_dis))
                other_object_area.append(str(area_id) + "_" + str(0))
                if 'children' in function_area[area_id]['children'][0].keys():
                    function_area[area_id]['children'][0]['children'].append(o_object)
                else:
                    function_area[area_id]['children'][0]['children'] = [o_object]

            # 3. Update the area size and position
            function_area = update_area_size_and_position(function_area,is_update_structure)

            # 4. Attempt to cluster regions that are very similar and have the same functions
            right_function_area = []  # [{label, children:[{label, length, width, height, left,top,depth,orientation,children:[]}}],...] for area
            for area in function_area:
                sign = 0
                for right_area in right_function_area:
                    if area['label'] == right_area['label']:
                        for right_area_child_object in right_area['children']:
                            if is_same_area_no_merge(area['children'][0], right_area_child_object): 
                                # print(str(function_area.index(area))+"_"+str(0))
                                if str(function_area.index(area)) + "_" + str(0) in other_object_area:
                                    for k in range(len(other_object_area)):
                                        if other_object_area[k] == str(function_area.index(area)) + "_" + str(0):
                                            other_object_area[k] = str(right_function_area.index(right_area)) + "_" + str(len(right_area['children']))

                                right_area['children'].append(area['children'][0])
                                sign = 1
                                break
                if sign == 0:
                    # print(str(function_area.index(area)) + "_" + str(0))
                    if str(function_area.index(area)) + "_" + str(0) in other_object_area:
                        for k in range(len(other_object_area)):
                            if other_object_area[k] == str(function_area.index(area)) + "_" + str(0):
                                other_object_area[k] = str(len(right_function_area)) + "_" + str(0)
                    right_function_area.append(area)
                else:
                    right_function_area = update_area_size_and_position(right_function_area,is_update_structure)
            # print('ID:', id)
            # print('Other_object_area:', other_object_area)

            # 5. Add the relative position relationship between parent and child nodes
            for i in range(len(other_object_information)):
                area_index = other_object_area[i].split("_")[0]
                function_object_index = other_object_area[i].split("_")[1]
                other_object = other_object_information[i]
                try:
                    parent_object = right_function_area[int(area_index)]['children'][int(function_object_index)]
                except:
                    ipdb.set_trace()
                relations = get_objects_relative_position(other_object, parent_object)
                for obj in right_function_area[int(area_index)]['children'][int(function_object_index)]['children']:
                    if obj['label'] == other_object['label']:
                        obj['relative position'] = relations
            hierarchical_tree['label'] = room_type
            hierarchical_tree['length'] = room_max_length
            hierarchical_tree['width'] = room_max_width
            hierarchical_tree['2d_bounding_box'] = [0, 0, room_max_length, room_max_width]
            hierarchical_tree['children'] = right_function_area
            hierarchical_trees[id] = hierarchical_tree

    # ipdb.set_trace()
    return hierarchical_trees


def get_hierarchical_tree(data, unit, is_update_structure = False, room_type = 'bedroom'):
    new_hierarchical_trees = {}
    hierarchical_trees = get_hierarchical_tree_content(data, unit, is_update_structure, room_type = room_type)
    for id in hierarchical_trees:
        new_hierarchical_tree = {}
        hierarchical_tree = hierarchical_trees[id]
        new_hierarchical_tree['label'] = hierarchical_tree['label']
        new_hierarchical_tree['length'] = hierarchical_tree['length']
        new_hierarchical_tree['width'] = hierarchical_tree['width']
        new_hierarchical_tree['2d_bounding_box'] = hierarchical_tree['2d_bounding_box']
        if not is_update_structure:
            new_areas = []
            for area in hierarchical_tree['children']:
                new_area = {
                    'label': area['label'],
                    'length': area['length'],
                    'width': area['width'],
                    '2d bounding box': area['2d_bounding_box'],
                }
                fun_obj = area['children']
                new_fun_obj = {
                    'label': fun_obj['label'],
                    'length': fun_obj['length'],
                    'width': fun_obj['width'],
                    'height': fun_obj['height'],
                    '3d bounding box': fun_obj['3d_bounding_box'],
                    'orientation': fun_obj['orientation'],
                }
                if 'children' in fun_obj.keys():
                    new_fun_obj_children = []
                    for obj in fun_obj['children']:
                        new_obj = {
                            'label': obj['label'],
                            'length': obj['length'],
                            'width': obj['width'],
                            'height': obj['height'],
                            'orientation': obj['orientation'],
                            'relative position': obj['relative position']
                        }
                        new_fun_obj_children.append(new_obj)
                    new_fun_obj['children'] = new_fun_obj_children
                new_area['children'] = new_fun_obj
                new_areas.append(new_area)
        else:
            new_areas = []
            for area in hierarchical_tree['children']:
                new_area = {
                    'label': area['label'],
                    'length': area['length'],
                    'width': area['width'],
                    '2d bounding box': area['2d_bounding_box'],
                }
                new_area_children = []
                for fun_obj in area['children']:
                    new_fun_obj = {
                        'label': fun_obj['label'],
                        'length': area['length'],
                        'width': area['width'],
                        'height': area['height'],
                        '3d bounding box': fun_obj['3d_bounding_box'],
                        'orientation': fun_obj['orientation'],
                    }
                    if 'children' in fun_obj.keys():
                        new_fun_obj_children = []
                        for obj in fun_obj['children']:
                            new_obj = {
                                'label': obj['label'],
                                'length': obj['length'],
                                'width': obj['width'],
                                'height': obj['height'],
                                'orientation': obj['orientation'],
                                'relative position': obj['relative position']
                            }
                            new_fun_obj_children.append(new_obj)
                        new_fun_obj['children'] = new_fun_obj_children
                    new_area_children.append(new_fun_obj)
                new_area['children'] = new_area_children
                new_areas.append(new_area)
        new_hierarchical_tree['children'] = new_areas
        new_hierarchical_trees[id] = new_hierarchical_tree
    return new_hierarchical_trees

def add_hierarchical_tree(data,unit, is_update_structure = False):
    new_data = {}
    hierarchical_trees = get_hierarchical_tree(data,unit, is_update_structure)
    for id in data:
        condition = data[id][0]
        hierarchical_tree = hierarchical_trees[id]
        layout = 'Layout:\n' + str(hierarchical_tree) + '\n'
        new_id_data = [condition, layout]
        new_data[id] = new_id_data
    return new_data

def add_prompt_and_hierarchical_tree(data,unit, is_update_structure = False):
    new_data={}
    prompts = get_prompt(data, unit)
    hierarchical_trees = get_hierarchical_tree(data,unit, is_update_structure)
    for id in data:
        condition = data[id][0]
        prompt = prompts[id]
        hierarchical_tree = hierarchical_trees[id]
        layout = 'Layout:\n' + str(hierarchical_tree) + '\n'
        new_id_data = [prompt, condition, layout]
        new_data[id] = new_id_data
        # ipdb.set_trace()
    # ipdb.set_trace()
    return new_data, prompts

def add_new_hierarchical_tree(data,unit, is_update_structure = False, room_type = 'bedroom'):
    new_data = {}
    hierarchical_trees = get_hierarchical_tree(data, unit, is_update_structure, room_type)
    for id in data:
        condition = data[id][0]
        room_max_length = condition.split('\n')[2].split(': ')[1].split(', ')[0].split(' ')[-1][:-1]
        room_max_width = condition.split('\n')[2].split(': ')[1].split(', ')[1].split(' ')[-1][:-1]
        new_condition = condition.split('\n')[0]+'\n'+ \
                        condition.split('\n')[1]+'\n'+ \
                        f"Room Size: {room_max_length} x {room_max_width} meters\n"
        hierarchical_tree = hierarchical_trees[id]
        new_hierarchical_tree = ''
        for area in hierarchical_tree['children']:
            area_label = area['label']
            fun_obj = area['children']
            fun_obj_label = fun_obj['label']
            fun_obj_size = f"{fun_obj['length']}x{fun_obj['width']}x{fun_obj['height']} meters"
            other_obj_info = ''
            if 'children' in fun_obj.keys():
                for obj in fun_obj['children']:
                    obj_label = obj['label']
                    obj_size = f"{obj['length']}x{obj['width']}x{obj['height']} meters"
                    obj_rel = obj['relative position'][1]+', '+obj['relative position'][2]
                    other_obj_info+=obj_label+' | '+obj_size+' | '+obj_rel+'\n'
            new_hierarchical_tree+=area_label+': \n'+ fun_obj_label+' | '+fun_obj_size+' | '+'\n'+other_obj_info+'\n'
        new_layout = 'Layout:\n' + str(new_hierarchical_tree)
        new_id_data = [new_condition, new_layout]
        new_data[id] = new_id_data
    return new_data

if __name__ == "__main__":
    with open(f"../dataset/3D/bedroom/bedroom.val.json", "r") as file:
        val_data = json.load(file)
    hierarchical_val_data = add_hierarchical_tree(val_data, 'm', is_update_structure=False)
    ipdb.set_trace()
