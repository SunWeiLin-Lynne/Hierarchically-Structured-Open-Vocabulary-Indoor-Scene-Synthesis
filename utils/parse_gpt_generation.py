import ast
import json

import ipdb
from plotly.figure_factory import np

from utils.parse_relative_position import parse_relative_position, get_close_scores


def parse_new_hierarchy(gpt_output, is_area = False, is_update_structure = False):
    result = {}
    result["object_list"] = []
    if is_area:
      areas = [gpt_output]
    else:
      areas = gpt_output['children']
    for area in areas:
      if is_update_structure:
        func_objs = area['children']
        for func_obj in func_objs:
          res_func_obj = [func_obj["label"].split(" ")[0],
                          {
                            "length": func_obj["length"],
                            "width": func_obj["width"],
                            "height": func_obj["height"],
                            "left": func_obj["left"],
                            "top": func_obj["top"],
                            "depth": func_obj["depth"],
                            "orientation": func_obj["orientation"],
                            'description': func_obj['description']
                          }]
          result["object_list"].append(res_func_obj)
          if "children" in func_obj.keys():
            for obj in func_obj["children"]:
              res_obj = [obj["label"].split(" ")[0],
                         {
                           "length": obj["length"],
                           "width": obj["width"],
                           "height": obj["height"],
                           "left": obj["left"],
                           "top": obj["top"],
                           "depth": obj["depth"],
                           "orientation": obj["orientation"],
                           'description': obj['description']
                         }
              ]
              result["object_list"].append(res_obj)
      else:
        func_obj = area['children']
        res_func_obj = [func_obj["label"].split(" ")[0],
                        {
                          "length": func_obj["length"],
                          "width": func_obj["width"],
                          "height": func_obj["height"],
                          "left": func_obj["left"],
                          "top": func_obj["top"],
                          "depth": func_obj["depth"],
                          "orientation": func_obj["orientation"],
                          'description': func_obj['description']
                        }]
        result["object_list"].append(res_func_obj)
        if "children" in func_obj.keys():
          for obj in func_obj["children"]:
            res_obj = [obj["label"].split(" ")[0],
                       {
                         "length": obj["length"],
                         "width": obj["width"],
                         "height": obj["height"],
                         "left": obj["left"],
                         "top": obj["top"],
                         "depth": obj["depth"],
                         "orientation": obj["orientation"],
                         'description': obj['description']
                       }
                       ]
            result["object_list"].append(res_obj)
    return result


def parse_new_hierarchy_gpt_output(area_list, func_obj_list, other_obj_list, room_length_width, type = 'one-time', old_hierarchy_data = None, room_type = 'bedroom'):
    # New Format->Json Hierarchy
    if type == 'one-time' or type == 'generate':
        hierarchy_data = {'label': room_type, 'length': room_length_width.split('x')[0],
                          'width': room_length_width.split('x')[1], 'children': []}
        for i in range(len(area_list)):
            area = {}
            area['label'] = area_list[i]['inform'][0]
            area['length'] = float(area_list[i]['inform'][1].split('x')[0])
            try:
                area['width'] = float(area_list[i]['inform'][1].split('x')[1].split('meters')[0])
            except:
                ipdb.set_trace()

            for j in range(len(func_obj_list)):
                func_obj_inform = func_obj_list[j]
                if func_obj_inform['area_index'] == i:
                    try:
                        func_obj = {}
                        func_obj['label'] = func_obj_inform['inform'][0]
                        func_obj['description'] = func_obj_inform['inform'][1]
                        func_obj['length'] = float(func_obj_inform['inform'][2].split('x')[0])
                        func_obj['width'] = float(func_obj_inform['inform'][2].split('x')[1])
                        func_obj['height'] = float(func_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                        if len(func_obj_inform['inform'])>=5:
                            func_obj['orientation'] = float(func_obj_inform['inform'][4])
                        # func_obj['optimization'] = []
                        func_obj['children'] = []
                        for k in range(len(other_obj_list)):
                            try:
                                other_obj_inform = other_obj_list[k]
                                if other_obj_inform['func_index'] == j:
                                    other_obj = {}
                                    other_obj['label'] = other_obj_inform['inform'][0]
                                    other_obj['description'] = other_obj_inform['inform'][1]
                                    other_obj['length'] = float(other_obj_inform['inform'][2].split('x')[0])
                                    other_obj['width'] = float(other_obj_inform['inform'][2].split('x')[1])
                                    other_obj['height'] = float(other_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                                    other_obj['relative position'] = [other_obj_inform['inform'][0],
                                                                      other_obj_inform['inform'][3].split(', ')[0],
                                                                      func_obj_inform['inform'][0]]
                                    if len(other_obj_inform['inform']) >= 5:
                                        other_obj['orientation'] = float(other_obj_inform['inform'][4])
                                    # other_obj['optimization'] = []
                                    func_obj['children'].append(other_obj)
                            except:
                                print('other_obj_inform', other_obj_inform)
                                pass
                        area['children'] = func_obj
                    except:
                        ipdb.set_trace()
                        print('func_obj_inform', func_obj_inform)
                        pass
            # area['optimization'] = []
            # area['rel_pos_constraints'] = []

            hierarchy_data['children'].append(area)
    elif type == 'modify' and old_hierarchy_data is not None:
        hierarchy_data = {'label': room_type, 'length': room_length_width.split('x')[0],
                          'width': room_length_width.split('x')[1], 'children': []}
        for i in range(len(area_list)):
            area = {}
            old_area_is_found = False
            if area_list[i]['modify_type'] == '(#)':
                area['label'] = area_list[i]['inform'][0][3:]
                area['length'] = float(area_list[i]['inform'][1].split('x')[0])
                area['width'] = float(area_list[i]['inform'][1].split('x')[1].split('meters')[0])
                area['modify_type'] = area_list[i]['modify_type']
                # area['optimization'] = []
                # area['rel_pos_constraints'] = []
            elif area_list[i]['modify_type'] == '(-)':
                continue
            else:
                try:
                    area['label'] = area_list[i]['inform'][0]
                    area['length'] = float(area_list[i]['inform'][1].split('x')[0])
                    area['width'] = float(area_list[i]['inform'][1].split('x')[1].split('meters')[0])

                    area['modify_type'] = area_list[i]['modify_type']
                    for o_area in old_hierarchy_data['children']:
                        if o_area['label'].strip() == area['label'].strip() and o_area['length'] == area['length'] and o_area['width'] == area['width']:
                            old_area = o_area
                            old_area_is_found = True
                            break
                    # area['optimization'] = []
                    # area['rel_pos_constraints'] = []
                    area['bdb_x'] = old_area['bdb_x']
                    area['bdb_y'] = old_area['bdb_y']
                    area['bdb_z'] = old_area['bdb_z']
                    area['bdb_l'] = old_area['bdb_l']
                    area['bdb_w'] = old_area['bdb_w']
                    area['bdb_h'] = old_area['bdb_h']
                    area['orientation'] = old_area['orientation']
                    area['tight_length'] = old_area['tight_length']
                    area['tight_width'] = old_area['tight_width']
                except:
                    ipdb.set_trace()
            for j in range(len(func_obj_list)):
                func_obj_inform = func_obj_list[j]
                if func_obj_inform['area_index'] == i:
                    try:
                        func_obj = {}
                        old_func_is_found = False
                        if func_obj_inform['modify_type'] == '(#)':
                            func_obj['label'] = func_obj_inform['inform'][0][3:]
                            func_obj['description'] = func_obj_inform['inform'][1]
                            func_obj['length'] = float(func_obj_inform['inform'][2].split('x')[0])
                            func_obj['width'] = float(func_obj_inform['inform'][2].split('x')[1])
                            func_obj['height'] = float(func_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                            # func_obj['optimization'] = []
                            if len(func_obj_inform['inform']) >= 5:
                                func_obj['orientation'] = float(func_obj_inform['inform'][4])
                            func_obj['children'] = []
                            func_obj['modify_type'] = func_obj_inform['modify_type']

                        elif func_obj_inform['modify_type'] == '(-)':
                            continue
                        else:
                            if old_area_is_found:
                                old_func_obj = old_area['children']
                                old_func_is_found = True
                                func_obj['label'] = old_func_obj['label']
                                func_obj['description'] = old_func_obj['description']
                                func_obj['length'] = old_func_obj['length']
                                func_obj['width'] = old_func_obj['width']
                                func_obj['height'] = old_func_obj['height']
                                # func_obj['optimization'] = []
                                func_obj['children'] = []
                                func_obj['modify_type'] = func_obj_inform['modify_type']
                                func_obj['bdb_x'] = old_func_obj['bdb_x']
                                func_obj['bdb_y'] = old_func_obj['bdb_y']
                                func_obj['bdb_z'] = old_func_obj['bdb_z']
                                func_obj['bdb_l'] = old_func_obj['bdb_l']
                                func_obj['bdb_w'] = old_func_obj['bdb_w']
                                func_obj['bdb_h'] = old_func_obj['bdb_h']
                                func_obj['left'] = old_func_obj['left']
                                func_obj['top'] = old_func_obj['top']
                                func_obj['depth'] = old_func_obj['depth']
                                func_obj['orientation'] = old_func_obj['orientation']
                                func_obj['bdb_ori'] = old_func_obj['bdb_ori']
                            else:
                                func_obj['label'] = func_obj_inform['inform'][0]
                                func_obj['description'] = func_obj_inform['inform'][1]
                                func_obj['length'] = float(func_obj_inform['inform'][2].split('x')[0])
                                func_obj['width'] = float(func_obj_inform['inform'][2].split('x')[1])
                                func_obj['height'] = float(func_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                                # func_obj['optimization'] = []
                                if len(func_obj_inform['inform']) >= 5:
                                    func_obj['orientation'] = float(func_obj_inform['inform'][4])
                                func_obj['children'] = []
                                func_obj['modify_type'] = func_obj_inform['modify_type']


                        for k in range(len(other_obj_list)):
                            try:
                                other_obj_inform = other_obj_list[k]
                                if other_obj_inform['func_index'] == j:
                                    other_obj = {}
                                    if other_obj_inform['modify_type'] == '(#)':
                                        other_obj['label'] = other_obj_inform['inform'][0][3:]
                                        other_obj['description'] = other_obj_inform['inform'][1]
                                        other_obj['length'] = float(other_obj_inform['inform'][2].split('x')[0])
                                        other_obj['width'] = float(other_obj_inform['inform'][2].split('x')[1])
                                        other_obj['height'] = float(other_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                                        other_obj['relative position'] = [other_obj['label'],
                                                                          other_obj_inform['inform'][3].split(', ')[0],
                                                                          func_obj['label']]
                                        if len(other_obj_inform['inform']) >= 5:
                                            other_obj['orientation'] = float(other_obj_inform['inform'][4])
                                        # other_obj['optimization'] = []
                                        other_obj['modify_type'] = other_obj_inform['modify_type']
                                    elif other_obj_inform['modify_type'] == '(-)':
                                        continue
                                    else:
                                        other_obj['label'] = other_obj_inform['inform'][0]
                                        other_obj['description'] = other_obj_inform['inform'][1]
                                        other_obj['length'] = float(other_obj_inform['inform'][2].split('x')[0])
                                        other_obj['width'] = float(other_obj_inform['inform'][2].split('x')[1])
                                        other_obj['height'] = float(other_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                                        if old_func_is_found:
                                            for o_obj in old_func_obj['children']:
                                                if o_obj['label'] == other_obj['label'] and o_obj['length'] == other_obj['length'] and o_obj['width'] == other_obj['width']:
                                                    old_obj = o_obj
                                            other_obj['relative position'] = old_obj['relative position']
                                            # other_obj['optimization'] = []
                                            other_obj['modify_type'] = other_obj_inform['modify_type']
                                            other_obj['bdb_x'] = old_obj['bdb_x']
                                            other_obj['bdb_y'] = old_obj['bdb_y']
                                            other_obj['bdb_z'] = old_obj['bdb_z']
                                            other_obj['bdb_l'] = old_obj['bdb_l']
                                            other_obj['bdb_w'] = old_obj['bdb_w']
                                            other_obj['bdb_h'] = old_obj['bdb_h']
                                            other_obj['left'] = old_obj['left']
                                            other_obj['top'] = old_obj['top']
                                            other_obj['depth'] = old_obj['depth']
                                            other_obj['orientation'] = old_obj['orientation']
                                            other_obj['bdb_ori'] = old_obj['bdb_ori']

                                        else:
                                            other_obj['relative position'] = [other_obj['label'],
                                                                              other_obj_inform['inform'][3].split(', ')[0],
                                                                              func_obj['label']]
                                            # other_obj['optimization'] = []
                                            if len(other_obj_inform['inform']) >= 5:
                                                other_obj['orientation'] = float(other_obj_inform['inform'][4])
                                            other_obj['modify_type'] = other_obj_inform['modify_type']

                                    func_obj['children'].append(other_obj)
                            except:
                                ipdb.set_trace()
                                print('other_obj_inform', other_obj_inform)
                                pass
                        area['children'] = func_obj
                    except:
                        ipdb.set_trace()
                        print('func_obj_inform', func_obj_inform)
                        pass

            hierarchy_data['children'].append(area)

    return hierarchy_data


def parse_new_hierarchy_gpt_output_with_bdb(area_list, func_obj_list, other_obj_list, room_length_width, type = 'one-time', old_hierarchy_data = None, room_type = 'bedroom'):
    hierarchy_data = {'label': room_type, 'length': room_length_width.split('x')[0],
                      'width': room_length_width.split('x')[1], 'children': []}
    for i in range(len(area_list)):
        area = {}
        area['label'] = area_list[i]['inform'][0]
        area['length'] = float(area_list[i]['inform'][1].split('x')[0])
        area['width'] = float(area_list[i]['inform'][1].split('x')[1].split('meters')[0])

        for j in range(len(func_obj_list)):
            func_obj_inform = func_obj_list[j]
            if func_obj_inform['area_index'] == i:
                try:
                    func_obj = {}
                    func_obj['label'] = func_obj_inform['inform'][0]
                    func_obj['description'] = func_obj_inform['inform'][1]

                    length = float(func_obj_inform['inform'][2].split('x')[0])
                    width = float(func_obj_inform['inform'][2].split('x')[1])
                    height = float(func_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                    func_obj_center = ast.literal_eval(func_obj_inform['inform'][3])
                    left, top, depth = func_obj_center[0], func_obj_center[1], func_obj_center[2]
                    ori = float(func_obj_inform['inform'][4])

                    func_obj['length'] = length
                    func_obj['width'] = width
                    func_obj['height'] = height
                    func_obj['left'] = left
                    func_obj['top'] = top
                    func_obj['depth'] = depth
                    func_obj['orientation'] = ori
                    if 'children' not in area.keys():
                        func_obj['children'] = []
                        area['children'] = func_obj
                    else:
                        area['children']['children'].append(func_obj)
                except:
                    ipdb.set_trace()
                    print('func_obj_inform', func_obj_inform)
                    pass

        hierarchy_data['children'].append(area)
    return hierarchy_data


def parse_new_hierarchy_gpt_output_with_bdb_and_rel_position(area_list, func_obj_list, other_obj_list, room_length_width, type = 'one-time', old_hierarchy_data = None, room_type = 'bedroom'):
    hierarchy_data = {'label': room_type, 'length': room_length_width.split('x')[0],
                      'width': room_length_width.split('x')[1], 'children': []}
    for i in range(len(area_list)):
        area = {}
        area['label'] = area_list[i]['inform'][0].split(' ')[0]
        area['length'] = float(area_list[i]['inform'][1].split('x')[0])
        area['width'] = float(area_list[i]['inform'][1].split('x')[1].split('meters')[0])

        for j in range(len(func_obj_list)):
            func_obj_inform = func_obj_list[j]
            if func_obj_inform['area_index'] == i:
                try:
                    func_obj = {}
                    func_obj['label'] = func_obj_inform['inform'][0]
                    func_obj['description'] = func_obj_inform['inform'][1]
                    length = float(func_obj_inform['inform'][2].split('x')[0])
                    width = float(func_obj_inform['inform'][2].split('x')[1])
                    height = float(func_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                    func_obj_center = ast.literal_eval(func_obj_inform['inform'][3])
                    left, top, depth = func_obj_center[0], func_obj_center[1], func_obj_center[2]
                    ori = float(func_obj_inform['inform'][4])

                    func_obj['length'] = length
                    func_obj['width'] = width
                    func_obj['height'] = height
                    func_obj['left'] = left
                    func_obj['top'] = top
                    func_obj['depth'] = depth
                    func_obj['orientation'] = ori
                    func_obj['children'] = []
                    for k in range(len(other_obj_list)):
                        try:
                            other_obj_inform = other_obj_list[k]
                            if other_obj_inform['func_index'] == j:
                                other_obj = {}
                                other_obj['label'] = other_obj_inform['inform'][0]
                                other_obj['description'] = other_obj_inform['inform'][1]
                                other_obj['length'] = float(other_obj_inform['inform'][2].split('x')[0])
                                other_obj['width'] = float(other_obj_inform['inform'][2].split('x')[1])
                                other_obj['height'] = float(
                                    other_obj_inform['inform'][2].split('x')[2].split('meters')[0])
                                other_obj['relative position'] = [other_obj_inform['inform'][0],
                                                                  other_obj_inform['inform'][3].split(', ')[0],
                                                                  func_obj_inform['inform'][0]]
                                other_obj['orientation'] = float(other_obj_inform['inform'][4])
                                # other_obj['optimization'] = []
                                func_obj['children'].append(other_obj)
                        except:
                            print('other_obj_inform', other_obj_inform)
                            pass
                    area['children'] = func_obj
                except:
                    ipdb.set_trace()
                    print('func_obj_inform', func_obj_inform)
                    pass

        hierarchy_data['children'].append(area)
    try:
        hierarchy_data = parse_relative_position(hierarchy_data)
    except:
        ipdb.set_trace()
    return hierarchy_data


def parse_gpt_output(func_obj_list):
    object_list = []
    for j in range(len(func_obj_list)):
        func_obj_inform = func_obj_list[j]
        label = func_obj_inform['inform'][0]
        description = func_obj_inform['inform'][1]
        length = float(func_obj_inform['inform'][2].split('x')[0])
        width = float(func_obj_inform['inform'][2].split('x')[1])
        height = float(func_obj_inform['inform'][2].split('x')[2].split('meters')[0])
        func_obj_center = ast.literal_eval(func_obj_inform['inform'][3])
        left, top, depth = func_obj_center[0], func_obj_center[1], func_obj_center[2]
        ori = float(func_obj_inform['inform'][4])
        res_func_obj = [label.split(" ")[0],
                        {
                            "length": length,
                            "width": width,
                            "height": height,
                            "left": left,
                            "top": top,
                            "depth": depth,
                            "orientation": ori,
                            'description': description
                        }]
        object_list.append(res_func_obj)
    return object_list


def parse_new_hierarchy_gpt_input(data):
    '''

    Args:
        data: hierarchy_data

    Returns: new format

    f'storage_area | 1.66 x 0.65 meters:\n' \
    f'wardrobe 1 | a high-class minimalist wardrobe | 0.86x0.17x1.23 meters\n\n' \
    f'sleeping_area | 1.72 x 2.7 meters:\n' \
    f'double_bed 1 | a sleek and comfortable queen-sized bed with a modern design | 1.19 x 1.14 x 0.67 meters\n' \
    f'nightstand 1 | a delicate nightstand | 0.25 x 0.21 x 0.26 meters | in the right behind of, double_bed 1\n' \
    f'pendant_lamp 1 | a pendant lamp with a modern design | 0.24 x 0.22 x 0.67 meters | above, double_bed 1\n\n' \

    '''
    new_format = ''
    for area in data['children']:
        if 'is_delete' not in area.keys():
            area_label = area['label']
            area_length = area['length']
            area_width = area['width']
            new_format += area_label+' | '+str(area_length)+' x '+str(area_width)+ ' meters:\n'
            func_obj = area['children']
            func_obj_label = func_obj['label']
            func_obj_description = func_obj['description']
            func_obj_length = func_obj['length']
            func_obj_width = func_obj['width']
            func_obj_height = func_obj['height']
            new_format += func_obj_label+' | '+func_obj_description+' | '+str(func_obj_length)+' x '+str(func_obj_width)+' x '+str(func_obj_height)+' meters\n'

            if 'children' in func_obj.keys():
                for obj in func_obj['children']:
                    if 'is_delete' not in obj.keys():
                        obj_label = obj['label']
                        obj_description = obj['description']
                        obj_length = obj['length']
                        obj_width = obj['width']
                        obj_height = obj['height']
                        obj_relationship = f"{obj['relative position'][1]}, {obj['relative position'][2]}"
                        new_format += obj_label + ' | ' + obj_description + ' | ' + str(obj_length) + ' x ' + str(
                            obj_width) + ' x ' + str(obj_height) + ' meters' + ' | ' + obj_relationship + '\n'
            new_format += '\n'
    return new_format


def use_rel_pos_constraint(o_i, constraints, room_type):
    o_1, o_r, o_2 = o_i['relative position'][0], o_i['relative position'][1], o_i['relative position'][2]
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


def roty_3d(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])


def update_obj_position_ori_by_GVAE(val_id, hierarchy_data, rel_pos_lib, room_type):
    rel_pos = rel_pos_lib[val_id.split('_')[1]]
    for i in range(len(hierarchy_data['children'])):
        area = hierarchy_data['children'][i]
        rel_pos_constraint = rel_pos
        func_obj = area['children']
        func_obj_R = func_obj['orientation']
        func_obj_x, func_obj_y, func_obj_z = func_obj['left'], func_obj['top'], func_obj['depth']
        R = roty_3d(-func_obj_R / 180. * np.pi)
        if 'children' in func_obj.keys():
            for obj in func_obj['children']:
                is_use, rel_pos_constraint_i = use_rel_pos_constraint(obj, rel_pos_constraint, room_type)
                if is_use:
                    rel_x_i = rel_pos_constraint_i['relative_coordinate'][0]
                    rel_y_i = rel_pos_constraint_i['relative_coordinate'][1]
                    rel_z_i = rel_pos_constraint_i['relative_coordinate'][2]
                    rel_ori_i = rel_pos_constraint_i['rel_ori']
                    world_coordinate = [-rel_x_i, rel_y_i, rel_z_i] @ R + [func_obj_x, func_obj_y, func_obj_z]
                    world_obj_ori = func_obj_R + rel_ori_i
                    if world_obj_ori<0:
                        world_obj_ori+=360
                    if world_obj_ori>=360:
                        world_obj_ori-=360
                    obj['left'], obj['top'], obj['depth'] = world_coordinate[0], world_coordinate[1], world_coordinate[2]
                    obj['orientation'] = world_obj_ori
    return hierarchy_data

if __name__ == "__main__":
    result = parse_gpt_output(gpt_output, is_area = True, is_update_structure = False)
    prediction_list={
      'query_id': gpt_output["query_id"],
      'iter': 0,
      # 'prompt': gpt_output["prompt"],
      'object_list': result["object_list"],
    }
    ipdb.set_trace()
    with open('../llm_output/3D_hierarchy/delete_try_new.px.json', 'w', encoding='utf-8') as file:
      file.write(json.dumps(result, indent=2, ensure_ascii=False))