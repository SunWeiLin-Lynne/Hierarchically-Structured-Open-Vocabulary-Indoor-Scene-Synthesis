import difflib
import json
import math
import pdb
import torch
from PIL import Image
import ipdb
from shapely.geometry import LineString, box
from shapely import affinity
import numpy as np
import networkx as nx
OBJ_LIST = ['dressing_table', 'double_bed', 'kids_bed', 'single_bed', 'coffee_table', 'tv_stand', 'sofa', 'bookshelf', 'cabinet', 'children_cabinet', 'wardrobe', 'shelf',
            'desk', 'table', 'armchair', 'ceiling_lamp', 'chair', 'dressing_chair', 'floor_lamp', 'nightstand', 'pendant_lamp', 'stool']

def load_json(fname):
    with open(fname, "r") as file:
        data = json.load(file)
    return data


def write_json(fname, data):
    with open(fname, "w") as file:
        json.dump(data, file, indent=4, separators=(",",":"), sort_keys=True)


def bb_relative_position(boxA, boxB):
    xA_c = (boxA[0]+boxA[2])/2
    yA_c = (boxA[1]+boxA[3])/2
    xB_c = (boxB[0]+boxB[2])/2
    yB_c = (boxB[1]+boxB[3])/2
    dist = np.sqrt((xA_c - xB_c)**2 + (yA_c - yB_c)**2)
    cosAB = (xA_c-xB_c) / dist
    sinAB = (yB_c-yA_c) / dist
    return cosAB, sinAB
    

def eval_spatial_relation(bbox1, bbox2):
    theta = np.sqrt(2)/2
    relation = 'diagonal'

    if bbox1 == bbox2:
        return relation
    
    cosine, sine = bb_relative_position(bbox1, bbox2)

    if cosine > theta:
        relation = 'right'
    elif sine > theta:
        relation = 'top'
    elif cosine < -theta:
        relation = 'left'
    elif sine < -theta:
        relation = 'bottom'
    
    return relation


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def my_iou_3d(box1, box2):

    result_xy, result_z, result_v = [], [], []
    for b in [box1, box2]:
        x, y, z, l, w, h, yaw = b
        # ipdb.set_trace()

        result_v.append(l * w * h)


        ls = LineString([[0, z - h / 2], [0, z + h / 2]])
        result_z.append(ls)


        poly = box(x - l / 2, y - w / 2, x + l / 2, y + w / 2)
        poly_rot = affinity.rotate(poly, yaw, use_radians=True)
        result_xy.append(poly_rot)

    overlap_xy = result_xy[0].intersection(result_xy[1]).area
    overlap_z = result_z[0].intersection(result_z[1]).length

    overlap_xyz = overlap_z * overlap_xy
    return overlap_xyz / (np.sum(result_v) - overlap_xyz)


def actual_distance_of_2d_boxes(box1, box2): 

    x1, y1, l1, w1 = box1
    x2, y2, l2, w2 = box2

    Dx = abs(x2 - x1)
    Dy = abs(y2 - y1)

    # Two rectangles do not intersect. For two rectangles that partially overlap along the X-axis, 
    # the minimum distance is the distance between the bottom edge of the upper rectangle and the top edge of the lower rectangle.
    if (Dx < ((l1 + l2) / 2)) and (Dy >= ((w1 + w2) / 2)):
        min_dist_xy = Dy - ((w1 + w2) / 2)

    # Two rectangles do not intersect, but partially overlap along the Y-axis. 
    # The minimum distance is the distance between the right edge of the left rectangle and the left edge of the right rectangle.
    elif (Dx >= ((l1 + l2) / 2)) and (Dy < ((w1 + w2) / 2)):
        min_dist_xy = Dx - ((l1 + l2) / 2)

    #Two rectangles do not intersect. For two rectangles that do not overlap along the X-axis and Y-axis, 
    # the minimum distance is the distance between the two closest vertices.
    elif (Dx >= ((l1 + l2) / 2)) and (Dy >= ((w1 + w2) / 2)):
        delta_x = Dx - ((l1 + l2) / 2)
        delta_y = Dy - ((w1 + w2) / 2)
        min_dist_xy = math.sqrt(delta_x * delta_x + delta_y * delta_y)

    # If two rectangles intersect, the minimum distance is negative, return -1
    else:
        min_dist_xy = -1

    return min_dist_xy

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, s],
                     [-s,c]])

def roty_3d(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, s, 0],
                     [-s, c, 0],
                     [0, 0, 1]])


def categorical_kl(p, q):
    return (p * (np.log(p + 1e-6) - np.log(q + 1e-6))).sum()


def get_room_info(condition, unit = ''):
    '''

    Args:
        layout: string represent of layout
        unit: '', px, m

    Returns:

    '''
    # 1. get room information from condition
    room_type = condition.split('\n')[1].split(': ')[1]
    if unit in ['px']:
        room_max_length = condition.split('\n')[2].split(': ')[1].split(', ')[0].split(' ')[-1][:-2]
        room_max_width = condition.split('\n')[2].split(': ')[1].split(', ')[1].split(' ')[-1][:-2]
    elif unit in ['']:
        room_max_length = condition.split('\n')[2].split(': ')[1].split(', ')[0].split(' ')[-1]
        room_max_width = condition.split('\n')[2].split(': ')[1].split(', ')[1].split(' ')[-1]
    elif unit in ['m']:
        room_max_length = condition.split('\n')[2].split(': ')[1].split(', ')[0].split(' ')[-1][:-1]
        room_max_width = condition.split('\n')[2].split(': ')[1].split(', ')[1].split(' ')[-1][:-1]

    return room_type, room_max_length, room_max_width


def get_layout_info(layout, unit = ''):
    '''

    Args:
        layout: string represent of layout
        unit: '', px, m

    Returns:

    '''
    # get layout information from layout
    layout_information = []
    for i in range(1, len(layout.split('\n')) - 1):
        if unit in ['px']:
            object_info = layout.split('\n')[i]
            object_name = object_info.split(' {')[0]
            object_length = object_info.split(' {')[1].split(';')[0].split(': ')[1][:-2]
            object_width = object_info.split(' {')[1].split(';')[1].split(': ')[1][:-2]
            object_height = object_info.split(' {')[1].split(';')[2].split(': ')[1][:-2]
            object_left = object_info.split(' {')[1].split(';')[3].split(': ')[1][:-2]
            object_top = object_info.split(' {')[1].split(';')[4].split(': ')[1][:-2]
            object_depth = object_info.split(' {')[1].split(';')[5].split(': ')[1][:-2]
            object_orientation = object_info.split(' {')[1].split(';')[6].split(': ')[1].split(' ')[0]
        elif unit in ['']:
            object_info = layout.split('\n')[i]
            object_name = object_info.split(' {')[0]
            object_length = object_info.split(' {')[1].split(';')[0].split(': ')[1]
            object_width = object_info.split(' {')[1].split(';')[1].split(': ')[1]
            object_height = object_info.split(' {')[1].split(';')[2].split(': ')[1]
            object_left = object_info.split(' {')[1].split(';')[3].split(': ')[1]
            object_top = object_info.split(' {')[1].split(';')[4].split(': ')[1]
            object_depth = object_info.split(' {')[1].split(';')[5].split(': ')[1]
            object_orientation = object_info.split(' {')[1].split(';')[6].split(': ')[1].split(' ')[0]
        elif unit in ['m']:
            object_info = layout.split('\n')[i]
            object_name = object_info.split(' {')[0]
            object_length = object_info.split(' {')[1].split(';')[0].split(': ')[1][:-1]
            object_height = object_info.split(' {')[1].split(';')[1].split(': ')[1][:-1]
            object_width = object_info.split(' {')[1].split(';')[2].split(': ')[1][:-1]
            object_orientation = object_info.split(' {')[1].split(';')[3].split(': ')[1].split(' ')[0]
            object_left = object_info.split(' {')[1].split(';')[4].split(': ')[1][:-1]
            object_top = object_info.split(' {')[1].split(';')[5].split(': ')[1][:-1]
            object_depth = object_info.split(' {')[1].split(';')[6].split(': ')[1][:-1]

        object_info_list = [object_name, float(object_length), float(object_width), float(object_height),
                            float(object_left), float(object_top), float(object_depth), float(object_orientation)]

        layout_information.append(object_info_list)

    return layout_information


def dict_bbox_to_vec(dict_box):
    '''
    input: {'min': [1,2,3], 'max': [4,5,6]}
    output: [1,2,3,4,5,6]
    # input: {'min': [1,2,3], 'max': [4,5,6]}
    # output: [1,3,2,4,6,5]
    '''
    #
    # dict_box['min'][1], dict_box['min'][2] = dict_box['min'][2], dict_box['min'][1]
    # dict_box['max'][1], dict_box['max'][2] = dict_box['max'][2], dict_box['max'][1]
    return dict_box['min'] + dict_box['max']


def normalize_pi(theta):
    if -math.pi<=theta<=math.pi:
        return theta
    elif -math.pi<=theta+2*math.pi<=math.pi:
        return theta+2*math.pi
    elif -math.pi<=theta-2*math.pi<=math.pi:
        return theta - 2 * math.pi
    else:
        ipdb.set_trace()

def compute_rel(box1, box2, R = None, room_max_length=None, room_max_width=None):
    '''

    Args:
        box1: [0,1,2,3,4,5], coordinates of the minimum and maximum angles
        box2:
        R: the relative position between box1 and box2, calculated in the coordinate system of box2 (in radians)

    Returns:

    '''
    center1 = np.array([(box1[0] + box1[3]) / 2, (box1[1] + box1[4]) / 2, (box1[2] + box1[5]) / 2])
    center2 = np.array([(box2[0] + box2[3]) / 2, (box2[1] + box2[4]) / 2, (box2[2] + box2[5]) / 2])

    d = center1 - center2
    theta = math.atan2(d[1], d[0])
    if R is not None:
        theta = normalize_pi(theta + R)

    distance = (d[1] ** 2 + d[0] ** 2) ** 0.5

    p = None
    if room_max_length is not None and room_max_width is not None:
        # eliminate relation in vertical axis
        if abs(center1[2] - center2[2]) - abs(center1[2] - box1[2]) - abs(center2[2] - box2[2]) > 2*(abs(box1[5]-box1[2])+abs(box2[5]-box2[2])):
            return p, distance

        # eliminate relationships on the horizontal plane
        if abs(center1[0] - center2[0]) - abs(center1[0] - box1[0]) - abs(center2[0] - box2[0]) > float(
                room_max_width) / 3 or abs(center1[1] - center2[1]) - abs(center1[1] - box1[1]) - abs(
            center2[1] - box2[1]) > float(room_max_length) / 3:
            return p, distance

    # 2.3 "on" relationship:
    if center1[0] >= box2[0] and center1[0] <= box2[3]:
        if center1[1] >= box2[1] and center1[1] <= box2[4]:
            delta1 = center1[2] - center2[2] 
            delta2 = (box1[5] - box1[2] + box2[5] - box2[2]) / 2
            if 0 < (delta1 - delta2) < min((box1[5] - box1[2]),(box2[5] - box2[2]))/5: 
                p = 'on top of'
                return p, distance
            elif min((box1[5] - box1[2]),(box2[5] - box2[2]))/5 < (delta1 - delta2):
                p = 'above'
                return p, distance
            elif min((box1[5] - box1[2]),(box2[5] - box2[2]))/5 < (delta2 - delta1):
                p = 'below'
                return p, distance


    # 2.4 Other relationship
    sx0, sy0, sz0, sx1, sy1, sz1 = box1
    ox0, oy0, oz0, ox1, oy1, oz1 = box2
    ix0, iy0, ix1, iy1 = max(sx0, ox0), max(sy0, oy0), min(sx1, ox1), min(sy1, oy1)
    area_s = abs((sx1 - sx0) * (sy1 - sy0))
    area_o = abs((ox1 - ox0) * (oy1 - oy0))
    area_i = abs(max(0, ix1 - ix0) * max(0, iy1 - iy0))
    iou = area_i / (area_s + area_o - area_i)
    touching = 0 < iou < 0.05

    if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
        p = 'surrounding'
    elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
        p = 'inside'
    # 60 degree intervals along each direction
    elif (5 * math.pi / 6 <= theta <= math.pi) or (-math.pi <= theta < -5 * math.pi / 6):
        p = 'left of'
    elif -2 * math.pi / 3 <= theta < -math.pi / 3:
        p = 'behind'
    elif -math.pi / 6 <= theta < math.pi / 6:
        p = 'right of'
    elif math.pi / 3 <= theta < 2 * math.pi / 3:
        p = 'in front of'
    # 30 degree intervals along each direction
    elif 2 * math.pi / 3 <= theta < 5 * math.pi / 6:
        p = 'left front of'
    elif math.pi / 6 <= theta < math.pi / 3:
        p = 'right front of'
    elif -5 * math.pi / 6 <= theta < -2 * math.pi / 3:
        p = 'left behind'
    elif -math.pi / 3 <= theta < -math.pi / 6:
        p = 'right behind'
    if p == None:
        ipdb.set_trace()
    return p, distance


def get_2d_bdbox(obj, is_rad = False):
    if is_rad:
        R = roty(obj['orientation'])
    else:
        R = roty(obj['orientation'] / 180 * math.pi)
    box_vertices = np.asarray([[-obj['length'] / 2, -obj['width'] / 2],
                               [obj['length'] / 2, -obj['width'] / 2],
                               [-obj['length'] / 2, obj['width'] / 2],
                               [obj['length'] / 2, obj['width'] / 2]])
    box_vertices = box_vertices @ R
    if 'left' in obj.keys():
        box_vertices += np.asarray([[obj['left'], obj['top']]])
    return box_vertices


def get_3d_bdbox(obj, is_rad = False):
    if is_rad:
        R = roty_3d(obj['orientation'])
    else:
        R = roty_3d(obj['orientation'] / 180 * math.pi)
    box_vertices = np.asarray([[-obj['length'] / 2, -obj['width'] / 2, -obj['height'] / 2],
                                [obj['length'] / 2, -obj['width'] / 2, -obj['height'] / 2],
                                [-obj['length'] / 2, obj['width'] / 2, -obj['height'] / 2],
                                [obj['length'] / 2, obj['width'] / 2, -obj['height'] / 2],
                                [-obj['length'] / 2, -obj['width'] / 2, obj['height'] / 2],
                                [obj['length'] / 2, -obj['width'] / 2, obj['height'] / 2],
                                [-obj['length'] / 2, obj['width'] / 2, obj['height'] / 2],
                                [obj['length'] / 2, obj['width'] / 2, obj['height'] / 2]])
    box_vertices = box_vertices @ R
    if 'left' in obj.keys():
       box_vertices += np.asarray([[obj['left'], obj['top'], obj['depth']]])

    return box_vertices


def get_start_end_node(o1, o2, objs):
    for i in range(len(objs)):
        if objs[i]['label'] == o1:
            start = i
        if objs[i]['label'] == o2:
            end = i
    if start is None or end is None:
        ipdb.set_trace()
    return start, end

def area_to_graph(objs, rel_positions, use_vector = False, encoder_model = None, is_gt = False):
    graph = nx.DiGraph()
    if use_vector:
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='D:/Code/LayoutGPT/models/open_clip_pytorch_model.bin', device=device)
        # clip_tokenizer = open_clip.get_tokenizer('ViT-H-14')
        device = encoder_model.device
        clip_model, clip_preprocess = encoder_model.clip_model, encoder_model.clip_preprocess
        clip_tokenizer = encoder_model.clip_tokenizer
        obj_img_dir = 'dataset/ATISS/data_3d_future/3D_future_img/'

        if is_gt:
            for nid in range(len(objs)):
                o = objs[nid]['label']
                size = [objs[nid]['length'], objs[nid]['width'], objs[nid]['height']]
                with torch.no_grad():
                    image = clip_preprocess(Image.open(obj_img_dir + objs[nid]['obj_jid']+".png")).unsqueeze(0).to(device)
                    image_features = clip_model.encode_image(image)
                graph.add_node(nid, tag=o, geometry=size, features = image_features, info = image)

            for eid in range(len(rel_positions)):
                rel_position = rel_positions[eid]
                o1 = rel_position[0]
                rl = rel_position[1]
                o2 = rel_position[2]
                e = get_start_end_node(o1, o2, objs)
                graph.add_edge(e[1], e[0], tag=rl)
        else:
            for nid in range(len(objs)):
                o = objs[nid]['label']
                size = [objs[nid]['length'], objs[nid]['width'], objs[nid]['height']]
                with torch.no_grad():
                    text_inputs = clip_tokenizer(objs[nid]['description']).to(device)
                    text_features = clip_model.encode_text(text_inputs)
                graph.add_node(nid, tag=o, geometry=size, features = text_features, info = text_inputs)

            for eid in range(len(rel_positions)):
                rel_position = rel_positions[eid]
                o1 = rel_position[0]
                rl = rel_position[1]
                o2 = rel_position[2]
                e = get_start_end_node(o1, o2, objs)
                graph.add_edge(e[1], e[0], tag=rl)
    else:
        for nid in range(len(objs)):
            o = objs[nid]['label']
            size = [objs[nid]['length'], objs[nid]['width'], objs[nid]['height']]
            graph.add_node(nid, tag=o, geometry=size)

        for eid in range(len(rel_positions)):
            rel_position = rel_positions[eid]
            o1 = rel_position[0]
            rl = rel_position[1]
            o2 = rel_position[2]
            e = get_start_end_node(o1, o2, objs)
            graph.add_edge(e[1], e[0], tag=rl)

    return graph