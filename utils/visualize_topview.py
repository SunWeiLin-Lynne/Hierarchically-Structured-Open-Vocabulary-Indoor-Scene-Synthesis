import math
import os
import os.path as op
import ipdb
import matplotlib.pyplot as plt
import numpy as np
import json
try:
    from utils.utils import OBJ_LIST, roty
except:
    from utils import OBJ_LIST, roty
import difflib
color_list = ['r', 'b', 'g', 'c', 'm', 'y']
area_color_list = {
    "dressing_area":"green",
    "sleeping_area":"red",
    "living_area":"blue",
    "storage_area":"yellow",
    "study_area":"purple"}
obj_color_list = {'dressing_table': 'cyan',
                  'double_bed': 'brown', 'kids_bed': 'tomato', 'single_bed': 'lightcoral',
                  'coffee_table': 'lightskyblue', 'tv_stand':'steelblue', 'sofa':'cornflowerblue', 'bookshelf':'sienna',
                  'cabinet':'gold', 'children_cabinet':'orange', 'wardrobe':'chocolate', 'shelf':'sandybrown',
                  'desk':'slateblue', 'table':'plum', 'armchair':'violet', 'ceiling_lamp':'coral', 'chair':'thistle', 'dressing_chair':'lightsage',
                  'floor_lamp':'slategray', 'nightstand':'darkred', 'pendant_lamp':'salmon', 'stool':'pink'}

root_dir = 'xxx'
output_dir = 'xxx'

topview_output_dir = 'xxx'

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def offset_from_center(x, y, centerX, centerY):
    return (x - centerX, y - centerY)


def rotate_point_around_center(theta, x, y, centerX, centerY):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    offset_x, offset_y = offset_from_center(x, y, centerX, centerY)

    x_new = cos_theta * offset_x - sin_theta * offset_y + centerX
    y_new = sin_theta * offset_x + cos_theta * offset_y + centerY

    return x_new, y_new


def rotate_around_center(angle, length, width):
    R = roty(angle / 180 * math.pi)
    box_vertices = np.asarray([[-length / 2, -width / 2],
                               [length / 2, -width / 2],
                               [-length/ 2, width / 2],
                               [length / 2, width / 2]])
    box_vertices = box_vertices @ R

    return abs(max(box_vertices[:,0])-min(box_vertices[:,0])), abs(max(box_vertices[:,1])-min(box_vertices[:,1]))


def draw_arrow(ax, angle, centerX, centerY, color):
    axis_size = 0.05  # meter:0.5; px:10
    if angle == 0:
        return ax.arrow(centerX, centerY, 0, axis_size, length_includes_head=False,
                 head_width=axis_size / 2, fc=color, ec=color)
    elif angle == 45:
        return ax.arrow(centerX, centerY, axis_size, -axis_size, length_includes_head=False,
                 head_width=axis_size / 2, fc=color, ec=color)
    elif angle == 90:
        return ax.arrow(centerX, centerY, axis_size, 0, length_includes_head=False,
                 head_width=axis_size / 2, fc=color, ec=color)
    elif angle == 180:
        return ax.arrow(centerX, centerY, 0, -axis_size, length_includes_head=False,
                 head_width=axis_size / 2, fc=color, ec=color)
    elif angle == 270 or angle == -90:
        return ax.arrow(centerX, centerY, -axis_size, 0, length_includes_head=False,
                 head_width=axis_size / 2, fc=color, ec=color)


def visualize_all_area_topview(hierarchy_data, interactive_num, val_id = None):
    if val_id is not None:
        topview_name = val_id.split('_')[-1]
    else:
        topview_name = 'topview'
    room_length = float(hierarchy_data['length'])
    room_width = float(hierarchy_data['width'])
    y_room = -room_width
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    plt.plot([0, room_length, room_length, 0, 0],
             [y_room + room_width, y_room + room_width, 0 + room_width, 0 + room_width,
              y_room + room_width], 'k')
    col = 0
    for area in hierarchy_data['children']:
        if 'is_delete' not in area.keys():
            label = area['label']
            centerX = area['bdb_x']
            centerY = -area['bdb_y']
            y_len = area['tight_width']
            x_len = area['tight_length']
            angle = area['orientation']
            center = (centerX, centerY)
            length = x_len
            width = y_len

            theta = (angle / 180) * math.pi


            top_left = np.array([center[0] - length / 2, center[1] + width / 2])
            top_right = np.array([center[0] + length / 2, center[1] + width / 2])
            bottom_left = np.array([center[0] - length / 2, center[1] - width / 2])
            bottom_right = np.array([center[0] + length / 2, center[1] - width / 2])

            top_left_rotated = np.array(
                rotate_point_around_center(theta, top_left[0], top_left[1], centerX, centerY))
            top_right_rotated = np.array(
                rotate_point_around_center(theta, top_right[0], top_right[1], centerX, centerY))
            bottom_left_rotated = np.array(
                rotate_point_around_center(theta, bottom_left[0], bottom_left[1], centerX, centerY))
            bottom_right_rotated = np.array(
                rotate_point_around_center(theta, bottom_right[0], bottom_right[1], centerX, centerY))

            plt.plot(
                [top_left_rotated[0], top_right_rotated[0], bottom_right_rotated[0],
                 bottom_left_rotated[0], top_left_rotated[0]],
                [top_left_rotated[1] + room_width, top_right_rotated[1] + room_width,
                 bottom_right_rotated[1] + room_width, bottom_left_rotated[1] + room_width,
                 top_left_rotated[1] + room_width], color_list[col])

            plt.text(top_left_rotated[0], top_left_rotated[1] + room_width, label, fontsize=12,
                     ha='center',
                     va='center', color=color_list[col])

            axis_size = 0.05  # meter:0.5; px:10
            if angle == 0:
                ax.arrow(centerX, room_width + centerY, 0, -axis_size, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 45:
                ax.arrow(centerX, room_width + centerY, axis_size, -axis_size, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 90:
                ax.arrow(centerX, room_width + centerY, axis_size, 0, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 180:
                ax.arrow(centerX, room_width + centerY, 0, axis_size, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 270 or angle == -90:
                ax.arrow(centerX, room_width + centerY, -axis_size, 0, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            # print(angle)
            col = (col + 1) % 6

            plt.axis('equal')
            plt.axis('off')


    # plt.show()
    os.makedirs(op.join(root_dir, 'topview'), exist_ok=True)
    if interactive_num != 0:
        plt.savefig(root_dir + '/topview/' + f"{topview_name}_area_{interactive_num}" + '.png', dpi=128, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(root_dir + '/topview/' + f"{topview_name}_area" + '.png', dpi=128, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_all_object_topview(hierarchy_data, object_list, interactive_num, val_id = None):
    if val_id is not None:
        topview_name = val_id.split('_')[-1]
    else:
        topview_name = 'topview'
    room_length = float(hierarchy_data['length'])
    room_width = float(hierarchy_data['width'])
    y_room = -room_width
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    plt.plot([0, room_length, room_length, 0, 0],
             [y_room + room_width, y_room + room_width, 0 + room_width, 0 + room_width,
              y_room + room_width], 'k')
    col = 0
    for object in object_list:
        if len(object) != 0:
            label = object[0] 
            centerX = object[1]['left']
            centerY = -object[1]['top']
            centerZ = object[1]['depth']
            z_len = object[1]['height']
            y_len = object[1]['width']
            x_len = object[1]['length']
            angle = object[1]['orientation']

            center = (centerX, centerY)
            length = x_len
            width = y_len
            theta = (angle / 180) * math.pi


            top_left = np.array([center[0] - length / 2, center[1] + width / 2])
            top_right = np.array([center[0] + length / 2, center[1] + width / 2])
            bottom_left = np.array([center[0] - length / 2, center[1] - width / 2])
            bottom_right = np.array([center[0] + length / 2, center[1] - width / 2])

            top_left_rotated = np.array(
                rotate_point_around_center(theta, top_left[0], top_left[1], centerX, centerY))
            top_right_rotated = np.array(
                rotate_point_around_center(theta, top_right[0], top_right[1], centerX, centerY))
            bottom_left_rotated = np.array(
                rotate_point_around_center(theta, bottom_left[0], bottom_left[1], centerX, centerY))
            bottom_right_rotated = np.array(
                rotate_point_around_center(theta, bottom_right[0], bottom_right[1], centerX, centerY))

            plt.plot(
                [top_left_rotated[0], top_right_rotated[0], bottom_right_rotated[0],
                 bottom_left_rotated[0], top_left_rotated[0]],
                [top_left_rotated[1] + room_width, top_right_rotated[1] + room_width,
                 bottom_right_rotated[1] + room_width, bottom_left_rotated[1] + room_width,
                 top_left_rotated[1] + room_width], color_list[col])

            plt.text(top_left_rotated[0], top_left_rotated[1] + room_width, label, fontsize=15,
                     ha='center',
                     va='center', color=color_list[col])

            axis_size = 0.05  # meter:0.5; px:10
            if angle == 0:
                ax.arrow(centerX, room_width + centerY, 0, -axis_size, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 45:
                ax.arrow(centerX, room_width + centerY, axis_size, -axis_size, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 90:
                ax.arrow(centerX, room_width + centerY, axis_size, 0, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 180:
                ax.arrow(centerX, room_width + centerY, 0, axis_size, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            elif angle == 270 or angle == -90:
                ax.arrow(centerX, room_width + centerY, -axis_size, 0, length_includes_head=False,
                         head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
            # print(angle)
            col = (col + 1) % 6

            plt.axis('equal')
            plt.axis('off')


    # plt.show()
    os.makedirs(op.join(root_dir, 'topview'), exist_ok=True)
    if interactive_num != 0:
        plt.savefig(root_dir + '/topview/' + f"{topview_name}_obj_{interactive_num}" + '.png', dpi=128, bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig(root_dir + '/topview/' + f"{topview_name}" + '.png', dpi=128, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_all_area_info_topview(hierarchy_data, all_area_prediction_list, interactive_num, val_id = None):
    if val_id is not None:
        topview_name = val_id.split('_')[-1]
    else:
        topview_name = 'topview'
    room_length = float(hierarchy_data['length'])
    room_width = float(hierarchy_data['width'])
    for area in all_area_prediction_list:
        area_label =area['area_label'].split(' ')[0]
        area_object_list = area['object_list']
        area_rel_pos_constraints_object_list = area['rel_pos_constraints']
        area_length = float(area['hierarchy_data']['length'])
        area_width = float(area['hierarchy_data']['width'])

        y_area = -area_width
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()
        plt.plot([0, area_length, area_length, 0, 0],
                 [y_area + area_width, y_area + area_width, 0 + area_width, 0 + area_width,
                  y_area + area_width], 'k')

        col = 0
        for object in area_object_list:
            if len(object) != 0:
                label = object[0]
                centerX = object[1]['left']
                centerY = -object[1]['top']
                centerZ = object[1]['depth']
                z_len = object[1]['height']
                y_len = object[1]['width']
                x_len = object[1]['length']
                angle = object[1]['orientation']

                center = (centerX, centerY)
                length = x_len
                width = y_len
                theta = (angle / 180) * math.pi

                top_left = np.array([center[0] - length / 2, center[1] + width / 2])
                top_right = np.array([center[0] + length / 2, center[1] + width / 2])
                bottom_left = np.array([center[0] - length / 2, center[1] - width / 2])
                bottom_right = np.array([center[0] + length / 2, center[1] - width / 2])

                top_left_rotated = np.array(
                    rotate_point_around_center(theta, top_left[0], top_left[1], centerX, centerY))
                top_right_rotated = np.array(
                    rotate_point_around_center(theta, top_right[0], top_right[1], centerX, centerY))
                bottom_left_rotated = np.array(
                    rotate_point_around_center(theta, bottom_left[0], bottom_left[1], centerX, centerY))
                bottom_right_rotated = np.array(
                    rotate_point_around_center(theta, bottom_right[0], bottom_right[1], centerX, centerY))

                plt.plot(
                    [top_left_rotated[0], top_right_rotated[0], bottom_right_rotated[0],
                     bottom_left_rotated[0], top_left_rotated[0]],
                    [top_left_rotated[1] + area_width, top_right_rotated[1] + area_width,
                     bottom_right_rotated[1] + area_width, bottom_left_rotated[1] + area_width,
                     top_left_rotated[1] + area_width], color_list[col])

                plt.text(top_left_rotated[0], top_left_rotated[1] + area_width, label, fontsize=15,
                         ha='center',
                         va='center', color=color_list[col])

                axis_size = 0.05  # meter:0.5; px:10
                if angle == 0:
                    ax.arrow(centerX, area_width + centerY, 0, -axis_size, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 45:
                    ax.arrow(centerX, area_width + centerY, axis_size, -axis_size, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 90:
                    ax.arrow(centerX, area_width + centerY, axis_size, 0, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 180:
                    ax.arrow(centerX, area_width + centerY, 0, axis_size, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 270 or angle == -90:
                    ax.arrow(centerX, area_width + centerY, -axis_size, 0, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                # print(angle)
                col = (col + 1) % 6

                plt.axis('equal')
                plt.axis('off')


        # plt.show()
        os.makedirs(op.join(root_dir, 'topview'), exist_ok=True)
        if interactive_num != 0:
            plt.savefig(root_dir + '/topview/' + f"{topview_name}_{area_label}_{interactive_num}" + '.png',
                        dpi=128, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(root_dir + '/topview/' + f"{topview_name}_{area_label}" + '.png',
                        dpi=128, bbox_inches='tight', pad_inches=0)
        plt.close()

        col = 0
        y_room = -room_width
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot()

        for object in area_rel_pos_constraints_object_list:
            if len(object) != 0:
                label = object[0]
                centerX = object[1]['left']
                centerY = -object[1]['top']
                centerZ = object[1]['depth']
                z_len = object[1]['height']
                y_len = object[1]['width']
                x_len = object[1]['length']
                angle = object[1]['orientation']

                center = (centerX, centerY)
                length = x_len
                width = y_len
                theta = (angle / 180) * math.pi

                top_left = np.array([center[0] - length / 2, center[1] + width / 2])
                top_right = np.array([center[0] + length / 2, center[1] + width / 2])
                bottom_left = np.array([center[0] - length / 2, center[1] - width / 2])
                bottom_right = np.array([center[0] + length / 2, center[1] - width / 2])

                top_left_rotated = np.array(
                    rotate_point_around_center(theta, top_left[0], top_left[1], centerX, centerY))
                top_right_rotated = np.array(
                    rotate_point_around_center(theta, top_right[0], top_right[1], centerX, centerY))
                bottom_left_rotated = np.array(
                    rotate_point_around_center(theta, bottom_left[0], bottom_left[1], centerX, centerY))
                bottom_right_rotated = np.array(
                    rotate_point_around_center(theta, bottom_right[0], bottom_right[1], centerX, centerY))

                plt.plot(
                    [top_left_rotated[0], top_right_rotated[0], bottom_right_rotated[0],
                     bottom_left_rotated[0], top_left_rotated[0]],
                    [top_left_rotated[1] + room_width, top_right_rotated[1] + room_width,
                     bottom_right_rotated[1] + room_width, bottom_left_rotated[1] + room_width,
                     top_left_rotated[1] + room_width], color_list[col])

                plt.text(top_left_rotated[0], top_left_rotated[1] + room_width, label, fontsize=15,
                         ha='center',
                         va='center', color=color_list[col])

                axis_size = 0.05  # meter:0.5; px:10
                if angle == 0:
                    ax.arrow(centerX, room_width + centerY, 0, -axis_size, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 45:
                    ax.arrow(centerX, room_width + centerY, axis_size, 10, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 90:
                    ax.arrow(centerX, room_width + centerY, axis_size, 0, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 180:
                    ax.arrow(centerX, room_width + centerY, 0, axis_size, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                elif angle == 270 or angle == -90:
                    ax.arrow(centerX, room_width + centerY, -axis_size, 0, length_includes_head=False,
                             head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                # print(angle)
                col = (col + 1) % 6

                plt.axis('equal')
                plt.axis('off') 

        # plt.show()
        os.makedirs(op.join(root_dir, 'topview'), exist_ok=True)
        if interactive_num != 0:
            plt.savefig(
                root_dir + '/topview/' + f"{topview_name}_rel_pos_constraints_{area_label}_{interactive_num}" + '.png', dpi=128, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(
                root_dir + '/topview/' + f"{topview_name}_rel_pos_constraints_{area_label}" + '.png', dpi=128, bbox_inches='tight', pad_inches=0)
        plt.close()


def visualize_area_obj_topview(area_label, area_l, area_w, object_list, name = '', type = 'one-time', interactive_num = 0):
    fig = plt.figure(num=1, figsize=(8, 8)) 
    ax = fig.add_subplot(1, 1, 1)
    ax.add_patch(plt.Rectangle(xy=(0, 0), width=area_l, height=area_w, angle=0, linewidth=1, fill=False, color=area_color_list[area_label]))
    ax.text(0, 0, area_label, bbox={'facecolor': f'{area_color_list[area_label]}', 'alpha':0.5})

    for object in object_list:
        if len(object) != 0:
            label = object['label']
            similar_label = difflib.get_close_matches(label, OBJ_LIST, cutoff=0.0)[0]
            centerX = object['bdb_x']
            centerY = object['bdb_y']
            length = object['bdb_l']
            width = object['bdb_w']
            angle = object['bdb_ori']
            length_rotated, width_rotated = rotate_around_center(angle, length, width)

            ax.add_patch(plt.Rectangle(xy=(centerX-length_rotated/2, centerY-width_rotated/2), width=length_rotated, height=width_rotated, fill=False, edgecolor=obj_color_list[similar_label]))
            ax.text(centerX-length_rotated/2, centerY-width_rotated/2, label, bbox={'facecolor': f'{obj_color_list[similar_label]}', 'alpha': 0.5})

            draw_arrow(ax, angle, centerX, centerY, obj_color_list[similar_label])


    plt.axis('equal')
    plt.gca().invert_yaxis()
    # plt.show()
    path_name = output_dir + f"/{type}_{interactive_num}/"
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    plt.savefig(path_name + f'{name}.png', dpi=128,bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_area_topview(room_l, room_w, area_list, name = '', type = 'one-time', interactive_num = 0):
    fig = plt.figure(num=1, figsize=(8, 8)) 
    ax = fig.add_subplot(1, 1, 1)
    ax.add_patch(plt.Rectangle(xy=(0, 0), width=room_l, height=room_w, angle=0, linewidth=1, fill=False, color='k'))

    for area in area_list:
        if len(area) != 0:
            label = area['label'] 
            centerX = area['bdb_x']
            centerY = area['bdb_y']
            length = area['bdb_l']
            width = area['bdb_w']
            angle = area['bdb_ori']

            ax.add_patch(plt.Rectangle(xy=(centerX-length/2, centerY-width/2), width=length, height=width, fill=False, edgecolor=area_color_list[label]))
            ax.text(centerX-length/2, centerY-width/2, label, bbox={'facecolor': f'{area_color_list[label]}', 'alpha': 0.5})


            draw_arrow(ax, angle, centerX, centerY, area_color_list[label])


    plt.axis('equal')
    plt.gca().invert_yaxis()
    path_name = output_dir + f"/{type}_{interactive_num}/"
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    plt.savefig(path_name + f'{name}.png', dpi=128, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_all_area_obj_topview(room_l, room_w, data, name='', type = 'one-time', interactive_num = 0):
    fig = plt.figure(num=1, figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.add_patch(plt.Rectangle(xy=(0, 0), width=room_l, height=room_w, angle=0, linewidth=1, fill=False, color='k'))
    for area in data['children']:
        if len(area) != 0:
            label = area['label']
            centerX = area['bdb_x']
            centerY = area['bdb_y']
            length = area['bdb_l']
            width = area['bdb_w']
            angle = area['orientation']
            ax.add_patch(plt.Rectangle(xy=(centerX-length/2, centerY-width/2), width=length, height=width, fill=False, edgecolor=area_color_list[label]))

            if 'children' in area.keys():
                func_obj = area['children']
                func_label = func_obj['label']
                func_similar_label = difflib.get_close_matches(func_label, OBJ_LIST, cutoff=0.0)[0]
                func_centerX = func_obj['left']
                func_centerY = func_obj['top']
                func_length = func_obj['length']
                func_width = func_obj['width']
                func_angle = func_obj['orientation']
                func_length_rotated, func_width_rotated = rotate_around_center(func_angle, func_length, func_width)
                # print(func_length, func_width, func_angle, func_length_rotated, func_width_rotated)
                ax.add_patch(plt.Rectangle(xy=(func_centerX - func_length_rotated/2, func_centerY - func_width_rotated/2), width=func_length_rotated, height=func_width_rotated,
                                          fill=False, edgecolor=obj_color_list[func_similar_label]))
                ax.text(func_centerX - func_length_rotated / 2, func_centerY - func_width_rotated / 2, func_label,
                        bbox={'facecolor': f'{obj_color_list[func_similar_label]}', 'alpha': 0.5})

                draw_arrow(ax, func_angle, func_centerX, func_centerY, obj_color_list[func_similar_label])

                if 'children' in func_obj.keys():
                    for obj in func_obj['children']:
                        obj_label = obj['label']
                        obj_similar_label = difflib.get_close_matches(obj_label, OBJ_LIST, cutoff=0.0)[0]
                        obj_centerX = obj['left']
                        obj_centerY = obj['top']
                        obj_length = obj['length']
                        obj_width = obj['width']
                        obj_angle = obj['orientation']
                        obj_length_rotated, obj_width_rotated = rotate_around_center(obj_angle, obj_length, obj_width)
                        # print(obj_length, obj_width, obj_angle, obj_length_rotated, obj_width_rotated)
                        ax.add_patch(plt.Rectangle(xy=(obj_centerX - obj_length_rotated / 2, obj_centerY - obj_width_rotated / 2),
                                                   width=obj_length_rotated, height=obj_width_rotated, fill=False,
                                                   edgecolor=obj_color_list[obj_similar_label]))
                        ax.text(obj_centerX - obj_length_rotated / 2, obj_centerY - obj_width_rotated / 2, obj_label,
                                bbox={'facecolor': f'{obj_color_list[obj_similar_label]}', 'alpha': 0.5})
                        draw_arrow(ax, obj_angle, obj_centerX, obj_centerY, obj_color_list[obj_similar_label])
    plt.axis('equal')
    plt.gca().invert_yaxis()
    path_name = output_dir + f"/{type}_{interactive_num}/"
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    plt.savefig(path_name + f'{name}.png', dpi=128, bbox_inches='tight', pad_inches=0)
    plt.close()


def visualize_topview_from_json(json_files):
    for json_file in json_files:
        topview_json = load_json(json_file)
        for room in topview_json:
            # for i in range(len(room['object_list'])):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot()
            try:
                topview_name = room['query_id'].split('_')[-1]
            except:
                topview_name = str(room['query_id'])
            # topview_name = room['query_id'].split('_')[-1]+'.'+room['area_label']
            col = 0
            try:
               room_length_width = room['prompt'].split('Room Size: ')[1].split(' meters')[0]
               room_length = float(room_length_width.split('x')[0])
               room_width = float(room_length_width.split('x')[1])
            except:
                try:
                    room_length = float(room['room_length'])
                    room_width = float(room['room_width'])
                except:
                    try:
                        room_length = room['hierarchy_data']['children'][0]['length']
                        room_width = room['hierarchy_data']['children'][0]['width']
                    except:
                       room_length_width = room['prompt'].split('Room Size: ')[1]
                       room_length = float(room_length_width.split()[2].split('px')[0])
                       room_width = float(room_length_width.split()[5].split('px')[0])


            y_room = -room_width
            # print(tree_name)
            plt.plot([0, room_length, room_length, 0, 0],
                     [y_room + room_width, y_room + room_width, 0 + room_width, 0 + room_width,
                      y_room + room_width], 'k')
            # ipdb.set_trace()
            if room['object_list'] is not None:

                for object in room['object_list']:
                    if len(object) != 0:
                        label = object[0] 
                        centerX = object[1]['left']
                        centerY = -object[1]['top']
                        centerZ = object[1]['depth']
                        z_len = object[1]['height']
                        y_len = object[1]['width']
                        x_len = object[1]['length']
                        angle = object[1]['orientation']
                        center = (centerX, centerY)
                        length = x_len
                        width = y_len
                        theta = (angle / 180) * math.pi

                        top_left = np.array([center[0] - length / 2, center[1] + width / 2])
                        top_right = np.array([center[0] + length / 2, center[1] + width / 2])
                        bottom_left = np.array([center[0] - length / 2, center[1] - width / 2])
                        bottom_right = np.array([center[0] + length / 2, center[1] - width / 2])

                        top_left_rotated = np.array(
                            rotate_point_around_center(theta, top_left[0], top_left[1], centerX, centerY))
                        top_right_rotated = np.array(
                            rotate_point_around_center(theta, top_right[0], top_right[1], centerX, centerY))
                        bottom_left_rotated = np.array(
                            rotate_point_around_center(theta, bottom_left[0], bottom_left[1], centerX, centerY))
                        bottom_right_rotated = np.array(
                            rotate_point_around_center(theta, bottom_right[0], bottom_right[1], centerX, centerY))

                        plt.plot(
                            [top_left_rotated[0], top_right_rotated[0], bottom_right_rotated[0],
                             bottom_left_rotated[0], top_left_rotated[0]],
                            [top_left_rotated[1] + room_width, top_right_rotated[1] + room_width,
                             bottom_right_rotated[1] + room_width, bottom_left_rotated[1] + room_width,
                             top_left_rotated[1] + room_width], color_list[col])

                        plt.text(top_left_rotated[0], top_left_rotated[1] + room_width, label, fontsize=12,
                                 ha='center',
                                 va='center', color=color_list[col])

                        axis_size = 0.05  # meter:0.5; px:10
                        if angle == 0:
                            ax.arrow(centerX, room_width + centerY, 0, -axis_size, length_includes_head=False,
                                     head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                        elif angle == 45:
                            ax.arrow(centerX, room_width + centerY, axis_size, -axis_size, length_includes_head=False,
                                     head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                        elif angle == 90:
                            ax.arrow(centerX, room_width + centerY, axis_size, 0, length_includes_head=False,
                                     head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                        elif angle == 180:
                            ax.arrow(centerX, room_width + centerY, 0, axis_size, length_includes_head=False,
                                     head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                        elif angle == 270 or angle == -90:
                            ax.arrow(centerX, room_width + centerY, -axis_size, 0, length_includes_head=False,
                                     head_width=axis_size / 2, fc=color_list[col], ec=color_list[col])
                        # print(angle)
                        col = (col + 1) % 6

                        plt.axis('equal')

            os.makedirs(op.join(topview_output_dir, 'topview'), exist_ok=True)
            plt.savefig(topview_output_dir + '/topview/' + topview_name + '.png', dpi=128,
                        bbox_inches='tight', pad_inches=0)
            plt.close()
        #     break
        # break


if __name__ == '__main__':
    room_json_files = []
    area_json_files = []
    visualize_topview_from_json(room_json_files)
