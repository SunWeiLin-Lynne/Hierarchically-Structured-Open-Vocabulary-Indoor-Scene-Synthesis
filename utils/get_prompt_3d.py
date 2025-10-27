import json
import math
import random
import num2words
import ipdb
import numpy as np

from utils.utils import dict_bbox_to_vec, compute_rel, get_room_info, get_layout_info, get_3d_bdbox
# from utils import dict_bbox_to_vec, compute_rel, get_room_info, get_layout_info, get_3d_bdbox

def convert_to_ordinal(number):
    ordinal = num2words.num2words(number, ordinal=True)
    return ordinal


def plural(word):
    if word[-1] == 'y':
        return word[:-1]+'ies'
    elif word[-1] in 'sx' or word[-2:] in ['sh','ch']:
        return word+'es'
    elif word[-2:-1] == 'an':
        return word[-2:]+'en'
    else:
        return word+'s'


def get_object_relations(object_information, room_max_length, room_max_width):
    relations = []

    for object_name in object_information:
        this_box_postion = np.array([object_information[object_name]['left'],object_information[object_name]['top'],object_information[object_name]['depth']])
        this_box_sizes = np.array([object_information[object_name]['length'],object_information[object_name]['width'],object_information[object_name]['height']])
        this_box = {'min': list(this_box_postion - this_box_sizes*0.5), 'max': list(this_box_postion + this_box_sizes*0.5)}

        # only backward relations
        choices = [other_object_name for other_object_name in object_information if other_object_name != object_name]

        for other_object_name in choices:
            prev_box_postion = np.array([object_information[other_object_name]['left'],object_information[other_object_name]['top'],object_information[other_object_name]['depth']])
            prev_box_sizes = np.array([object_information[other_object_name]['length'],object_information[other_object_name]['width'],object_information[other_object_name]['height']])
            prev_box = {'min': list(prev_box_postion - prev_box_sizes*0.5), 'max': list(prev_box_postion + prev_box_sizes*0.5)}
            box1 = dict_bbox_to_vec(this_box)
            box2 = dict_bbox_to_vec(prev_box)

            relation_str, distance = compute_rel(box1, box2, object_information[other_object_name]['orientation']/ 180 * math.pi, room_max_length, room_max_width)
            # print(relation_str, object_name, other_object_name)
            if relation_str is not None:
                relation = (object_name, relation_str, other_object_name, distance)
                relations.append(relation)
    return relations

def get_prompt(data, unit):
    '''
    prompt_format:
    This is a ROOM TYPE with a length of ROOM MAX LENGTH and a width of ROOM MAX WIDTH.
    The ROOM TYPE have NUM object1, NUM object2 and NUM object3.
    The first object1 RELATIONSHIP the object2.
    The second object1 RELATIONSHIP the object2.
    '''
    # new_data = {}
    prompts = {}
    for id in data:
        condition = data[id][0]
        layout = data[id][1]
        room_type, room_max_length, room_max_width = get_room_info(condition, unit=unit)
        layout_information_list = get_layout_info(layout, unit=unit)
        if unit in ['px']:
            first_sentence = 'This is a ' + room_type + ' with a length of ' + room_max_length + 'px and a width of ' + room_max_width + 'px.'
        elif unit in ['']:
            first_sentence = 'This is a ' + room_type + ' with a length of ' + room_max_length + ' and a width of ' + room_max_width + '.'
        elif unit in ['m']:
            first_sentence = 'This is a ' + room_type + ' with a length of ' + room_max_length + ' meter and a width of ' + room_max_width + ' meter.'

        layout_information = {}
        for object_info_list in layout_information_list:
            object_name, object_length, object_width, object_height, object_left, object_top, object_depth, object_orientation = object_info_list
            object_info_dir = {'length': float(object_length),
                               'width': float(object_width),
                               'height': float(object_height),
                               'left': float(object_left),
                               'top': float(object_top),
                               'depth': float(object_depth),
                               'orientation': int(object_orientation)}
            if object_name in layout_information.keys():
                o_id = len(layout_information[object_name]) + 1
                layout_information[object_name][o_id] = object_info_dir
            else:
                layout_information[object_name] = {}
                layout_information[object_name][1] = object_info_dir

        second_sentence = 'The '+ room_type + ' '
        for i, object_name in enumerate(layout_information):
            num = len(layout_information[object_name])
            num_to_words = num2words.num2words(num)
            if num >1:
                object_name = plural(object_name)
            if i == 0:
                if num > 1:
                    second_sentence += 'have '+ num_to_words + ' ' + object_name + ', '
                else:
                    second_sentence += 'has ' + num_to_words + ' ' + object_name + ', '
            elif i == len(layout_information)-1:
                second_sentence += 'and ' + num_to_words + ' ' + object_name + '.'
            else:
                second_sentence += num_to_words + ' ' + object_name + ', '

        object_information = {}
        for object_name in layout_information:
            for i in layout_information[object_name]:
                object_information[object_name+'*'+str(i)] = layout_information[object_name][i]
        object_relations = get_object_relations(object_information, room_max_length, room_max_width)

        possible_relations = random.sample(object_relations, 2) if len(object_relations)>=2 else object_relations

        third_sentence = ''
        for relation in possible_relations:
            (o_n1, rel, o_n2, d) = relation
            o1 = o_n1.split('*')[0]
            o2 = o_n2.split('*')[0]
            try:
               n1 = int(o_n1.split('*')[1])
               n2 = int(o_n2.split('*')[1])
            except:
                ipdb.set_trace()
            if len(layout_information[o1])  > 1:
                o1 = convert_to_ordinal(n1) + ' ' + o1
            if len(layout_information[o2]) > 1:
                o2 = convert_to_ordinal(n2) + ' ' + o2
            if 'touching' in rel:
                third_sentence += F'The {o1} is next to the {o2}'
            elif rel in ('left of', 'right of'):
                third_sentence += f'The {o1} is to the {rel} the {o2}'
            elif rel in ('surrounding', 'inside', 'behind', 'in front of', 'on', 'above', 'in the left front of', 'in the left behind of', 'in the right front of', 'in the right behind of',):
                third_sentence += F'The {o1} is {rel} the {o2}'
            third_sentence += '.'
        prompt = 'Prompt:\n'+first_sentence+'\n'+second_sentence+'\n'+third_sentence+'\n'

        prompts[id] = prompt
        # print(prompt)
    return prompts


def add_prompt(data, unit):
    '''
    prompt_format:
    This is a ROOM TYPE with a length of ROOM MAX LENGTH and a width of ROOM MAX WIDTH.
    The ROOM TYPE have NUM object1, NUM object2 and NUM object3.
    The first object1 RELATIONSHIP the object2.
    The second object1 RELATIONSHIP the object2.
    '''
    new_data = {}
    prompts = get_prompt(data, unit)
    for id in data:
        condition = data[id][0]
        layout = data[id][1]
        prompt = prompts[id]
        new_id_data = [prompt, condition, layout]
        new_data[id] = new_id_data
        # prompts[id] = prompt
        # print(prompt)
    return new_data, prompts


if __name__ == "__main__":
    with open(f"dataset/3D/bedroom/bedroom.val.json", "r") as file:
        val_data = json.load(file)
    ipdb.set_trace()
    val_data, prompt = add_prompt(val_data)
    ipdb.set_trace()
