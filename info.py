AREA_FUNCTION_OBJ_FROM_BEDROOM = {
"dressing_area":['dressing_table'],
"sleeping_area":['double_bed', 'kids_bed', 'single_bed'],
"living_area":['coffee_table', 'sofa', 'table'],
"storage_area":['bookshelf', 'cabinet', 'children_cabinet', 'wardrobe', 'shelf'],
"study_area":['desk'],
}
OTHER_OBJ_FROM_BEDROOM = 'armchair, ceiling_lamp, chair, dressing_chair, floor_lamp, nightstand, pendant_lamp, stool, tv_stand, bookshelf, cabinet, children_cabinet, wardrobe, shelf'


AREA_FUNCTION_OBJ_FROM_LIVINGROOM = {
    "living_area":['l_shaped_sofa', 'chaise_longue_sofa', 'lazy_sofa', 'loveseat_sofa', 'multi_seat_sofa'],
    "tv_area":['tv_stand'],
    "dining_area":['dining_table', 'wine_cabinet'],
    "storage_area":['cabinet', 'wardrobe', 'shelf'],
    "office_area":['bookshelf', 'desk'],
    "rest_area":['lounge_chair', 'round_end_table']
}
OTHER_OBJ_FROM_LIVINGROOM = 'armchair, ceiling_lamp, chinese_chair, console_table, corner_side_table, pendant_lamp, stool, coffee_table, dining_chair'

COLOR = {
"dressing_area":"green",
"sleeping_area":"red",
"living_area":"blue",
"tv_area":"gold",
"storage_area":"yellow",
"study_area":"black",
"dining_area":"pink",
"office_area":"gray",
"rest_area":"purple",
"other_area":"orange"}


EXMAPLE_FOR_BEDROOM_REL_OPTIM = f"Condition:\n" \
                     f"Room Type: bedroom\n" \
                     f"Room Size: 3.66 x 3.60 meters\n" \
                     f"Layout:\n" \
                     f"sleeping_area 1 | 3.00 x 2.20 meters:\n" \
                     f"double_bed 1 | a sleek and comfortable queen-sized bed with a modern design | 1.80 x 2.00 x 0.94 meters\n" \
                     f"nightstand 1 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | right behind, double_bed 1\n" \
                     f"nightstand 2 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | left behind, double_bed 1\n" \
                     f"pendant_lamp 1 | a pendant lamp with a modern design | 0.40 x 0.38 x 0.26 meters | above, double_bed 1\n\n"\
                     f"storage_area 1 | 1.85 x 0.50 meters:\n" \
                     f"wardrobe 1 | a high-class minimalist wardrobe | 1.72 x 0.45 x 2.10 meters\n\n"

EXAMPLE_FOR_INTERACTIVE_GENERATION = f"Requirement:\n" \
                     f"I have a bedroom that is about 3.66 x 3.60 meters. I want to design it as a modern bedroom.\n" \
                     f"Layout:\n" \
                     f"sleeping_area 1 | 3.00 x 2.20 meters:\n" \
                     f"double_bed 1 | a sleek and comfortable queen-sized bed with a modern design | 1.80 x 2.00 x 0.94 meters\n" \
                     f"nightstand 1 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | right behind, double_bed 1\n" \
                     f"nightstand 2 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | left behind, double_bed 1\n" \
                     f"pendant_lamp 1 | a pendant lamp with a modern design | 0.40 x 0.38 x 0.26 meters | above, double_bed 1\n\n"\
                     f"storage_area 1 | 1.85 x 0.50 meters:\n" \
                     f"wardrobe 1 | a high-class minimalist wardrobe | 1.72 x 0.45 x 2.10 meters\n\n"



EXMAPLE_FOR_BEDROOM_REL = f"Condition:\n" \
                     f"Room Type: bedroom\n" \
                     f"Room Size: 3.66 x 3.60 meters\n" \
                     f"Layout:\n" \
                     f"sleeping_area 1 | 3.00 x 2.20 meters:\n" \
                     f"double_bed 1 | a sleek and comfortable queen-sized bed with a modern design | 1.80 x 2.00 x 0.94 meters | [1.93, 1.00, 0.47] | 0\n" \
                     f"nightstand 1 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | right behind, double_bed 1 | 0\n" \
                     f"nightstand 2 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | left behind, double_bed 1 | 0\n" \
                     f"pendant_lamp 1 | a pendant lamp with a modern design | 0.40 x 0.38 x 0.26 meters | above, double_bed 1 | 0\n\n"\
                     f"storage_area 1 | 1.85 x 0.50 meters:\n" \
                     f"wardrobe 1 | a high-class minimalist wardrobe | 1.72 x 0.45 x 2.10 meters | [0.86, 3.38, 1.05] | 180\n\n"

EXMAPLE_FOR_BEDROOM = f"Condition:\n" \
                     f"Room Type: bedroom\n" \
                     f"Room Size: 3.66 x 3.60 meters\n" \
                     f"Layout:\n" \
                     f"sleeping_area 1 | 3.00 x 2.20 meters:\n" \
                     f"double_bed 1 | a sleek and comfortable queen-sized bed with a modern design | 1.80 x 2.00 x 0.94 meters | [1.93, 1.00, 0.47] | 0\n" \
                     f"nightstand 1 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | [3.06, 0.21, 0.25] | 0\n" \
                     f"nightstand 2 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | [0.80, 0.21, 0.25] | 0\n" \
                     f"pendant_lamp 1 | a pendant lamp with a modern design | 0.40 x 0.38 x 0.26 meters | [1.93, 1.00, 2.165] | 0\n\n"\
                     f"storage_area 1 | 1.85 x 0.50 meters:\n" \
                     f"wardrobe 1 | a high-class minimalist wardrobe | 1.72 x 0.45 x 2.10 meters | [0.86, 3.38, 1.05] | 180\n\n"

BASELINE_EXMAPLE_FOR_BEDROOM = f"Condition:\n" \
                     f"Room Type: bedroom\n" \
                     f"Room Size: max length 3.66m, max width 3.60m\n" \
                     f"Layout:\n" \
                     f"double_bed 1 | a sleek and comfortable queen-sized bed with a modern design | 1.80 x 2.00 x 0.94 meters | [1.93, 1.00, 0.47] | 0\n" \
                     f"wardrobe 1 | a high-class minimalist wardrobe | 1.72 x 0.45 x 2.10 meters | [0.86, 3.38, 1.05] | 180\n"\
                     f"nightstand 1 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | [3.06, 0.21, 0.25] | 0\n" \
                     f"nightstand 2 | a delicate nightstand | 0.45 x 0.40 x 0.50 meters | [0.80, 0.21, 0.25] | 0\n" \
                     f"pendant_lamp 1 | a pendant lamp with a modern design | 0.40 x 0.38 x 0.26 meters | [1.93, 1.00, 2.165] | 0\n\n"\


BASELINE_EXMAPLE_FOR_BEDROOM_JSON = f"Condition:\n" \
                     f"Room Type: bedroom\n" \
                     f"Room Size: max length 3.66m, max width 3.60m\n" \
                     f"Layout:\n" \
                     f"double_bed {{length: 1.80m; width: 2.00m; height: 0.94m; orientation: 0 degrees; left: 1.93m; top: 1.00m; depth: 0.47m;}}\n" \
                     f"wardrobe {{length: 1.72m; width: 0.45m; height: 2.10m; orientation: 180 degrees; left: 0.86m; top: 3.38m; depth: 1.05;}}\n" \
                     f"nightstand {{length: 0.45m; width: 0.40m; height: 0.50m; orientation: 0 degrees; left: 3.06m; top: 0.21m; depth: 0.25m;}}\n" \
                     f"nightstand {{length: 0.45m; width: 0.40m; height: 0.50m; orientation: 0 degrees; left: 0.80m; top: 0.21m; depth: 0.25m;}}\n" \
                     f"pendant_lamp {{length: 0.40m; width: 0.38m; height: 0.26m; orientation: 0 degrees; left: 1.93m; top: 1.00m; depth: 2.165m;}}" \



EXMAPLE_FOR_LIVINGROOM_REL_OPTIM = f"Condition:\n"\
                      f"Room Type: living room\n" \
                      f"Room Size: 3.61 x 5.25 meters\n"\
                      f"Layout:\n"\
                      f"living_area 1 | 2 x 1.5 meters:\n"\
                      f"l_shaped_sofa 1 | an elegant, modern L-shaped sofa with plush cushions and a sleek design | 1.95 x 1.22 x 0.84 meters\n\n" \
                      f"tv_area 1 | 1.5 x 0.5 meters:\n" \
                      f"tv_stand 1 | a minimalist tv stand | 1.4 x 0.45 x 0.44 meters\n\n"\
                      f"dining_area 1 | 2.11 x 2.00 meters:\n"\
                      f"dining_table 1 | a wooden dining table with clean lines and a minimalist design | 1.20 x 0.75 x 0.76 meters\n"\
                      f"dining_chair 1 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | in front of, dining_table 1\n"\
                      f"dining_chair 2 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | behind, dining_table 1\n" \
                      f"dining_chair 3 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | left front of, dining_table 1\n" \
                      f"dining_chair 4 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | left behind, dining_table 1\n" \
                      f"dining_chair 5 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | right front of, dining_table 1\n" \
                      f"dining_chair 6 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | right behind, dining_table 1\n\n"\


EXMAPLE_FOR_LIVINGROOM_REL = f"Condition:\n"\
                      f"Room Type: living room\n" \
                      f"Room Size: 3.61 x 5.25 meters\n"\
                      f"Layout:\n"\
                      f"living_area 1 | 2 x 1.5 meters:\n"\
                      f"l_shaped_sofa 1 | an elegant, modern L-shaped sofa with plush cushions and a sleek design | 1.95 x 1.22 x 0.84 meters | [3.00, 0.975, 0.42] | 270\n\n" \
                      f"tv_area 1 | 1.5 x 0.5 meters:\n" \
                      f"tv_stand 1 | a minimalist tv stand | 1.4 x 0.45 x 0.44 meters | [0.225, 0.975, 0.22] | 90\n\n"\
                      f"dining_area 1 | 2.11 x 2.00 meters:\n"\
                      f"dining_table 1 | a wooden dining table with clean lines and a minimalist design | 1.20 x 0.75 x 0.76 meters | [2.54, 4.25, 0.38] | 180\n"\
                      f"dining_chair 1 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | in front of, dining_table 1 | 0\n"\
                      f"dining_chair 2 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | behind, dining_table 1 | 180\n" \
                      f"dining_chair 3 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | left front of, dining_table 1 | 0\n" \
                      f"dining_chair 4 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | left behind, dining_table 1 | 180\n" \
                      f"dining_chair 5 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | right front of, dining_table 1 | 0\n" \
                      f"dining_chair 6 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | right behind, dining_table 1 | 180\n\n"\


EXMAPLE_FOR_LIVINGROOM = f"Condition:\n"\
                      f"Room Type: living room\n" \
                      f"Room Size: 3.61 x 5.25 meters\n"\
                      f"Layout:\n"\
                      f"living_area 1 | 2 x 1.5 meters:\n"\
                      f"l_shaped_sofa 1 | an elegant, modern L-shaped sofa with plush cushions and a sleek design | 1.95 x 1.22 x 0.84 meters | [3.00, 0.975, 0.42] | 270\n\n" \
                      f"tv_area 1 | 1.5 x 0.5 meters:\n" \
                      f"tv_stand 1 | a minimalist tv stand | 1.4 x 0.45 x 0.44 meters | [0.225, 0.975, 0.22] | 90\n\n"\
                      f"dining_area 1 | 2.11 x 2.00 meters:\n"\
                      f"dining_table 1 | a wooden dining table with clean lines and a minimalist design | 1.20 x 0.75 x 0.76 meters | [2.54, 4.25, 0.38] | 180\n"\
                      f"dining_chair 1 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.59, 3.60, 0.40] | 0\n"\
                      f"dining_chair 2 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.59, 4.90, 0.40] | 180\n" \
                      f"dining_chair 3 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [3.10, 3.60, 0.40] | 0\n"\
                      f"dining_chair 4 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [3.10, 4.90, 0.40] | 180\n" \
                      f"dining_chair 5 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.08, 3.60, 0.40] | 0\n" \
                      f"dining_chair 6 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.08, 4.90, 0.40] | 180\n\n" \


BASELINE_EXMAPLE_FOR_LIVINGROOM = f"Condition:\n" \
                     f"Room Type: living room\n" \
                     f"Room Size: max length 3.61m, max width 5.25m\n" \
                     f"Layout:\n" \
                     f"l_shaped_sofa 1 | an elegant, modern L-shaped sofa with plush cushions and a sleek design | 1.95 x 1.22 x 0.84 meters | [3.00, 0.975, 0.42] | 270\n" \
                     f"tv_stand 1 | a minimalist tv stand | 1.4 x 0.45 x 0.44 meters | [0.225, 0.975, 0.22] | 90\n"\
                     f"dining_table 1 | a wooden dining table with clean lines and a minimalist design | 1.20 x 0.75 x 0.76 meters | [2.54, 4.25, 0.38] | 180\n"\
                     f"dining_chair 1 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.59, 3.60, 0.40] | 0\n"\
                     f"dining_chair 2 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.59, 4.90, 0.40] | 180\n" \
                     f"dining_chair 3 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [3.10, 3.60, 0.40] | 0\n"\
                     f"dining_chair 4 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [3.10, 4.90, 0.40] | 180\n" \
                     f"dining_chair 5 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.08, 3.60, 0.40] | 0\n" \
                     f"dining_chair 6 | a wooden dining_chair | 0.40 x 0.38 x 0.80 meters | [2.08, 4.90, 0.40] | 180\n\n" \

BASELINE_EXMAPLE_FOR_LIVINGROOM_JSON = f"Condition:\n"\
                      f"Room Type: living room\n" \
                      f"Room Size: max length 3.61m, max width 5.25m\n"\
                      f"Layout:\n"\
                      f"l_shaped_sofa {{length: 1.95m; width: 1.22m; height: 0.84m; orientation: 270 degrees; left: 3.00m; top: 0.975m; depth: 0.42m;}}\n" \
                      f"tv_stand {{length: 1.40m; width: 0.45m; height: 0.44m; orientation: 90 degrees; left: 0.225m; top: 0.975m; depth: 0.22m;}}\n" \
                      f"dining_table {{length: 1.20m; width: 0.75m; height: 0.76m; orientation: 180 degrees; left: 2.54m; top: 4.25m; depth: 0.38m;}}\n" \
                      f"dining_chair {{length: 0.40m; width: 0.38m; height: 0.80m; orientation: 0 degrees; left: 2.59m; top: 3.60m; depth: 0.40m;}}\n" \
                      f"dining_chair {{length: 0.40m; width: 0.38m; height: 0.80m; orientation: 180 degrees; left: 2.59m; top: 4.90m; depth: 0.40m;}}\n" \
                      f"dining_chair {{length: 0.40m; width: 0.38m; height: 0.80m; orientation: 0 degrees; left: 3.10m; top: 3.60m; depth: 0.40m;}}\n" \
                      f"dining_chair {{length: 0.40m; width: 0.38m; height: 0.80m; orientation: 180 degrees; left: 3.10m; top: 4.90m; depth: 0.40m;}}\n" \
                      f"dining_chair {{length: 0.40m; width: 0.38m; height: 0.80m; orientation: 0 degrees; left: 2.08m; top: 3.60m; depth: 0.40m;}}\n"\
                      f"dining_chair {{length: 0.40m; width: 0.38m; height: 0.80m; orientation: 180 degrees; left: 2.08m; top: 4.90m; depth: 0.40m;}}" \


HIER_AREA_FUNCTION_OBJ_FROM_BEDROOM = {
"dressing_area":['dressing_table'],
"sleeping_area":['double_bed', 'kids_bed', 'single_bed'],
"living_area":['sofa','coffee_table','table'],
"storage_area":['bookshelf', 'cabinet', 'children_cabinet', 'wardrobe', 'shelf'],
"study_area":['desk']
}
HIER_AREA_OTHER_OBJ_FROM_BEDROOM = {
"dressing_area":['dressing_chair'],
"sleeping_area":['nightstand', 'tv_stand'],
"living_area":[],
"storage_area":[],
"study_area":[],
"any_area":['armchair', 'ceiling_lamp', 'chair', 'floor_lamp', 'pendant_lamp', 'stool']
}


HIER_AREA_FUNCTION_OBJ_FROM_LIVINGROOM = {
    "living_area":['l_shaped_sofa', 'chaise_longue_sofa', 'lazy_sofa', 'loveseat_sofa', 'multi_seat_sofa', 'corner_side_table'],
    "tv_area":['tv_stand'],
    "dining_area":['dining_table', 'wine_cabinet'],
    "storage_area":['cabinet', 'wardrobe', 'shelf'],
    "office_area":['bookshelf', 'desk'],
    "rest_area":['lounge_chair', 'round_end_table']
}
HIER_AREA_OTHER_OBJ_FROM_LIVINGROOM = {
"living_area":['coffee_table'],
"tv_area": [],
"dining_area":['dining_chair'],
"storage_area":[],
"office_area":[],
"rest_area":[],
"any_area":['armchair', 'ceiling_lamp', 'chinese_chair', 'console_table', 'pendant_lamp', 'stool']
}
