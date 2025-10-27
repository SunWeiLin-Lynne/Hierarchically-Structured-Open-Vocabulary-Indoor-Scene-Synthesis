# Hierarchically-Structured-Open-Vocabulary-Indoor-Scene-Synthesis

## Installation & Dependencies
```
conda create -n llm_hierarchy python=3.9 -y
pip install -r requirements.txt
```

## Data Preparation

## Layout Generation 
```
python run_one_time_hierarchy_generation.py --dataset_dir ./dataset/ATISS/data_output/bedroom --room bedroom --gpt_type gpt4 --unit m --regular_floor_plan --tmp_name PATH/TO/TMP_NAME --use_relative_position True --use_rel_pos_lib True --optimization_way merge_area_galo
--retrieval_way GVAE --gvae_epoch 360
```
