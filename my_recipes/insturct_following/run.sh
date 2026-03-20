cd /code/yehangcheng/verl/recipe/insturct_following
SAVE_PATH=/code/yehangcheng/verl/recipe/insturct_following/test
python3 -m evaluation_main \
  --input_data=/code/yehangcheng/verl/recipe/insturct_following/input_data.jsonl \
  --input_response_data=$SAVE_PATH/ifeval.jsonl \
  --output_dir=$SAVE_PATH