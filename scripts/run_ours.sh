python3 cli.py \
--method pet \
--pattern_ids 1 2 3 \
--data_dir TU_tweets/data \
--model_type roberta \
--model_name_or_path roberta-base \
--task_name tweet-task \
--output_dir out-tweet$RANDOM \
--do_train \
--do_eval

