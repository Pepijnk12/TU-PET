python cli.py \
--method pet \
--pattern_ids 1 2 3 4 \
--data_dir fewglue-master/FewGLUE/COPA \
--model_type roberta \
--model_name_or_path roberta-large \
--task_name copa \
--output_dir out \
--pet_per_gpu_eval_batch_size 1 \
--pet_per_gpu_train_batch_size 1 \
--do_train \
--do_eval


