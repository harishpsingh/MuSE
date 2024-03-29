@echo off
set TASK_NAME=twitter15
set alpha=0.0001
set beta=0.0001
set theta=0.1
set sigma=1.0
set replace_start=1
set replace_end=3
set cls_init=0
set num_layers=6
set crf_dropout=0.5
set learning_rate=1e-4
set crf_learning_rate=1e-4
set bert_type=uncased
set cross_dropout=0.2

set CUDA_VISIBLE_DEVICES=0
python -u main.py ^
    --task %TASK_NAME% ^
    --per_gpu_train_batch_size 40 ^
    --per_gpu_eval_batch_size 40 ^
    --alpha %alpha% ^
    --beta %beta% ^
    --theta %theta% ^
    --output_dir ../outputs/%TASK_NAME%_output/alpha%alpha%_beta%beta%_theta%theta%_sigma%sigma%_rs%replace_start%_re%replace_end%_cls%cls_init%_l%num_layers%_lr%learning_rate%_clr%crf_learning_rate%_%bert_type%_cd%cross_dropout%_last/ ^
    --do_train ^
    --do_eval ^
    --num_train_epochs 10 ^
    --logging_steps 100 ^
    --save_steps 100 ^
    --evaluate_during_training ^
    --num_workers 8 ^
    --learning_rate %learning_rate% ^
    --crf_learning_rate %crf_learning_rate% ^
    --num_layers %num_layers% ^
    --replace_start %replace_start% ^
    --replace_end %replace_end% ^
    --cls_init %cls_init% ^
    --crf_dropout %crf_dropout% ^
    --skip_connection ^
    --use_quantile ^
    --bert_type %bert_type% ^
    --cross_dropout %cross_dropout%
