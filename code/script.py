import os

TASK_NAME = "twitter15"
alpha = 0.0001
beta = 0.0001
theta = 0.1
sigma = 1.0
replace_start = 1
replace_end = 3
cls_init = 0
num_layers = 6
crf_dropout = 0.5
learning_rate = 1e-4
crf_learning_rate = 1e-4
bert_type = "uncased"
cross_dropout = 0.2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

command = (
    f"python -u main.py "
    f"--task {TASK_NAME} "
    f"--per_gpu_train_batch_size 40 "
    f"--per_gpu_eval_batch_size 40 "
    f"--alpha {alpha} "
    f"--beta {beta} "
    f"--theta {theta} "
    f"--output_dir ../outputs/{TASK_NAME}_output/alpha{alpha}_beta{beta}_theta{theta}_sigma{sigma}_rs{replace_start}_re{replace_end}_cls{cls_init}_l{num_layers}_lr{learning_rate}_clr{crf_learning_rate}_{bert_type}_cd{cross_dropout}_last/ "
    f"--do_train "
    f"--do_eval "
    f"--num_train_epochs 10 "
    f"--logging_steps 100 "
    f"--save_steps 100 "
    f"--evaluate_during_training "
    f"--num_workers 8 "
    f"--learning_rate {learning_rate} "
    f"--crf_learning_rate {crf_learning_rate} "
    f"--num_layers {num_layers} "
    f"--replace_start {replace_start} "
    f"--replace_end {replace_end} "
    f"--cls_init {cls_init} "
    f"--crf_dropout {crf_dropout} "
    f"--skip_connection "
    f"--use_quantile "
    f"--bert_type {bert_type} "
    f"--cross_dropout {cross_dropout}"
)

os.system(command)




