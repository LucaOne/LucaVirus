#!/bin/bash
# only for prot
# the dataset only contain prot dataset
# 数据集信息
# 数据集
DATASET_NAME="lucavirus-prot"
# 版本号
DATASET_TYPE="v1.0"
# 预训练任务类别
TASK_TYPE="token_level,span_level,seq_level"
# 预训练任务
PRETRAIN_TASK_LEVEL_NAME="prot_mask,prot_site,prot_homo,prot_domain,prot_taxonomy,prot_keyword"

# 输入数据处理信息
# 支持的最大序列长度
MAX_LENGTH=3074
# batch内，小于最大长度则padding，padding方式，右边padding，batch_size=1则不会padding
PADDING_TYPE="right"
# 大于最大长度则truncate，truncate方式，右边truncate
TRUNCATION_TYPE="right"
# pooling方式，实际上没有使用
POOLING_TYPE="value_attention"

# 模型信息
# 模型类型
MODEL_TYPE="lucavirus"
# 最好的checkpoint的选择方式，使用loss最小的epoch
BEST_METRIC_TYPE="loss"
# embedding长度
HIDDEN_SIZE=2560
# 模型层数
NUM_ATTENTION_LAYERS=12
# 模型头数
NUM_ATTENTION_HEADS=20


# 最大迭代次数
max_epochs=4

# 梯度累积平均（多少个step进行累积）
gradient_accumulation_steps=32

# 间隔多少个step在log文件中写入loss（实际上是gradient_accumulation_steps与loss_logging_steps的最小公倍数, 这里是4000）
loss_logging_steps=1000

# 间隔多少个step在log文件中写入信息（实际上是gradient_accumulation_steps与logging_steps的最小公倍数, 这里是4000）
logging_steps=1000


# checkpoint的间隔step数(训练了n_gpu * 100,000个样本保存一个checkpoint)
save_steps=100000

# warmup_steps个step到达peak lr，实际上是warmup_steps=warmup_steps/gradient_accumulation_steps
warmup_steps=32000
# 更新lr的方式
scheduler_type="step"

# 最大迭代step次数(这么多次后，peak lr1变为lr2, 需要根据epoch,样本数量,n_gpu,batch_size,gradient_accumulation_steps进行估算）
# 最后想要变成多大的值比如从peak lr->lr2，那么就是(max_epochs*sample_cnt)*lr1/(n_gpu * batch_size * gradient_accumulation_steps*lr2))进行估算
# 4 * 5,239,537 * 2e-4/(8 * 1 * 32 * 5e-5)=327,471
max_steps=400000

# batch size for one GPU
batch_size=1
# 最大学习速率
learning_rate=2e-4

# 数据加载器worker数
worker_num=8

time_str=$(date "+%Y%m%d%H%M%S")

# 多少卡
nproc_per_node=2

export CUDA_VISIBLE_DEVICES=2,3

cd ../..
python -W ignore -m torch.distributed.launch --nnodes 1 --node_rank 0 --master_port 29999 --nproc_per_node=$nproc_per_node \
       run.py \
       --time_str $time_str \
       --tb_log_dir ../tb-logs/$MODEL_TYPE/$DATASET_TYPE/$TASK_TYPE/$DATASET_NAME/$time_str \
       --log_dir ../logs/$MODEL_TYPE/$DATASET_TYPE/$TASK_TYPE/$DATASET_NAME/$time_str \
       --output_dir ../models/$MODEL_TYPE/$DATASET_TYPE/$TASK_TYPE/$DATASET_NAME/$time_str \
       --num_attention_heads $NUM_ATTENTION_HEADS \
       --num_hidden_layers $NUM_ATTENTION_LAYERS \
       --hidden_size $HIDDEN_SIZE \
       --max_length $MAX_LENGTH  \
       --vocab_path ../vocab/$MODEL_TYPE/$DATASET_TYPE/vocab.txt \
       --tokenizer_dir ../vocab/$MODEL_TYPE/$DATASET_TYPE/vocab.txt \
       --add_special_tokens \
       --padding $PADDING_TYPE \
       --truncation $TRUNCATION_TYPE \
       --pooling_type $POOLING_TYPE \
       --model_type $MODEL_TYPE \
       --model_config ../config/$MODEL_TYPE.json \
       --train_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/train/ \
       --dev_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/dev/ \
       --test_data_dir ../dataset/$DATASET_NAME/$DATASET_TYPE/test/ \
       --gene_mask_label_filepath ../vocab/$MODEL_TYPE/$DATASET_TYPE/vocab.txt \
       --prot_mask_label_filepath ../vocab/$MODEL_TYPE/$DATASET_TYPE/vocab.txt \
       --gene_type_label_filepath ../label/$MODEL_TYPE/$DATASET_TYPE/lucavirus_gene_type_span_level_label.txt \
       --prot_homo_label_filepath ../label/$MODEL_TYPE/$DATASET_TYPE/lucavirus_prot_homo_span_level_label.txt \
       --prot_site_label_filepath ../label/$MODEL_TYPE/$DATASET_TYPE/lucavirus_prot_site_span_level_label.txt \
       --prot_domain_label_filepath ../label/$MODEL_TYPE/$DATASET_TYPE/lucavirus_prot_domain_span_level_label.txt \
       --gene_taxonomy_label_filepath ../label/$MODEL_TYPE/$DATASET_TYPE/lucavirus_gene_taxonomy_seq_level_label.txt \
       --prot_taxonomy_label_filepath ../label/$MODEL_TYPE/$DATASET_TYPE/lucavirus_prot_taxonomy_seq_level_label.txt \
       --prot_keyword_label_filepath ../label/$MODEL_TYPE/$DATASET_TYPE/lucavirus_prot_keyword_seq_level_label.txt \
       --gene_mask_output_mode multi_class \
       --prot_mask_output_mode multi_class \
       --gene_type_output_mode multi_class \
       --prot_homo_output_mode multi_class \
       --prot_site_output_mode multi_class \
       --prot_domain_output_mode multi_class \
       --gene_taxonomy_output_mode multi_class \
       --prot_taxonomy_output_mode multi_class \
       --prot_keyword_output_mode multi_label \
       --gene_mask_loss_type cce \
       --prot_mask_loss_type cce \
       --gene_type_loss_type cce \
       --prot_homo_loss_type cce \
       --prot_site_loss_type cce \
       --prot_domain_loss_type cce \
       --gene_taxonomy_loss_type cce \
       --prot_taxonomy_loss_type cce \
       --prot_keyword_loss_type bce \
       --ignore_index -100 \
       --gene_mask_classifier_size 2048 \
       --prot_mask_classifier_size 2048 \
       --gene_type_classifier_size 128 \
       --prot_homo_classifier_size 4096 \
       --prot_site_classifier_size 1024 \
       --prot_domain_classifier_size 10240 \
       --gene_taxonomy_classifier_size 2048 \
       --prot_taxonomy_classifier_size 2048 \
       --prot_keyword_classifier_size 2048 \
       --gene_mask_weight 1.0 \
       --prot_mask_weight 1.0 \
       --gene_type_weight 0.2 \
       --prot_homo_weight 0.2 \
       --prot_site_weight 0.2 \
       --prot_domain_weight 0.2 \
       --gene_taxonomy_weight 0.2 \
       --prot_taxonomy_weight 0.2 \
       --prot_keyword_weight 1.0 \
       --buffer_size 10240 \
       --worker_num $worker_num \
       --seed 1111 \
       --pretrain_task_level_type $TASK_TYPE \
       --pretrain_task_level_name $PRETRAIN_TASK_LEVEL_NAME \
       --per_gpu_train_batch_size $batch_size \
       --per_gpu_eval_batch_size $batch_size \
       --learning_rate $learning_rate \
       --num_train_epochs $max_epochs \
       --do_train \
       --do_eval \
       --do_test \
       --do_metrics \
       --evaluate_during_training \
       --best_metric_type $BEST_METRIC_TYPE \
       --logging_steps $logging_steps \
       --save_steps $save_steps \
       --gradient_accumulation_steps $gradient_accumulation_steps \
       --save_all \
       --start_epoch 2 \
       --dropout_prob 0.0 \
       --no_position_embeddings \
       --no_token_dropout \
       --scheduler_type $scheduler_type \
       --warmup_steps $warmup_steps \
       --max_steps $max_steps \
       --beta1 0.9 \
       --beta2 0.99 \
       --weight_decay 0.01 \
       --no_use_embed_layer_norm \
       --loss_logging_steps $loss_logging_steps \
       --model_dirpath ../models/lucagplm/v2.0/token_level,span_level,seq_level,structure_level/lucaone_gplm/20231125113045/checkpoint-step17600000