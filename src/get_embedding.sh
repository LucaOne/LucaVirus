# lucavirus for gene
python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 20480 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/analysis/gene_species_for_analysis.fasta \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus/gene_species_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

# lucavirus for prot
python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_specie_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus/prot_specie_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_homo_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus/prot_homo_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_domain_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus/prot_domain_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1


python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20240815023346 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_site_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus/prot_site_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

# lucavirus-mask for gene
python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 1400000 \
    --truncation_seq_length 20480 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/analysis/gene_species_for_analysis.fasta \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-mask/gene_species_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

# lucavirus-mask for prot
python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 1400000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_specie_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-mask/prot_specie_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1


python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 1400000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_homo_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-mask/prot_homo_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 1400000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_domain_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-mask/prot_domain_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1


python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-mask \
    --llm_version v1.0 \
    --llm_task_level token_level \
    --llm_time_str 20250113063529 \
    --llm_step 1400000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_site_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-mask/prot_site_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

# lucavirus-gene for gene
python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-gene \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250118234004 \
    --llm_step 3800000 \
    --truncation_seq_length 20480 \
    --trunc_type right \
    --seq_type gene \
    --input_file ../data/analysis/gene_species_for_analysis.fasta \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-gene/gene_species_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1


# lucavirus-prot for gene
python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_species_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-prot/prot_species_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_homo_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-prot/prot_homo_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1

python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_domain_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-prot/prot_domain_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1


python get_embedding.py \
    --llm_dir ..  \
    --llm_type lucavirus-prot \
    --llm_version v1.0 \
    --llm_task_level token_level,span_level,seq_level \
    --llm_time_str 20250504090749 \
    --llm_step 3800000 \
    --truncation_seq_length 10240 \
    --trunc_type right \
    --seq_type prot \
    --input_file ../data/analysis/prot_site_for_analysis.fasta  \
    --save_path /mnt2/sanyuan.hy/embedding/lucavirus/analysis/lucavirus-prot/prot_site_for_analysis/ \
    --embedding_type matrix \
    --matrix_add_special_token \
    --embedding_complete \
    --embedding_complete_seg_overlap \
    --gpu_id -1