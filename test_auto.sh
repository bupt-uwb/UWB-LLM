llm_name=DeepSeek-7B

python -u ./run.py \
    --model_mode 0\
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 50 \
    --token_len 256 \
    --llm_name $llm_name\
    --experts_rank 16 \
    --experts_scale 2.0 \
    --experts_num_per_tok 2\
    --experts_num 8\
    --gpu 0 \
    --plot \