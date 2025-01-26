model_name=FACTS
pred_len=720  # 96 192 336 720 -- sota: [0.139, 0.159, 0.176, 0.204]

num_factors=128
slot_size=64
d_model=64
n_heads=8
init_method=learnable
graph_embed=ms_conv2d
chunk_size=96

python -m demos._tests_.test_0 \
  --seed 2021 \
  --debugging 1 \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/time_series/electricity/ \
  --data_path electricity.csv \
  --log_path ./logs/FACTS/ \
  --model_id ECL_96'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --graph_embed ms_conv2d \
  --init_method $init_method \
  --router sfmx_attn \
  --decoder fgd \
  --rev_in 1 \
  --num_factors $num_factors \
  --slot_size $slot_size \
  --n_heads $n_heads \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --chunk_size $chunk_size \
  --batch_size 16 \
  --e_layers 2 \
  --enc_in 321 \
  --d_conv 2 \
  --d_model $d_model \
  --dropout 0.1 \
  --patience 3 \
  --learning_rate 0.0005 \
  --optimizer adam \
  --lradj cosine \
  --fast_approx 1 \
  --des 'Exp' \
  --num_workers 4 \
  --itr 1 \
  --train_epochs 15


find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf