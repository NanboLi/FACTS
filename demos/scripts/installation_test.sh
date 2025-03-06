# Run this script to test the installation of FACTS

python -m demos._tests_.test_0 \
  --enc_in 32 \
  --num_factors 32 \
  --slot_size 8 \
  --init_method learnable \
  --router sfmx_attn \
  --slim_mode 1 \
  --fast_mode 1 \
  --chunk_size -1 \
  --batch_size 8 \
  --seq_len 128 \
  --n_heads 4 

find . | grep -E "(__pycache__|\.pyc$)" | xargs rm -rf