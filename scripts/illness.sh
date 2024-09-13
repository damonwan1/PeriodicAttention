export CUDA_VISIBLE_DEVICES=2

model_name=PAformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_24 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 24 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_36 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 36 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_48 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 48 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 128 \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id illness_96_60 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 60 \
  --e_layers 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --d_model 16 \
  --d_ff 128 \
  --itr 1