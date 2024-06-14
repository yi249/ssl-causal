#!/bin/bash
exp_name="$1"
run_cmd="python main.py \
    --exp_name=$exp_name \
    --data_dir=../datasets/col_morphomnist/missing/3000
    --hps cmmnist \
    --lr=0.001 \
    --bs=32 \
    --wd=0.01 \
    --beta=10 \
    --cw=2
    --expand_pa \
    --eval_freq=1\
    --epochs=1000 \
    --beta_warmup_steps=0 \
    --vae=hierarchical \
    --zw=0.1 \
    --zrw=0.2 \
    --random"

#run_cmd="python main.py \
#    --exp_name=$exp_name \
#    --data_dir=../../causalssl/datasets/mimic
#    --hps mimic192 \
#    --lr=0.001 \
#    --bs=12 \
#    --wd=0.01 \
#    --beta=10 \
#    --cw=2
#    --expand_pa \
#    --eval_freq=1\
#    --epochs=1000 \
#    --labelled=0.1 \
#    --beta_warmup_steps=0 \
#    --vae=hierarchical \
#    --zw=0.1 \
#    --zrw=0.2 "

if [ "$2" = "nohup" ]
then
  nohup ${run_cmd} > $exp_name.out 2>&1 &
  echo "Started training in background with nohup, PID: $!"
else
  ${run_cmd}
fi
