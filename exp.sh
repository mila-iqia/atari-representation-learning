#!/usr/bin/env bash
python run_contrastive.py --env-name 'PongNoFrameskip-v4' --contrastive-lr 1e-3 --cuda-id 0 &
python run_contrastive.py --env-name 'PongNoFrameskip-v4' --contrastive-lr 5e-3 --cuda-id 1 &
python run_contrastive.py --env-name 'PongNoFrameskip-v4' --contrastive-lr 1e-4 --cuda-id 2 &
python run_contrastive.py --env-name 'PongNoFrameskip-v4' --contrastive-lr 5e-4 --cuda-id 3 &
wait
python run_contrastive.py --env-name 'MontezumaRevenge-v4' --contrastive-lr 1e-3 --cuda-id 0 &
python run_contrastive.py --env-name 'MontezumaRevenge-v4' --contrastive-lr 5e-3 --cuda-id 1 &
python run_contrastive.py --env-name 'MontezumaRevenge-v4' --contrastive-lr 1e-4 --cuda-id 2 &
python run_contrastive.py --env-name 'MontezumaRevenge-v4' --contrastive-lr 5e-4 --cuda-id 3 &