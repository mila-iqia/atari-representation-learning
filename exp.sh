#!/usr/bin/env bash
python run_contrastive.sh --env-name 'Pong-NoFrameskip-v4' --contrastive-lr 1e-3 --cuda-id 0
python run_contrastive.sh --env-name 'Pong-NoFrameskip-v4' --contrastive-lr 5e-3 --cuda-id 1
python run_contrastive.sh --env-name 'Pong-NoFrameskip-v4' --contrastive-lr 1e-4 --cuda-id 2
python run_contrastive.sh --env-name 'Pong-NoFrameskip-v4' --contrastive-lr 3e-4 --cuda-id 3