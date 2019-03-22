#!/usr/bin/env bash
python -m scripts.run_contrastive --env-name 'PongNoFrameskip-v4' --lr 1e-3 --cuda-id 0 &
python -m scripts.run_contrastive --env-name 'PongNoFrameskip-v4' --lr 1e-4 --cuda-id 1 &
python -m scripts.run_contrastive --env-name 'PongNoFrameskip-v4' --lr 5e-4 --cuda-id 2 &
python -m scripts.run_contrastive --env-name 'PongNoFrameskip-v4' --lr 3e-4 --cuda-id 3 &