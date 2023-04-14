#!/bin/sh

#python3 run_models.py --niters 200 -n 10000 -l 15 --dataset ecg --latent-ode --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 50 --gru-units 50 --classif --linear-classif -b 1000

#python3 run_models.py --niters 400 -n 10000 -l 15 --dataset ecg --latent-ode --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 50 --gru-units 50 --classif --linear-classif -b 4000

# smaller learning rate
python3 run_models.py --niters 400 -n 10000 -l 15 --dataset ecg --latent-ode --rec-dims 100 --rec-layers 4 --gen-layers 2 --units 50 --gru-units 50 --classif --linear-classif -b 2000 --lr 0.0001
