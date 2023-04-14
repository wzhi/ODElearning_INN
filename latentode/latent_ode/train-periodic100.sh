#!/bin/sh

python3 run_models.py --niters 500 -n 1000 -s 50 -l 10 --dataset periodic  --latent-ode --noise-weight 0.01 --viz -t 100
