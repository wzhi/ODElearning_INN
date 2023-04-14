#!/bin/sh

(cd rotating_MNIST/ && python3 evaluate.py)
(cd stiff_ode/ && python3 run_rober.py)
(cd robotic/ && python3 run.py)
(cd systems/ && python3 LV_train_run_ours.py)
(cd systems/ && python3 LV_train_theirs.py)
(cd systems/ && python3 LV_train_run_theirs.py)
(cd systems/ && python3 lor_train_ours.py)
(cd systems/ && python3 lor_train_theirs.py)
(cd systems/ && python3 lor_run_all.py)
(cd latentode/latent_ode/ && ./train-periodic100.sh)
(cd latentode/latent_ode/ && ./train-periodic1000.sh)
(cd latentode/latent_ode/ && ./train-activity.sh)
(cd latentode/latent_ode/ && ./train-ecg.sh)

