###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.latent_ode import LatentODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver

import sys
sys.path.insert(0, '../ours_impl')
from lv_field import LVField

from torch.distributions.normal import Normal
from lib.ode_func import ODEFunc, ODEFunc_w_Poisson

#####################################################################################################

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device,
        classif_per_tp = False, n_labels = 1,
        BLOB_OF_MODEL_SETTINGS=None
        ):

        dim = args.latents
        if args.poisson:
                lambda_net = utils.create_net(dim, input_dim,
                        n_layers = 1, n_units = args.units, nonlinear = nn.Tanh)

                # ODE function produces the gradient for latent state and for poisson rate
                ode_func_net = utils.create_net(dim * 2, args.latents * 2,
                        n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

                gen_ode_func = ODEFunc_w_Poisson(
                        input_dim = input_dim,
                        latent_dim = args.latents * 2,
                        ode_func_net = ode_func_net,
                        lambda_net = lambda_net,
                        device = device).to(device)
        else:
                dim = args.latents
                ode_func_net = utils.create_net(dim, args.latents,
                        n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

                gen_ode_func = ODEFunc(
                        input_dim = input_dim,
                        latent_dim = args.latents,
                        ode_func_net = ode_func_net,
                        device = device).to(device)

        z0_diffeq_solver = None
        n_rec_dims = args.rec_dims
        enc_input_dim = int(input_dim) * 2 # we concatenate the mask
        gen_data_dim = input_dim

        z0_dim = args.latents
        if args.poisson:
                z0_dim += args.latents # predict the initial poisson rate

        if args.z0_encoder == "odernn":
                ode_func_net = utils.create_net(n_rec_dims, n_rec_dims,
                        n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)


                # using = 'ours'
                # using = 'theirs'


                # if using == 'ours':

                #       z0_diffeq_solver = LVField(
                #                       dim=enc_input_dim * args.latents,
                #                       augmented_dim=input_dim * args.latents + 5,
                #                       num_layers=3,
                #                       )
                #       # pred_y = self.lv_field(first_point, time_steps_to_predict)


                # elif using == 'theirs':

                if True:
                        """
                        We won't be using this as the integration time-steps within encoder tends to be very short.
                        """
                        rec_ode_func = ODEFunc(
                                input_dim = enc_input_dim,
                                latent_dim = n_rec_dims,
                                ode_func_net = ode_func_net,
                                device = device).to(device)

                        z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents,
                                odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)


                encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver,
                        z0_dim = z0_dim, n_gru_units = args.gru_units, device = device).to(device)



        elif args.z0_encoder == "rnn":
                encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
                        lstm_output_size = n_rec_dims, device = device).to(device)
        else:
                raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

        decoder = Decoder(args.latents, gen_data_dim).to(device)

        if BLOB_OF_MODEL_SETTINGS['model'] == 'node':
                diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func,
                BLOB_OF_MODEL_SETTINGS['node__int_method'],
                # 'euler',
                args.latents,
                odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

        elif BLOB_OF_MODEL_SETTINGS['model'] == 'ours':
                e_vec_factor = 1.0
                t_d_factor = 1.0
                if BLOB_OF_MODEL_SETTINGS['dataset'] == 'activity':
                    e_vec_factor = 0.001
                    t_d_factor = 0.1
                elif BLOB_OF_MODEL_SETTINGS['dataset'] == 'periodic':
                    if BLOB_OF_MODEL_SETTINGS['timepoints'] == 1000:
                        e_vec_factor = 1.0
                        t_d_factor = 0.01
                diffeq_solver = LVField(
                        dim=args.latents,
                        augmented_dim=args.latents + BLOB_OF_MODEL_SETTINGS['ours__extra_augnmented_dim'],
                        num_layers=BLOB_OF_MODEL_SETTINGS['ours__num_layers'],
                        hidden_dim=BLOB_OF_MODEL_SETTINGS['ours__inn_hidden_dim'],

                        e_vec_factor=e_vec_factor,
                        t_d_factor=t_d_factor,
                )

        model = LatentODE(
                input_dim = gen_data_dim,
                latent_dim = args.latents,
                encoder_z0 = encoder_z0,
                decoder = decoder,
                diffeq_solver = diffeq_solver,
                z0_prior = z0_prior,
                device = device,
                obsrv_std = obsrv_std,
                use_poisson_proc = args.poisson,
                use_binary_classif = args.classif,
                linear_classifier = args.linear_classif,
                classif_per_tp = classif_per_tp,
                n_labels = n_labels,
                train_classif_w_reconstr = (args.dataset == "physionet")
                ).to(device)

        return model
