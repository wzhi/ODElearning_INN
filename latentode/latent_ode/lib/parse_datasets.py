###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from lib.diffeq_solver import DiffeqSolver
from generate_timeseries import Periodic_1d
from torch.distributions import uniform

from torch.utils.data import DataLoader
from mujoco_physics import HopperPhysics
from physionet import PhysioNet, variable_time_collate_fn, get_data_min_max
from person_activity import PersonActivity, variable_time_collate_fn_activity

from sklearn import model_selection
import random

#####################################################################################################
def parse_datasets(args, device):


        def basic_collate_fn(batch, time_steps, args = args, device = device, data_type = "train"):
                batch = torch.stack(batch)
                data_dict = {
                        "data": batch,
                        "time_steps": time_steps}

                data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)
                return data_dict


        dataset_name = args.dataset

        n_total_tp = args.timepoints + args.extrap
        max_t_extrap = args.max_t / args.timepoints * n_total_tp

        ##################################################################
        # MuJoCo dataset
        if dataset_name == "hopper":
                dataset_obj = HopperPhysics(root='data', download=True, generate=False, device = device)
                dataset = dataset_obj.get_dataset()[:args.n]
                dataset = dataset.to(device)


                n_tp_data = dataset[:].shape[1]

                # Time steps that are used later on for exrapolation
                time_steps = torch.arange(start=0, end = n_tp_data, step=1).float().to(device)
                time_steps = time_steps / len(time_steps)

                dataset = dataset.to(device)
                time_steps = time_steps.to(device)

                if not args.extrap:
                        # Creating dataset for interpolation
                        # sample time points from different parts of the timeline,
                        # so that the model learns from different parts of hopper trajectory
                        n_traj = len(dataset)
                        n_tp_data = dataset.shape[1]
                        n_reduced_tp = args.timepoints

                        # sample time points from different parts of the timeline,
                        # so that the model learns from different parts of hopper trajectory
                        start_ind = np.random.randint(0, high=n_tp_data - n_reduced_tp +1, size=n_traj)
                        end_ind = start_ind + n_reduced_tp
                        sliced = []
                        for i in range(n_traj):
                                  sliced.append(dataset[i, start_ind[i] : end_ind[i], :])
                        dataset = torch.stack(sliced).to(device)
                        time_steps = time_steps[:n_reduced_tp]

                # Split into train and test by the time sequences
                train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

                n_samples = len(dataset)
                input_dim = dataset.size(-1)

                batch_size = min(args.batch_size, args.n)
                train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
                        collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "train"))
                test_dataloader = DataLoader(test_y, batch_size = n_samples, shuffle=False,
                        collate_fn= lambda batch: basic_collate_fn(batch, time_steps, data_type = "test"))

                data_objects = {"dataset_obj": dataset_obj,
                                        "train_dataloader": utils.inf_generator(train_dataloader),
                                        "test_dataloader": utils.inf_generator(test_dataloader),
                                        "input_dim": input_dim,
                                        "n_train_batches": len(train_dataloader),
                                        "n_test_batches": len(test_dataloader)}
                return data_objects

        ##################################################################
        # Physionet dataset

        if dataset_name == "physionet":
                train_dataset_obj = PhysioNet('data/physionet', train=True,
                                                                                quantization = args.quantization,
                                                                                download=True, n_samples = min(10000, args.n),
                                                                                device = device)
                # Use custom collate_fn to combine samples with arbitrary time observations.
                # Returns the dataset along with mask and time steps
                test_dataset_obj = PhysioNet('data/physionet', train=False,
                                                                                quantization = args.quantization,
                                                                                download=True, n_samples = min(10000, args.n),
                                                                                device = device)

                # Combine and shuffle samples from physionet Train and physionet Test
                total_dataset = train_dataset_obj[:len(train_dataset_obj)]

                if not args.classif:
                        # Concatenate samples from original Train and Test sets
                        # Only 'training' physionet samples are have labels. Therefore, if we do classifiction task, we don't need physionet 'test' samples.
                        total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]

                # Shuffle and split
                train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8,
                        random_state = 42, shuffle = True)

                record_id, tt, vals, mask, labels = train_data[0]

                n_samples = len(total_dataset)
                input_dim = vals.size(-1)

                batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
                data_min, data_max = get_data_min_max(total_dataset)

                train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False,
                        collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
                                data_min = data_min, data_max = data_max))
                test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False,
                        collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
                                data_min = data_min, data_max = data_max))

                attr_names = train_dataset_obj.params
                data_objects = {"dataset_obj": train_dataset_obj,
                                        "train_dataloader": utils.inf_generator(train_dataloader),
                                        "test_dataloader": utils.inf_generator(test_dataloader),
                                        "input_dim": input_dim,
                                        "n_train_batches": len(train_dataloader),
                                        "n_test_batches": len(test_dataloader),
                                        "attr": attr_names, #optional
                                        "classif_per_tp": False, #optional
                                        "n_labels": 1} #optional
                return data_objects

        ##################################################################
        # Human activity dataset

        if dataset_name == "activity":
                n_samples =  min(10000, args.n)
                dataset_obj = PersonActivity('data/PersonActivity',
                                                        download=True, n_samples =  n_samples, device = device)
                print(dataset_obj)
                # Use custom collate_fn to combine samples with arbitrary time observations.
                # Returns the dataset along with mask and time steps

                # Shuffle and split
                train_data, test_data = model_selection.train_test_split(dataset_obj, train_size= 0.8,
                        random_state = 42, shuffle = True)

                train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
                test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]

                record_id, tt, vals, mask, labels = train_data[0]
                input_dim = vals.size(-1)

                batch_size = min(min(len(dataset_obj), args.batch_size), args.n)

                train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False,
                        collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
                test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False,
                        collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))



                # vv=[a for a in test_dataloader]
                # print([vv[0].keys()])
                # for k, v in vv[0].items():
                #       print(k)
                #       if v != 'mode':
                #               print(v.shape)
                #       # else:
                #       print(v)

                data_objects = {"dataset_obj": dataset_obj,
                                        "train_dataloader": utils.inf_generator(train_dataloader),
                                        "test_dataloader": utils.inf_generator(test_dataloader),
                                        "input_dim": input_dim,
                                        "n_train_batches": len(train_dataloader),
                                        "n_test_batches": len(test_dataloader),
                                        "classif_per_tp": True, #optional
                                        "n_labels": labels.size(-1)}

                # print(data_objects)
                # asdasdsd

                return data_objects

        if dataset_name == "ecg":
                n_samples =  min(10000, args.n)
                # dataset_obj = PersonActivity('data/PersonActivity',
                #                                       download=True, n_samples =  n_samples, device = device)
                # print(dataset_obj)

                # Use custom collate_fn to combine samples with arbitrary time observations.
                # Returns the dataset along with mask and time steps

                # Shuffle and split
                # train_data, test_data = model_selection.train_test_split(dataset_obj, train_size= 0.8,
                #       random_state = 42, shuffle = True)

                # train_data = [train_data[i] for i in np.random.choice(len(train_data), len(train_data))]
                # test_data = [test_data[i] for i in np.random.choice(len(test_data), len(test_data))]


                import pandas as pd

                import os
                dir_path = os.path.dirname(os.path.realpath(__file__))


                mit_train = pd.read_csv(f'{dir_path}/../../neural_ode_classification/data/mitbih_train.csv',
                                                                header=None)
                mit_test = pd.read_csv(f'{dir_path}/../../neural_ode_classification/data/mitbih_test.csv', header=None)

                skip_zero_class = False
                def _get_x_y(_data_set, skip_zero_class):
                        # Separate target from data
                        y = _data_set[187]
                        x = _data_set.loc[:, :186]
                        if skip_zero_class:
                            _dataset_without_zero = y != 0
                            y = y[_dataset_without_zero]
                            x = x[_dataset_without_zero]
                        return x, y

                X_train, y_train = _get_x_y(mit_train, skip_zero_class)
                X_test, y_test = _get_x_y(mit_test, skip_zero_class)



                # record_id, tt, vals, mask, labels = train_data[0]
                # input_dim = vals.size(-1)


                # print("xtrain")
                # print(X_train)
                # print("ytrain")
                # print(y_train)


                # print(list(zip(X_train.to_numpy(), y_train.to_numpy()))[0])

                # print(X_train.to_numpy())


                # exit()



                def collect(batch):
                        # print('='*10)
                        # print('='*10)
                        # print('='*10)
                        # print(batch)
                        # print(len(batch))
                        # print(batch[0])
                        # print(batch[1])
                        # print(batch[0].shape)

                        total_tp = list(range(len(batch[0][0])))

                        # split_at = len(total_tp) // 2

                        # observed_data = torch.Tensor([b[0][:split_at] for b in batch]).unsqueeze(-1)
                        # # observed_tp = torch.Tensor([total_tp[:split_at] for _ in range(len(batch))])
                        # observed_tp = torch.Tensor(total_tp[:split_at])
                        # data_to_predict = torch.Tensor([b[0][split_at:] for b in batch]).unsqueeze(-1)
                        # # tp_to_predict = torch.Tensor([total_tp[split_at:] for _ in range(len(batch))])
                        # tp_to_predict = torch.Tensor(total_tp[split_at:])

                        labels = torch.Tensor([b[1] for b in batch])
                        if skip_zero_class:
                            labels = labels - 1 # THIS IS to remove class zero.

                        labels = torch.nn.functional.one_hot(labels.long())
                        # print(torch.nn.functional.one_hot(labels.long()))
                        # exit()




                        # print(labels)

                        # print(len(observed_data))
                        # print(len(observed_tp))
                        # print(len(data_to_predict))
                        # print(len(tp_to_predict))

                        # print(observed_data[0])
                        # print(observed_tp[0])
                        # print(data_to_predict[0])
                        # print(tp_to_predict[0])


                        data = torch.Tensor([b[0] for b in batch]).unsqueeze(-1).to(device)

                        splitted_dict = utils.split_data_interp(dict(
                                data=data,
                                time_steps=torch.Tensor(total_tp).to(device),
                                mask=torch.ones(data.shape).to(device),
                                labels=labels.to(device),
                        ))

                        # print(splitted_dict)
                        # exit()
                        # exit()









                        return dict(
                                observed_data=splitted_dict['observed_data'],
                                observed_tp=splitted_dict['observed_tp'],
                                data_to_predict=splitted_dict['data_to_predict'],
                                tp_to_predict=splitted_dict['tp_to_predict'],
                                observed_mask=splitted_dict['observed_mask'],
                                mask_predicted_data=splitted_dict['mask_predicted_data'],
                                labels=splitted_dict['labels'],


                                mode='interp',
                        )


                        # asdasd

                        # return dict(
                        #       observed_data=observed_data.to(device),
                        #       observed_tp=observed_tp.to(device),
                        #       data_to_predict=data_to_predict.to(device),
                        #       tp_to_predict=tp_to_predict.to(device),

                        #       # THIS IS DUMB -dummy mask
                        #       observed_mask=torch.ones(observed_data.shape).to(device),
                        #       mask_predicted_data=torch.ones(data_to_predict.shape).to(device),

                        #       labels=labels.to(device),
                        #       mode='interp',
                        # )


                batch_size = min(min(len(X_train), args.batch_size), args.n)
                train_dataloader = DataLoader(list(zip(X_train.to_numpy(), y_train.to_numpy())), batch_size= batch_size, shuffle=True,
                        collate_fn=collect)

                # [v for v in train_dataloader]
                test_dataloader = DataLoader(list(zip(X_test.to_numpy(), y_test.to_numpy())), batch_size=n_samples, shuffle=True,
                        collate_fn=collect)
                # train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False,
                #       collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "train"))
                # test_dataloader = DataLoader(test_data, batch_size=n_samples, shuffle=False,
                #       collate_fn= lambda batch: variable_time_collate_fn_activity(batch, args, device, data_type = "test"))



                input_dim = 1


                class_count = y_train.value_counts()
                num_samples = [item[1] for item in sorted(zip(class_count.index, class_count.values))]

                normedWeights = [1 - (x / sum(num_samples)) for x in num_samples]
                normedWeights = torch.FloatTensor(normedWeights).to(device)




                data_objects = {
                                        #"dataset_obj": dataset_obj,
                                        "train_dataloader": utils.inf_generator(train_dataloader),
                                        "test_dataloader": utils.inf_generator(test_dataloader),
                                        "input_dim": input_dim,
                                        "n_train_batches": len(train_dataloader),
                                        "n_test_batches": len(test_dataloader),
                                        # "classif_per_tp": True, #optional
                                        "n_labels": len(y_train.unique()),
                                        "class_weight": normedWeights,
                                        }

                print(data_objects)

                return data_objects

        ########### 1d datasets ###########

        # Sampling args.timepoints time points in the interval [0, args.max_t]
        # Sample points for both training sequence and explapolation (test)
        distribution = uniform.Uniform(torch.Tensor([0.0]),torch.Tensor([max_t_extrap]))
        time_steps_extrap =  distribution.sample(torch.Size([n_total_tp-1]))[:,0]
        time_steps_extrap = torch.cat((torch.Tensor([0.0]), time_steps_extrap))
        time_steps_extrap = torch.sort(time_steps_extrap)[0]

        dataset_obj = None
        ##################################################################
        # Sample a periodic function
        if dataset_name == "periodic":
                dataset_obj = Periodic_1d(
                        init_freq = None, init_amplitude = 1.,
                        final_amplitude = 1., final_freq = None,
                        z0 = 1.)

        ##################################################################

        if dataset_obj is None:
                raise Exception("Unknown dataset: {}".format(dataset_name))

        dataset = dataset_obj.sample_traj(time_steps_extrap, n_samples = args.n,
                noise_weight = args.noise_weight)

        # Process small datasets
        dataset = dataset.to(device)
        time_steps_extrap = time_steps_extrap.to(device)

        train_y, test_y = utils.split_train_test(dataset, train_fraq = 0.8)

        n_samples = len(dataset)
        input_dim = dataset.size(-1)

        batch_size = min(args.batch_size, args.n)
        train_dataloader = DataLoader(train_y, batch_size = batch_size, shuffle=False,
                collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "train"))
        test_dataloader = DataLoader(test_y, batch_size = args.n, shuffle=False,
                collate_fn= lambda batch: basic_collate_fn(batch, time_steps_extrap, data_type = "test"))

        data_objects = {#"dataset_obj": dataset_obj,
                                "train_dataloader": utils.inf_generator(train_dataloader),
                                "test_dataloader": utils.inf_generator(test_dataloader),
                                "input_dim": input_dim,
                                "n_train_batches": len(train_dataloader),
                                "n_test_batches": len(test_dataloader)}

        return data_objects


