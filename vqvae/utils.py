import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
from datasets.block import BlockDataset, LatentBlockDataset
import numpy as np
from PIL import Image
import sys
sys.path.append(f"{os.getcwd()}/../transformations")
import transformations


def load_sut():
    from datasets.DatasetGenerator import TransformationDataSequence
    train_data_file_path = "~/supervised-transformation-dataset-all/"
    valid_data_file_path = "~/supervised-transformation-validation/"
    image_size = (192, 108, 3) # (240, 132, 3)
    print(f"Input image size: {image_size}")
    training_dataset = TransformationDataSequence(train_data_file_path, image_size=image_size, 
                                                  transform=transforms.Compose([transforms.ToTensor()]),\
                                                  robustification=False, noise_level=None)
    valid_dataset = TransformationDataSequence(valid_data_file_path, image_size=image_size, 
                                               transform=transforms.Compose([transforms.ToTensor()]),\
                                               robustification=False, noise_level=None)
    return training_dataset, valid_dataset


def load_uust(topo=None, max_dataset_size=None, transf="fisheye"):
    from datasets.UUSTDatasetGenerator import TransformationDataSequence
    key = None
    valkey = None

    if topo is None or topo == "general":
        train_data_file_path = "../data/supervised-transformation-dataset-alltransforms3FULL-T/"
        valid_data_file_path = "../data/supervised-transformation-dataset-alltransforms3FULL-V/"

    elif topo is "baseline3":
        train_data_file_path = "../data/supervised-transformation-dataset-alltransforms3FULL-V/"
        valid_data_file_path = "../data/supervised-transformation-dataset-alltransforms3FULL-V/"

    if transf == "resdec":
        image_size = (96, 54, 3) # (120, 67, 3)
    elif transf == "resinc":
        image_size = (480, 270, 3)
    else:
        image_size = (192, 108, 3) # (240, 132, 3)
    print(f"Input image size: {image_size}")

    training_dataset = TransformationDataSequence(train_data_file_path, image_size=image_size, 
                                                  transform=transforms.Compose([transforms.ToTensor()]),\
                                                  robustification=False, noise_level=None, key=key, 
                                                  max_dataset_size=max_dataset_size, transf=transf)
    valid_dataset = TransformationDataSequence(valid_data_file_path, image_size=image_size, 
                                               transform=transforms.Compose([transforms.ToTensor()]),\
                                               robustification=False, noise_level=None, key=valkey, transf=transf)
    return training_dataset, valid_dataset


def load_rl(topo=None, max_dataset_size=None, transf="fisheye"):
    from datasets.RLDatasetGenerator import TransformationDataSequence
    key = None
    valkey = None
    if topo is None or topo == "general":
        train_data_file_path = "../data/supervised-transformation-dataset-all/"
        valid_data_file_path = "../data/supervised-transformation-validation-alltopos/" #"/p/sdbb/supervised-transformation-validation/"
        # key = "sample-base"
    elif topo == "windy":
        train_data_file_path = "/p/autosoft/Meriel/RL-datasets/RLtrainwindy-fisheye-max200-0.05eval-1_24-15_42-L4LPFS"
        valid_data_file_path = "/p/autosoft/Meriel/RL-datasets/RLtrainwindy-resinc-max200-0.05eval-1_22-6_34-K2O72W"
        valkey = "ep"
    elif topo == "straight":
        train_data_file_path = "/p/autosoft/Meriel/RL-datasets/RLtrainstraight-fisheye-max200-0.05eval-1_24-9_3-0SS62N"
        valid_data_file_path = "/p/autosoft/Meriel/RL-datasets/RLtrainstraight-resinc-max200-0.05eval-1_21-21_20-EHC5Z7"
        valkey = "ep"
    elif topo == "Lturn":
        train_data_file_path =  "/p/sdbb/supervised-transformation-dataset-all/"
        valid_data_file_path = "/p/autosoft/Meriel/RL-datases/RLtrainLturn-resinc-max200-0.05eval-1_21-12_3-2BZHYR"
        key = "west_coast_usa-12930-Lturn-fisheye.None-run00-7ZYMOA"
        # train_data_file_path = "/p/autosoft/Meriel/supervised-transformation-dataset/west_coast_usa-12930-Lturn-fisheye.None-run00-7ZYMOA"
    elif topo == "Rturn":
        train_data_file_path = "/p/autosoft/Meriel/RL-datasets/RLtrainRturn-fisheye-max200-0.05eval-1_23-13_5-W2O0EM"
        valid_data_file_path = "/p/autosoft/Meriel/RL-datasets/RLtrainRturn-resinc-max200-0.05eval-1_20-14_55-UA0C42"
        valkey = "ep"
    if transf == "resdec":
        image_size = (96, 54, 3) #(120, 67, 3)
    elif transf == "resinc":
        image_size = (480, 270, 3)
    else:
        image_size = (192, 108, 3) # (240, 132, 3)
    print(f"Input image size: {image_size}")
    training_dataset = TransformationDataSequence(train_data_file_path, image_size=image_size, 
                                                  transform=transforms.Compose([transforms.ToTensor()]),\
                                                  robustification=False, noise_level=None, key=key, 
                                                  max_dataset_size=max_dataset_size, transf=transf)
    valid_dataset = TransformationDataSequence(valid_data_file_path, image_size=image_size, 
                                               transform=transforms.Compose([transforms.ToTensor()]),\
                                               robustification=False, noise_level=None, key=valkey, transf=transf)
    return training_dataset, valid_dataset


def load_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/randact_traj_length_100_n_trials_1000_n_contexts_1.npy'

    train = BlockDataset(data_file_path, train=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                         ]))

    val = BlockDataset(data_file_path, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(
                               (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ]))
    return train, val


def load_latent_block():
    data_folder_path = os.getcwd()
    data_file_path = data_folder_path + \
        '/data/latent_e_indices.npy'

    train = LatentBlockDataset(data_file_path, train=True,
                         transform=None)

    val = LatentBlockDataset(data_file_path, train=False,
                       transform=None)
    return train, val


def data_loaders(train_data, val_data, batch_size, shuffle=False):

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True)
    return train_loader, val_loader


def load_data_and_data_loaders(dataset, batch_size, shuffle=False, topo=None, max_dataset_size=None, transf=None):
    if dataset == "UUST":
        training_data, validation_data = load_uust(topo=topo, max_dataset_size=max_dataset_size, transf=transf)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size, shuffle=shuffle)
        x_train_var = 0.125 #np.var(training_data.get_inputs_distribution()) #np.var(training_data.train_data / 255.0)

    elif dataset == "SUT":
        training_data, validation_data = load_sut()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size, shuffle=shuffle)
        x_train_var = 0.125 #np.var(training_data.get_inputs_distribution()) #np.var(training_data.train_data / 255.0)

    elif dataset == "RL":
        training_data, validation_data = load_rl(topo=topo, max_dataset_size=max_dataset_size, transf=transf)
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size, shuffle=shuffle)
        x_train_var = 0.125 #np.var(training_data.get_inputs_distribution()) #np.var(training_data.train_data / 255.0)

    elif dataset == 'BLOCK':
        training_data, validation_data = load_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size, shuffle=shuffle)

        x_train_var = np.var(training_data.data / 255.0)
    elif dataset == 'LATENT_BLOCK':
        training_data, validation_data = load_latent_block()
        training_loader, validation_loader = data_loaders(
            training_data, validation_data, batch_size, shuffle=shuffle)

        x_train_var = np.var(training_data.data)

    else:
        raise ValueError(
            'Invalid dataset: only CIFAR10 and BLOCK datasets are supported.')

    return training_data, validation_data, training_loader, validation_loader, x_train_var


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(
        ' ', '_').replace(':', '_').lower()


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'

    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,
            #    SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')
            SAVE_MODEL_PATH + '/vqvae_' + timestamp + '.pth')
    # print(f"Saved model to {SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth'}")


def get_depth_image(img_name):
    img_orig = Image.open(img_name)
    img_depth = Image.open(str(img_name).replace("base", "depth"))
    img_depth_proc = transformations.blur_with_depth_image(np.array(img_orig), np.array(img_depth))
    return img_depth_proc