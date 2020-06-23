# solution.py ---
#
# Filename: solution.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Sun Mar  4 19:58:30 2018 (-0800)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C), Visual Computing Group @ University of Victoria.

# Code:


import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from config import get_config, print_usage, print_config
from model import MyNetwork
from tensorboardX import SummaryWriter
from utils.cifar10 import load_data
from utils.datawrapper import CIFAR10Dataset
from utils.features import extract_h_histogram, extract_hog
from utils.preprocess import normalize


def data_criterion(config):
    """Returns the loss object based on the commandline argument for the data term

    """

    if config.loss_type == "cross_entropy":
        data_loss = nn.CrossEntropyLoss()
    elif config.loss_type == "svm":
        data_loss = nn.MultiMarginLoss()

    return data_loss


def model_criterion(config):
    """Loss function based on the commandline argument for the regularizer term"""

    def model_loss(model):
        loss = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                loss += torch.sum(param**2)

        return loss * config.l2_reg

    return model_loss


def train(config):
    """Training process.

    """

    # TODO (1 point): Initialize datasets for both training and validation
    train_data = CIFAR10Dataset(config,"train")
    valid_data = CIFAR10Dataset(config,"valid")

    # We'll load the mean and the standard deviation of the data that we
    # compute previously
    x_tr_mean = np.load("./data/{}_mean.npy".format(config.feature_type))
    x_tr_std = np.load("./data/{}_std.npy".format(config.feature_type))

    # TODO (1 point): Create data loader for training and validation.
    tr_data_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=False, num_workers=2,)
    va_data_loader = DataLoader(dataset=valid_data, batch_size=config.batch_size, shuffle=False, num_workers=2,)

    # TODO (1 point): Create model instance. We will use
    # `train_data.sample_shp`, mean and std from training data for
    # initialization.
    model = MyNetwork(config,train_data.sample_shp,x_tr_mean,x_tr_std)
    # Move model to gpu if cuda is available
    if torch.cuda.is_available():
        model = model.cuda()
    # Make sure that the model is set for training
    model.train()

    #
    # ALTHOUGH NOT IMPLEMENTED IN THE ASSIGNMENT, SEE BELOW ON HOW WE CONFIGURE
    # OUR LOSS FUNCTIONS
    #
    # Create loss objects
    data_loss = data_criterion(config)
    model_loss = model_criterion(config)

    #
    # ALTHOUGH NOT IMPLEMENTED IN THE ASSIGNMENT, SEE BELOW ON HOW WE CREATE AN
    # OPTIMIZER
    #
    # Create optimizier
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # No need to move the optimizer (as of PyTorch 1.0), it lies in the same
    # space as the model

    #
    # ALTHOUGH NOT IMPLEMENTED IN THE ASSIGNMENT, SEE BELOW ON HOW WE SETUP OUR
    # WRITERS SO THAT THEY WRITE TO A DIFFERENT SUB FOLDER. THIS ALLOWS US TO
    # EASILY DISPLAY SCALAR ON A SINGLE GRAPH IN TENSORBOARD.
    #
    # Create summary writer
    tr_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "train"))
    va_writer = SummaryWriter(
        log_dir=os.path.join(config.log_dir, "valid"))

    # Create log directory and save directory if it does not exist
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    # Initialize training
    iter_idx = -1  # make counter start at zero
    best_va_acc = 0  # to check if best validation accuracy
    # Prepare checkpoint file and model file to save and load from
    checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")
    bestmodel_file = os.path.join(config.save_dir, "best_model.pth")

    # Check for existing training results. If it existst, and the configuration
    # is set to resume `config.resume==True`, resume from previous training. If
    # not, delete existing checkpoint.
    if os.path.exists(checkpoint_file):
        if config.resume:
            # TODO (2 points): Use `torch.load` to load the checkpoint file and
            # the load the things that are required to continue training. For
            # the model and the optimizer, use `load_state_dict`. It's actually
            # a good idea to code the saving part first and then code this
            # part.
            print("Checkpoint found! Resuming")
            load_check = torch.load(checkpoint_file)
            model.load_state_dict(load_check['model'])
            optimizer.load_state_dict(load_check['optimizer'])
            
            # Note that we do not resume the epoch, since we will never be able
            # to properly recover the shuffling, unless we remember the random
            # seed, for example. For simplicity, we will simply ignore this,
            # and run `config.num_epoch` epochs regardless of resuming.
        else:
            os.remove(checkpoint_file)

    # Training loop
    for epoch in range(config.num_epoch):
        # For each iteration
        prefix = "Training Epoch {:3d}: ".format(epoch)
        all_training_yet = 0
        correct_training = 0
        val_all = 0
        val_corr = 0
        #
        # ALTHOUGH NOT IMPLEMENTED IN THE ASSIGNMENT, SEE BELOW ON HOW WE USE
        # OUR DATALOADER TO LOAD DATA. I'VE ALSO WRAPPED IT IN TQDM FOR
        # EYECANDY.
        #
        for data in tqdm(tr_data_loader, desc=prefix):
            # Counter
            iter_idx += 1

            # Split the data
            x, y = data

            # Send data to GPU if we have one
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # TODO (5 points): Implement the forward pass. Apply
            # `model.forward` to get the scores, then use `data_loss` and
            # `model_loss` to compute the total loss. Then compute the
            # gradients by performing the backward pass, apply the parameter
            # update using the optimizer, and finally prepare the optimizer for
            # the next iteration.
            scores = model.forward(x)
            loss_data = data_loss(scores,y.long())
            loss_model = model_loss(model)
            total_loss = loss_data+loss_model
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Monitor results every report interval
            if iter_idx % config.rep_intv == 0:
                # TODO (2 points): Compute accuracy (No gradients
                # required). We'll wrapp this part so that we prevent torch
                # from computing gradients.                
                _,prediction = torch.max(scores.data,1)
                all_training_yet += y.nelement()
                correct_training += prediction.eq(y.data).sum().item()
                train_accuracy = 100*(correct_training/all_training_yet)
                # TODO (2 points): Write loss and accuracy to tensorboard,
                # using keywords `loss` and `accuracy`.
                tr_writer.add_scalar("loss",total_loss,iter_idx)
                tr_writer.add_scalar("accuracy",train_accuracy,iter_idx)
                # TODO (1 point): Let's also save our most recent model so
                # that we can resume. We'll save this model in path defined by
                # `checkpoint_file`.  Note that in order to resume, we would
                # need to save, `iter_idx`, `best_va_acc`, `model`, and
                # `optimizer`. Note that for simplicity, we will not save
                # epoch.
                torch.save({'iter_idx':iter_idx,'best_va_acc':best_va_acc,'model':model.state_dict(),'optimizer':optimizer.state_dict() },checkpoint_file)

            # Validate results every validation interval
            if iter_idx % config.val_intv == 0:
                # List to contain all losses and accuracies for all the
                # training batches
                va_loss = []
                va_acc = []
                # TODO (1 point): Set model for evaluation
                model.eval()
                #
                # NOTE HOW WE ARE LOOPING OVER THE ENTIRE VALIDATION SET HERE.
                #
                for data in va_data_loader:

                    # Split the data
                    x, y = data

                    # Send data to GPU if we have one
                    if torch.cuda.is_available():
                        x = x.cuda()
                        y = y.cuda()

                    # TODO (2 points): Apply forward pass to compute the losses
                    # and accuracies for each of the validation batches. Note
                    # that we will again explicitly diable gradients.
                    scores = model.forward(x)
                    loss_data = data_loss(scores,y.long())
                    loss_model = model_loss(model)
                    total_loss = loss_data+loss_model
                    _,prediction = torch.max(scores.data,1)
                    val_all += y.nelement()
                    val_corr += prediction.eq(y.data).sum().item()
                    
                    val_accuracy = 100*(val_corr/val_all)
                    if(val_accuracy>best_va_acc):
                        best_va_acc = val_accuracy
                        torch.save({'iter_idx':iter_idx,'best_va_acc':best_va_acc,'model':model.state_dict(),'optimizer':optimizer.state_dict() },bestmodel_file)
                    va_loss.append(total_loss.item())
                    va_acc.append(val_accuracy)
                # TODO (1 point): Set model back for training
                model.train()
                # Take average
                va_loss = np.mean(va_loss)
                va_acc = np.mean(va_acc)

                # TODO (1 point): Write the average `loss` and `accurac` to
                # the validation writer. Also check if the best validation
                # accuracy is achieved, and save the model and all necessary
                # states using `torch.save` as `bestmodel_file`

                va_writer.add_scalar("loss",va_loss)
                va_writer.add_scalar("accuracy",va_acc)


def test(config):
    """Test routine"""

    # TODO: (5 points) Initialize Dataset for testing.
    test_data = CIFAR10Dataset(config,"test")

    # Load the mean and the standard deviation of the data that we compute
    # previously
    x_tr_mean = np.load("./data/{}_mean.npy".format(config.feature_type))
    x_tr_std = np.load("./data/{}_std.npy".format(config.feature_type))

    # Create data loader for the test dataset with two number of workers and no
    # shuffling.
    te_data_loader = DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=False, num_workers=2,)

    # Create model
    model = MyNetwork(config,test_data.sample_shp,x_tr_mean,x_tr_std)
    # Move to GPU if you have one.
    if torch.cuda.is_available():
        model = model.cuda()

    # Create loss objects
    data_loss = data_criterion(config)
    model_loss = model_criterion(config)

    # TODO: (10 points) Load our best model and set model for testing
    bestmodel_file = os.path.join(config.save_dir, "best_model.pth")
    load_check = torch.load(bestmodel_file)
    model.load_state_dict(load_check['model'])
    model.eval()    

    # TODO: (5 points) Implement The Test loop
    prefix = "Testing: "
    te_loss = []
    te_acc = []
    val_all = 0
    val_corr = 0
        
    for data in tqdm(te_data_loader, desc=prefix):
        x , y = data
        
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()
        scores = model.forward(x)
        loss_data = data_loss(scores,y.long())
        loss_model = model_loss(model)
        total_loss = loss_data+loss_model
        _,prediction = torch.max(scores.data,1)
        val_all += y.nelement()
        val_corr += prediction.eq(y.data).sum().item()
        val_accuracy = 100*(val_corr/val_all)
        te_loss.append(total_loss.item())
        te_acc.append(val_accuracy)    

    # Report Test loss and accuracy
    print("Test Loss = {}".format(np.mean(te_loss)))
    print("Test Accuracy = {}%".format(np.mean(te_acc)))


def main(config):
    """The main function."""

    if config.mode == "train":
        train(config)
    elif config.mode == "test":
        test(config)
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    print_config(config)
    main(config)


#
# solution.py ends here
