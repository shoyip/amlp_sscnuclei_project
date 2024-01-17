#!/usr/bin/env python
# coding: utf-8

# This script performs a "grid search" over parameters of depth, learning rate schedule, loss function.
# U-Net's are trained, model checkpoints are saved for further analysis and metrics are plotted.

import sys
sys.path.insert(1, '~/Documents/SegmStemNuclei/unet_nuclei')

from time import time

# these are all functions defined in a custom made module
from unet_nuclei.models import get_unet_model
from unet_nuclei.losses import weighted_bce_loss, dice_bce_loss, dice_wbce_loss, dice_loss
from unet_nuclei.metrics import dice_metric, iou_metric
from unet_nuclei.load_data import get_tif_loader

import tensorflow as tf
from tensorflow import keras
from keras.losses import binary_crossentropy as bce_loss

import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## DEFINE DICE + WBCE COMBINATIONS

def softdice_wbce_loss(y_true, y_pred):
    return dice_wbce_loss(y_true, y_pred, 3)

def dice_softwbce_loss(*args, **kwargs):
    return dice_wbce_loss(y_true, y_pred, 0.25)

for unet_depth, unet_depth_name in zip([4, 5, 6], ['4', '5', '6']):
    for loss_fx, loss_fx_name in zip([bce_loss, weighted_bce_loss, dice_loss, softdice_wbce_loss, dice_softwbce_loss], ['BCE', 'WBCE', 'Dice', 'SoftDiceWBCE', 'DiceSoftWBCE']):
        start = time()
        
        # state beginning of iteration
        print(f'\n\nSTARTING ITERATION WITH THE FOLLOWING CONFIG:')
        print(f'U-Net Depth: {unet_depth_name}')
        print(f'Loss Function: {loss_fx_name}')
        
        # load data
        train_gen, test_gen = get_tif_loader(dataset_path='../data/dataset/', batch_size=14)
        
        # define model with specified depth
        model = get_unet_model(depth=unet_depth)
        
        # define callbacks with Model Checkpoint
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath = f'run_Unet{unet_depth_name}_{loss_fx_name}Loss_constantLRSched'\
                    + '_model.{epoch:02d}-{val_loss:.2f}.keras', save_best_only=True)
        ]
            
        # compile the model with optimizer, loss function and metrics
        model.compile(
            optimizer = keras.optimizers.Adam(1e-4),
            loss = [loss_fx],
            metrics = ['accuracy', dice_metric, iou_metric]
        )
        
        # train the model
        history = model.fit(
            train_gen,
            epochs = 50,
            steps_per_epoch = 3 * 774 // 4,
            validation_steps = 3 * 331 // 4,
            validation_data = test_gen,
            callbacks = callbacks,
            verbose = 2
        )

        # plot and save the metrics
        figloss = plt.figure()
        plt.title(f'Loss [U-Net {unet_depth_name}, {loss_fx_name} Loss, constant LR Schedule]')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.grid()
        plt.legend()
        figloss.savefig(f'PlotLoss_Unet{unet_depth_name}_{loss_fx_name}Loss_constantLRSched', bbox_inches='tight')
        plt.close()
        
        figdice = plt.figure()
        plt.title(f'Dice Score [U-Net {unet_depth_name}, {loss_fx_name} Loss, constant LR Schedule]')
        plt.plot(history.history['dice_metric'], label='Training Dice Score')
        plt.plot(history.history['val_dice_metric'], label='Validation Dice Score')
        plt.xlabel('Epoch')
        plt.grid()
        plt.legend()
        figdice.savefig(f'PlotDice_Unet{unet_depth_name}_{loss_fx_name}Loss_constantLRSched', bbox_inches='tight')
        plt.close()

        end = time()
        print(f'It took {round((end-start)/60, 2)} to complete the training.')