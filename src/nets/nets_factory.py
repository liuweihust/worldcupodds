
import functools
import tensorflow as tf

from tensorflow.contrib import slim as slim
import WinLossNet
import soccer_dnn

networks_map = {'WinLossNet': WinLossNet.WinLossNet_V1,
                'dnn':soccer_dnn.SoccerDnnNet_V1,
                }

loss_map = {'WinLossNet': WinLossNet.WinLossNet_LOSS_V1,
                }


def get_network(name):
    return networks_map[name]

def get_loss(name):
    return loss_map[name]
