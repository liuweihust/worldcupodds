
import functools
import tensorflow as tf

from tensorflow.contrib import slim as slim
import WinLossNet

networks_map = {'WinLossNet': WinLossNet.WinLossNet_V1,
                }

loss_map = {'WinLossNet': WinLossNet.WinLossNet_LOSS_V1,
                }


def get_network(name):
    return networks_map[name]

def get_loss(name):
    return loss_map[name]
