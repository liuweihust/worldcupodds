
import functools
import tensorflow as tf

from tensorflow.contrib import slim as slim
import WinLossNet

networks_map = {'WinLossNet': WinLossNet.WinLossNet_V1,
                }


def get_network(name):
    """Get a network object from a name.
    """
    return networks_map[name]
