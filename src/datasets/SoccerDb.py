import pandas as pd
import numpy as np
import math
import tensorflow as tf

CSV_INPUT_COLUMN_NAMES = [ 'HostWin','Deuce','GuestWin']
CSV_LABLEL_COLUMN_NAMES = [ 'HostGoal','GuestGoal','Comments']
CSV_NATIONS_COLUMN_NAMES = [ 'Host','Guest']
RESULT_NAMES = ['GuestWin','Deuce','HostWin']#2,1,0

trainfiles='worldcup.csv'
evalfiles='worldcup-eval.csv'
predfiles='wc2018.csv'

def Score2Res90(rec):
    rownum = rec.shape[0]
    data = pd.DataFrame(data=np.ones(rownum) , columns=['Res90'],dtype=np.int32)

    for i in range(rownum):
        if rec['Comments'][i]!=('NRM'):
            continue
        
        if rec['HostGoal'][i] > rec['GuestGoal'][i]:
            data['Res90'][i] = 2
        elif rec['HostGoal'][i] < rec['GuestGoal'][i]:
            data['Res90'][i] = 0
        #1 is preset
    return data

def load_data(data_dir='../data/',csvfile=None):
    train_y=[]

    data = pd.read_csv(data_dir+csvfile, header=0)
    x = data[CSV_INPUT_COLUMN_NAMES]
    y = Score2Res90(data[CSV_LABLEL_COLUMN_NAMES])
    return (x, y)

def load_traindata(data_dir='../data/'):
    return load_data(data_dir,trainfiles)

def load_evaldata(data_dir='../data/'):
    return load_data(data_dir,evalfiles)

def load_preddata(data_dir='../data/'):
    data = pd.read_csv(data_dir+predfiles, header=0)
    print(data)
    x = data[CSV_INPUT_COLUMN_NAMES]
    host = data['Host']
    guest = data['Guest']
    return (x,host,guest)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]

def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

if __name__ == '__main__':
    (train_x, train_y) = load_traindata()
    print(train_x, train_y)

    (eval_x, eval_y) = load_evaldata()
    print(eval_x, eval_y)

    (x,host,guest) = load_preddata()
    print(x)
    print(host)
    print(guest)
