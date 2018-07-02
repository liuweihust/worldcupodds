import tensorflow as tf
from tensorflow.python.ops import nn

def SoccerDnnNet_V1(features,num_layer=3,num_size=128,learning_rate=0.001,
                    dropout=None,activation='relu',model_dir='/tmp/train'):
    my_feature_columns = []
    for key in features:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    layers=[]
    for i in range(num_layer):
        layers.append(num_size)

    if activation=='relu':
        activation_fn = nn.relu
    else:
        activation_fn = nn.sigmoid

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=layers,
        model_dir=model_dir,
        dropout=dropout,
        activation_fn=activation_fn,
        optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
        n_classes=3)
    return classifier
