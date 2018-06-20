import tensorflow as tf

WinLossNet_V1_Param=[3,32,64,128,256,3]

def SoccerDnnNet_V1(features,num_layer=3,num_size=128,learning_rate=0.003,model_dir='/tmp/train'):
    my_feature_columns = []
    for key in features:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[128, 128,128],
        # The model must choose between 3 classes.
        model_dir=model_dir,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003),
        n_classes=3)
    return classifier
