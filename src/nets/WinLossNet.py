import tensorflow as tf

WinLossNet_V1_Param=[3,32,64,128,256,3]

def WinLossNet_V1(inputs):
    input_n = WinLossNet_V1_Param[0]
    output_n = WinLossNet_V1_Param[-1]

    net=inputs
    with tf.variable_scope('WinLossNet_V1'):
        for i in range(len(WinLossNet_V1_Param)-2):
            dense = tf.layers.dense(inputs=net, units=WinLossNet_V1_Param[i+1], 
                        activation=tf.nn.relu,name='dense_%d'%i)
            net = tf.layers.dropout(inputs=dense, rate=0.4,name='dropout_%d'%i)

        logits = tf.layers.dense(inputs=net, units=output_n,name='dense_last')
        return logits

def WinLossNet_LOSS_V1(logits,Y):
    with tf.variable_scope('WinLossNet_LOSS_V1'):
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='cross_entropy')
        #loss = tf.reduce_mean(cross_entropy, name='loss')
        loss = tf.losses.sparse_softmax_cross_entropy(labels=Y, logits=logits)
        return loss
