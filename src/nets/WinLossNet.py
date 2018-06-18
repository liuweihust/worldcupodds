

WinLossNet_V1_Param=[3,32,64,128,256,3]

def WinLossNet_V1(inputs):
    input_n = WinLossNet_V1_Param[0]
    output_n = WinLossNet_V1_Param[-1]

    net=inputs
    with tf.name_scope('WinLossNet_V1'):
        for i in range(len(WinLossNet_V1_Param-2):
            dense = tf.layers.dense(inputs=net, units=WinLossNet_V1_Param[i+1], 
                        activation=tf.nn.relu,name='dense_%d'%i)
            net = tf.layers.dropout(inputs=dense, rate=0.4,name='dropout_%d'%i)

        logits = tf.layers.dense(inputs=net, units=output_n,name='dense_last')
        return logits
    
