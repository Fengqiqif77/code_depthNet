from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.compat.v1.keras.layers import ConvLSTM2D
import BasicConvLSTMCell


DISP_SCALING_RESNET50 = 10.0
MIN_DISP = 0.0125

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])




def rnn_depth_net_encoderlstm(current_input,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    W=current_input.get_shape()[2].value


    with tf.variable_scope('rnn_depth_net', reuse = tf.AUTO_REUSE) as sc:

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.relu
                            ):
            #I=slim.conv2d(current_input,1,[1,1],stride=2,padding='SAME')
            #cnv1,hidden1 = convLSTM(current_input, hidden_state[0], 32, [7, 7],stride=1, scope='cnv1_lstm')
            cnv1= ConvLSTM2D(32,[7,7],(1,1),padding='SAME',activation='relu',
                             recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv1_lstm')(current_input)

            #c1=tf.squeeze(cnv1,1)
            #cnv2,hidden2 = convLSTM(cnv1, hidden_state[1], 64, [5, 5],stride=2, scope='cnv2_lstm')
            #cnv2b, hidden2b = convLSTM(cnv2, hidden_state[2], 64, [5, 5], stride=1, scope='cnv2b_lstm')
            cnv2 = ConvLSTM2D(64, [5, 5], (2, 2), padding='SAME', activation='relu',
                               recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv2_lstm')(cnv1)

            cnv2b= ConvLSTM2D(64,[5,5],(1,1),padding='SAME',activation='relu',
                              recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv2b_lstm')(cnv2)

            #c2b=tf.squeeze(cnv2b,1)
            #cnv3, hidden3 = convLSTM(cnv2b, hidden_state[3], 128, [3, 3], stride=2,scope='cnv3_lstm')
            #cnv3b, hidden3b = convLSTM(cnv3, hidden_state[4], 128, [3, 3], stride=1, scope='cnv3b_lstm')
            cnv3  = ConvLSTM2D(128, [3, 3], (2, 2), padding='SAME', activation='relu',
                               recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv3_lstm')(cnv2b)

            cnv3b= ConvLSTM2D(128, [3, 3], (1, 1), padding='SAME', activation='relu',
                               recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv3b_lstm')(cnv3)

            #c3b=tf.squeeze(cnv3b,1)
            #cnv4, hidden4 = convLSTM(cnv3b, hidden_state[5], 256, [3, 3],stride=2, scope='cnv4_lstm')
            #cnv4b, hidden4b = convLSTM(cnv4, hidden_state[6], 256, [3, 3],stride=1, scope='cnv4b_lstm')
            cnv4  =ConvLSTM2D(256, [3, 3], (2, 2), padding='SAME', activation='relu',
                               recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv4_lstm')(cnv3b)

            #c4=tf.squeeze(cnv4,1)
            cnv4b=ConvLSTM2D(256, [3, 3], (1, 1), padding='SAME', activation='relu',
                               recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv4b_lstm')(cnv4)

            #cnv5, hidden5 = convLSTM(cnv4b, hidden_state[7], 512, [3, 3],stride=2, scope='cnv5_lstm')
            cnv5  = ConvLSTM2D(512, [3, 3], (2, 2), padding='SAME', activation='relu',
                               recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv5_lstm')(cnv4b)

            for i in range(current_input.get_shape()[1].value):
                cv5=tf.slice(cnv5,[0, i,0, 0, 0],
                             [-1,1 , -1,-1, -1])
                cv5=tf.squeeze(cv5,1)
                upcnv5 = slim.conv2d_transpose(cv5, 512, [3, 3], stride=1,scope='decnv5')
                upcnv5 = resize_like(upcnv5, cv5)
                i5_in  = tf.concat([upcnv5, cv5], axis=3)
                icnv5  = slim.conv2d(i5_in, 512, [3, 3], stride=1, scope='icnv5')

                cv4b = tf.slice(cnv4b, [0, i, 0, 0, 0],
                                [-1, 1, -1, -1, -1])
                cv4b = tf.squeeze(cv4b, 1)
                upcnv4b = slim.conv2d_transpose(icnv5, 256, [3, 3], stride=2, scope='decnv4b')
                upcnv4b = resize_like(upcnv4b, cv4b)
                i4b_in = tf.concat([upcnv4b, cv4b], axis=3)
                icnv4b = slim.conv2d(i4b_in, 256, [3, 3], stride=1, scope='icnv4b')

                cv4= tf.slice(cnv4, [0, i, 0, 0, 0],
                               [-1, 1,-1, -1, -1])
                cv4 = tf.squeeze(cv4,1)
                upcnv4 = slim.conv2d_transpose(icnv4b, 256, [3, 3], stride=1, scope='decnv4')
                upcnv4 = resize_like(upcnv4, cv4)
                i4_in  = tf.concat([upcnv4,cv4], axis=3)
                icnv4  = slim.conv2d(i4_in, 256, [3, 3], stride=1, scope='icnv4')

                cv3b = tf.slice(cnv3b, [0, i, 0, 0, 0],
                                [-1, 1, -1, -1, -1])
                cv3b = tf.squeeze(cv3b, 1)
                upcnv3b = slim.conv2d_transpose(icnv4, 128, [3, 3], stride=2, scope='decnv3b')
                upcnv3b = resize_like(upcnv3b, cv3b)
                i3b_in = tf.concat([upcnv3b, cv3b], axis=3)
                icnv3b = slim.conv2d(i3b_in, 128, [3, 3], stride=1, scope='icnv3b')

                cv3 = tf.slice(cnv3, [0, i, 0, 0, 0],
                                [-1, 1, -1, -1, -1])
                cv3 = tf.squeeze(cv3, 1)
                upcnv3 = slim.conv2d_transpose(icnv3b, 128, [3, 3], stride=1, scope='decnv3')
                upcnv3 = resize_like(upcnv3, cv3)
                i3_in = tf.concat([upcnv3, cv3], axis=3)
                icnv3 = slim.conv2d(i3_in, 128, [3, 3], stride=1, scope='icnv3')

                cv2b = tf.slice(cnv2b, [0, i, 0, 0, 0],
                                [-1, 1, -1, -1, -1])
                cv2b = tf.squeeze(cv2b, 1)
                upcnv2b = slim.conv2d_transpose(icnv3, 64, [3, 3], stride=2, scope='decnv2b')
                upcnv2b = resize_like(upcnv2b, cv2b)
                i2b_in = tf.concat([upcnv2b, cv2b], axis=3)
                icnv2b = slim.conv2d(i2b_in, 64, [3, 3], stride=1, scope='icnv2b')

                cv2 = tf.slice(cnv2, [0, i, 0, 0, 0],
                                [-1, 1, -1, -1, -1])
                cv2 = tf.squeeze(cv2, 1)
                upcnv2 = slim.conv2d_transpose(icnv2b, 64, [3, 3], stride=1, scope='decnv2')
                upcnv2 = resize_like(upcnv2, cv2)
                i2_in = tf.concat([upcnv2, cv2], axis=3)
                icnv2 = slim.conv2d(i2_in, 64, [3, 3], stride=1, scope='icnv2')



                cv1 = tf.slice(cnv1, [0, i, 0, 0, 0],
                               [-1, 1, -1, -1, -1])
                cv1 = tf.squeeze(cv1,1)
                upcnv1 = slim.conv2d_transpose(icnv2, 32,  [3, 3], stride=2, scope='decnv1')
                upcnv1 = resize_like(upcnv1, cv1)
                i1_in = tf.concat([upcnv1,cv1], axis=3)
                icnv1  = slim.conv2d(i1_in, 32,  [3, 3], stride=1, scope='icnv1')
                depth  = slim.conv2d(icnv1, 1,   [1, 1], stride=1,
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')*DISP_SCALING_RESNET50+MIN_DISP
                if i==0:
                    pred_depth=depth
                else:
                    pred_depth=tf.concat([pred_depth,depth],axis=2)
            return pred_depth










