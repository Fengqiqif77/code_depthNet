from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow.compat.v1.keras.layers import ConvLSTM2D
import BasicConvLSTMCell


DISP_SCALING_RESNET50 = 10.0
MIN_DISP = 0.01

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def convLSTM(input, hidden, filters, kernel, stride,scope):

    with tf.variable_scope(scope, initializer = tf.truncated_normal_initializer(stddev=0.1)):
        cell = BasicConvLSTMCell.BasicConvLSTMCell([input.get_shape()[1], input.get_shape()[2]], kernel, filters)

        if hidden is None:
              hidden = cell.zero_state(input.get_shape()[0], tf.float32)
        y_, hidden  = cell(input, hidden,stride)

    return y_, hidden



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
            cnv1= ConvLSTM2D(32,[3,3],(1,1),padding='SAME',activation='relu',
                             recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv1_lstm')(current_input)

            #c1=tf.squeeze(cnv1,1)
            #cnv2,hidden2 = convLSTM(cnv1, hidden_state[1], 64, [5, 5],stride=2, scope='cnv2_lstm')
            #cnv2b, hidden2b = convLSTM(cnv2, hidden_state[2], 64, [5, 5], stride=1, scope='cnv2b_lstm')
            cnv2 = ConvLSTM2D(64, [3, 3], (2, 2), padding='SAME', activation='relu',
                               recurrent_activation='hard_sigmoid',return_sequences=True,name='cnv2_lstm')(cnv1)

            cnv2b= ConvLSTM2D(64,[3,3],(1,1),padding='SAME',activation='relu',
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
                    activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')
                if i==0:
                    pred_depth=depth
                else:
                    pred_depth=tf.concat([pred_depth,depth],axis=2)
            return pred_depth




def rnn_depth_net_encoderlstm_wpose(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value

    with tf.variable_scope('rnn_depth_net', reuse = tf.AUTO_REUSE) as sc:

        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.005),
                            activation_fn=tf.nn.leaky_relu
                            ):

            cnv1  = slim.conv2d(current_input, 32,  [3, 3], stride=2, scope='cnv1')
            cnv1b, hidden1 = convLSTM(cnv1, hidden_state[0], 32, [3, 3], scope='cnv1_lstm')
            #cnv1b = slim.conv2d(cnv1,  32,  [3, 3], rate=2, stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [3, 3], stride=2, scope='cnv2')
            cnv2b, hidden2 = convLSTM(cnv2, hidden_state[1], 64, [3, 3], scope='cnv2_lstm')
            #cnv2b = slim.conv2d(cnv2,  64,  [3, 3], rate=2, stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b, hidden3 = convLSTM(cnv3, hidden_state[2], 128, [3, 3], scope='cnv3_lstm')
            #cnv3b = slim.conv2d(cnv3,  128, [3, 3], rate=2, stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b, hidden4 = convLSTM(cnv4, hidden_state[3], 256, [3, 3], scope='cnv4_lstm')
            #cnv4b = slim.conv2d(cnv4,  256, [3, 3], rate=2, stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
            cnv5b, hidden5 = convLSTM(cnv5, hidden_state[4], 256, [3, 3], scope='cnv5_lstm')
            #cnv5b = slim.conv2d(cnv5,  256, [3, 3], rate=2, stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b, hidden6 = convLSTM(cnv6, hidden_state[5], 256, [3, 3], scope='cnv6_lstm')
            #cnv6b = slim.conv2d(cnv6,  256, [3, 3], rate=2, stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b, hidden7 = convLSTM(cnv7, hidden_state[6], 512, [3, 3], scope='cnv7_lstm')
            #cnv7b = slim.conv2d(cnv7,  512, [3, 3], rate=2, stride=1, scope='cnv7b')

            with tf.variable_scope('pose'):

                pose_pred = slim.conv2d(cnv7b, 6, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                pose_final = tf.reshape(pose_avg, [-1, 6])*0.01


            upcnv7 = slim.conv2d_transpose(cnv7b, 256, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 256, [3, 3], stride=1, scope='icnv7')
            #icnv7, hidden8= convLSTM(i7_in, hidden_state[7], 256, [3, 3], scope='icnv7_lstm')

            upcnv6 = slim.conv2d_transpose(icnv7, 128, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 128, [3, 3], stride=1, scope='icnv6')
            #icnv6, hidden9= convLSTM(i6_in, hidden_state[8], 128, [3, 3], scope='icnv6_lstm')

            upcnv5 = slim.conv2d_transpose(icnv6, 128, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 128, [3, 3], stride=1, scope='icnv5')
            #icnv5, hidden10 = convLSTM(i5_in, hidden_state[9], 128, [3, 3], scope='icnv5_lstm')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            upcnv4 = resize_like(upcnv4, cnv3b)
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            #icnv4, hidden11 = convLSTM(i4_in, hidden_state[10], 128, [3, 3], scope='icnv4_lstm')

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            upcnv3 = resize_like(upcnv3, cnv2b)
            i3_in  = tf.concat([upcnv3, cnv2b], axis=3)
            icnv3  = slim.conv2d(i3_in, 64, [3, 3], stride=1, scope='icnv3')
            #icnv3, hidden12 = convLSTM(i3_in, hidden_state[11], 64, [3, 3], scope='icnv3_lstm')

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            upcnv2 = resize_like(upcnv2, cnv1b)
            i2_in  = tf.concat([upcnv2, cnv1b], axis=3)
            icnv2  = slim.conv2d(i2_in, 32, [3, 3], stride=1, scope='icnv2')
            #icnv2, hidden13 = convLSTM(i2_in, hidden_state[12], 32, [3, 3], scope='icnv2_lstm')

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            #icnv1, hidden14 = convLSTM(upcnv1, hidden_state[13], 16, [3, 3], scope='icnv1_lstm')
            icnv1  = slim.conv2d(upcnv1, 16,  [3, 3], stride=1, scope='icnv1')
            depth  = slim.conv2d(icnv1, 1,   [3, 3], stride=1,
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')*DISP_SCALING_RESNET50+MIN_DISP

            return depth,pose_final, [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]







