from tensorflow import float32

from data.data_loader_outdoor import *
from model import *
import time
from utils_lr import *


class RNN_depth_trainer:
    '''
    A wrapper class which create a dataloader, construct a network model and compute loss
    '''
    # ========================
    # Construct data loader
    # ========================
    def initDataloader(self,
                       dataset_dir,
                       batch_size=3,
                       img_height=128, #192,#
                       img_width=416, #256,#
                       num_views=3,
                       num_epochs=20,
                       is_training=True):

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.num_views = num_views
        self.num_epochs = num_epochs

        # Initialize data loader
        initloader = DataLoader(dataset_dir,
                                batch_size,
                                img_height,
                                img_width,
                                num_epochs,
                                self.num_views)

        dataLoader = initloader.inputs(is_training)

        return dataLoader

    def load_data(self, dataLoader):
        '''
        Load a single data sample
        '''
        with tf.device(None):
            data_dict = dataLoader.get_next()
        return data_dict

    # ========================
    # Construct model
    # ========================
    def construct_model(self, data_dict):

        # ------------------
        # Forward
        # ------------------
        image_seq = data_dict['image_seq']


        hidden_state = [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                        None]  # Initialize hidden state
        hidden_state_pose = [None, None, None, None, None, None, None]
        est_poses = []

        #for i in range(self.num_views):

           # image = tf.slice(image_seq,
            #                 [0, 0, self.img_width * i, 0],
             #                [-1, -1, int(self.img_width), -1])

            #image.set_shape([self.batch_size, self.img_height, self.img_width, 3])

            # Depth prediction
            #enconderlstm
            #data=tf.expand_dims(image,1)
        data=tf.reshape(image_seq,[self.batch_size,self.num_views,self.img_height,self.img_width,3])
        tf.summary.image('image', image_seq)
        pred_depth= rnn_depth_net_encoderlstm(data, is_training=True)

        return [pred_depth] #

    # ========================
    # Compute loss
    # ========================
    def compute_loss(self, estimates,  data_dict, global_step): #

        est_depths = estimates[0]
        all_losses = []  # keep different losses and visualize in tensorboard
        output_dict = {}
        lamda=tf.constant(0.5)
        before=data_dict['depth_seq']
        gt_depth =1.0/ data_dict['depth_seq']
        tf.summary.image('before_depth',before)
        output_dict['depth'] = est_depths
        data_dict['gt']=gt_depth
        data_dict['est']=est_depths
        loss= tf.zeros([], dtype=tf.float32)
        def compute(gt,est):
            di = tf.log(est)-tf.log(gt)
            di=tf.where(tf.is_nan(di),tf.zeros_like(di),di)
            di=tf.where(tf.is_inf(di),tf.zeros_like(di),di)
            div = tf.count_nonzero(di, dtype=tf.float32)
            first_part=tf.reduce_sum(tf.pow(di,2))/(div+0.000000001)
            second_part=tf.multiply(tf.pow(tf.reduce_sum(di)/(div+0.000000001),2),lamda)
            l = first_part-second_part
            return l

        for i in range(self.num_views):
            depth_slice = tf.slice(est_depths,
                                   [0, 0, self.img_width * i, 0],
                                   [-1, -1, int(self.img_width), -1])

            gt_depth_slice = tf.slice(gt_depth,
                                     [0,0,self.img_width*i,0],
                                     [-1,-1,int(self.img_width),-1])

            loss+=compute(gt_depth_slice,depth_slice)

        tv = tf.trainable_variables()
        l2loss = 0.005 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])
        loss+= l2loss
        all_losses.append(loss)


        return  loss,all_losses, output_dict
    def sub_depth(self, est_depths):
        depth_slice1 = tf.slice(est_depths,
                               [0, 0, self.img_width * 0, 0],
                               [-1, -1, int(self.img_width), -1])
        depth_slice2 = tf.slice(est_depths,
                               [0, 0, self.img_width * 1, 0],
                               [-1, -1, int(self.img_width), -1])
        depth_slice3 = tf.slice(est_depths,
                               [0, 0, self.img_width * 2, 0],
                               [-1, -1, int(self.img_width), -1])
        depth = 1.0/tf.concat([depth_slice1,depth_slice2,depth_slice3],axis=2)
        return depth


    def tb_summary(self, output_dict, loss, all_losses):
        #上传数据至tensorflow绘图


        # flow_bw = tf.expand_dims(tf.py_func(flow_to_image, [output_dict['flow_bw'][0,:,:,:]], tf.uint8),axis=0)
        # tf.summary.image('flow_bw', flow_bw)
        # flow_fw = tf.expand_dims(tf.py_func(flow_to_image, [output_dict['flow_fw'][0,:,:,:]], tf.uint8),axis=0)
        # tf.summary.image('flow_fw', flow_fw)



        depth =self.sub_depth(output_dict['depth'])



        tf.summary.scalar('losses', loss)
        tf.summary.image('est_depth', depth)


        # Distinguish different losses
        #tf.summary.scalar('losses/depth', all_losses[0]/depth)



    # ========================
    # Run session
    # ========================
    def save(self, sess, checkpoint_dir, step, saver):
        '''
        Save checkpoints
        '''
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            saver.save(sess,
                       os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            saver.save(sess,
                       os.path.join(checkpoint_dir, model_name),
                       global_step=step)


    def train(self, train_op, avg_loss, eval_step, args, data_dict):

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        # Saver has all the trainable parameters
        saver = tf.train.Saver()
        #saver1 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rnn_depth_net"))
        #saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="pose_net"))

        total_time = 0.0
        # Session start
        with tf.Session(config=config) as sess:

            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(args.checkpoint_dir + '/logs/train',
                                                 sess.graph)
            eval_writer = tf.summary.FileWriter(args.checkpoint_dir + '/logs/eval')

            merged = tf.summary.merge_all()

            # Load parameters
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            with tf.name_scope("parameter_count"):
                parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                                 for v in tf.trainable_variables()])
            print("parameter_count =", sess.run(parameter_count))

            # Restore model
            if args.continue_train == True:
                saver.restore(sess, tf.train.latest_checkpoint(args.restore_path))

            try:
                step = 0
                while True:
                    start_time = time.time()
                    fetches = {
                        "loss": avg_loss,
                        "summary": merged,
                        "data_dict": data_dict
                    }

                    if step % args.eval_freq == 0 and args.eval_set_dir is not None:
                        do_eval = True
                    else:
                        do_eval = False
                        fetches[" "] = train_op

                    results = sess.run(fetches, feed_dict={eval_step: do_eval})


                    duration = time.time() - start_time

                    total_time += duration

                    if step % args.eval_freq == 0 and args.eval_set_dir is not None:
                        print('Step %d: eval loss = %.5f (%.3f sec)' % (step,
                                                                        results["loss"],
                                                                        duration))

                        eval_writer.add_summary(results["summary"], step)

                    elif step % args.summary_freq == 0:
                        print('Step %d: loss = %.5f (%.3f sec)' % (step,
                                                                   results["loss"],
                                                                   duration))

                        #print(results['data_dict']['est'][1])
                        #print(results['data_dict']['gt'][1])
                        train_writer.add_summary(results["summary"], step)


                    # Save latest model
                    if step % args.save_latest_freq == 0:
                        self.save(sess, args.checkpoint_dir, step, saver)

                    step += 1

            except tf.errors.OutOfRangeError:
                print('Total time: %f' % total_time)
                print('Done training for %d epochs, %d steps.' % (self.num_epochs,
                                                                  step))


