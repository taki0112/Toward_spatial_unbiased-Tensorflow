from utils import *
import time
from tensorflow.python.data.experimental import AUTOTUNE
from networks import *
import PIL.Image
import scipy
import pickle
automatic_gpu_usage()

class Inception_V3(tf.keras.Model):
    def __init__(self, name='Inception_V3'):
        super(Inception_V3, self).__init__(name=name)

        # tf.keras.backend.image_data_format = 'channels_first'
        self.inception_v3_preprocess = tf.keras.applications.inception_v3.preprocess_input
        self.inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        self.inception_v3.trainable = False

    def torch_normalization(self, x):
        x /= 255.

        r, g, b = tf.split(axis=-1, num_or_size_splits=3, value=x)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x = tf.concat(axis=-1, values=[
            (r - mean[0]) / std[0],
            (g - mean[1]) / std[1],
            (b - mean[2]) / std[2]
        ])

        return x

    # @tf.function
    def call(self, x, training=False, mask=None):
        # x = self.torch_normalization(x)
        x = self.inception_v3(x, training=training)

        return x

    def inference_feat(self, x, training=False):
        inception_real_img = adjust_dynamic_range(x, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.float32)
        inception_real_img = tf.image.resize(inception_real_img, [299, 299], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
        inception_real_img = self.torch_normalization(inception_real_img)

        inception_feat = self.inception_v3(inception_real_img, training=training)

        return inception_feat

class StyleGAN2():
    def __init__(self, t_params, strategy):
        super(StyleGAN2, self).__init__()

        self.model_name = 'StyleGAN2'
        self.phase = t_params['phase']
        self.checkpoint_dir = t_params['checkpoint_dir']
        self.result_dir = t_params['result_dir']
        self.log_dir = t_params['log_dir']
        self.sample_dir = t_params['sample_dir']
        self.dataset_name = t_params['dataset']
        self.config = t_params['config']

        self.n_total_image = t_params['n_total_image'] * 1000

        self.strategy = strategy
        self.batch_size = t_params['batch_size']
        self.each_batch_size = t_params['batch_size'] // t_params['NUM_GPUS']

        self.NUM_GPUS = t_params['NUM_GPUS']
        self.iteration = self.n_total_image // self.batch_size

        self.n_samples = min(t_params['batch_size'], t_params['n_samples'])
        self.n_test = t_params['n_test']
        self.img_size = t_params['img_size']

        self.log_template = 'step [{}/{}]: elapsed: {:.2f}s, d_loss: {:.3f}, g_loss: {:.3f}, r1_reg: {:.3f}, pl_reg: {:.3f}, fid: {:.2f}, best_fid: {:.2f}, best_fid_iter: {}'

        self.lazy_regularization = t_params['lazy_regularization']
        self.print_freq = t_params['print_freq']
        self.save_freq = t_params['save_freq']

        self.r1_gamma = 10.0

        # setup optimizer params
        self.g_params = t_params['g_params']

        self.d_params = t_params['d_params']
        self.g_opt = t_params['g_opt']
        self.d_opt = t_params['d_opt']
        self.g_opt = self.set_optimizer_params(self.g_opt)
        self.d_opt = self.set_optimizer_params(self.d_opt)

        self.pl_minibatch_shrink = 2
        self.pl_decay = 0.01
        self.pl_weight = float(self.pl_minibatch_shrink)
        self.pl_denorm = tf.math.rsqrt(float(self.img_size) * float(self.img_size))
        self.pl_mean = tf.Variable(initial_value=0.0, name='pl_mean', trainable=False,
                                   synchronization=tf.VariableSynchronization.ON_READ,
                                   aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA
                                   )

        self.sample_dir = os.path.join(self.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(self.log_dir, self.model_dir)
        check_folder(self.log_dir)



        dataset_path = './dataset'
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)

        print(self.dataset_path)

        if os.path.exists('{}_mu_cov.pickle'.format(self.dataset_name)):
            with open('{}_mu_cov.pickle'.format(self.dataset_name), 'rb') as f:
                self.real_mu, self.real_cov = pickle.load(f)
            self.real_cache = True
            print("Pickle load success !!!")
        else:
            print("Pickle load fail !!!")
            self.real_cache = False

        self.fid_samples_num = 10000
        print()

        physical_gpus = tf.config.experimental.list_physical_devices('GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        print("Each batch size : ", self.each_batch_size)
        print("Global batch size : ", self.batch_size)
        print("Target image size : ", self.img_size)
        print("Print frequency : ", self.print_freq)
        print("Save frequency : ", self.save_freq)

        print("TF Version :", tf.__version__)

    def set_optimizer_params(self, params):
        if self.lazy_regularization:
            mb_ratio = params['reg_interval'] / (params['reg_interval'] + 1)
            params['learning_rate'] = params['learning_rate'] * mb_ratio
            params['beta1'] = params['beta1'] ** mb_ratio
            params['beta2'] = params['beta2'] ** mb_ratio
        return params

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):
        if self.phase == 'train':
            """ Input Image"""
            img_class = Image_data(self.img_size, self.g_params['z_dim'], self.g_params['labels_dim'], self.dataset_path)
            img_class.preprocess()

            dataset_num = len(img_class.train_images)
            if dataset_num > 10000:
                self.fid_samples_num = 50000
            print("Dataset number : ", dataset_num)
            print()

            dataset_slice = tf.data.Dataset.from_tensor_slices(img_class.train_images)

            gpu_device = '/gpu:0'

            dataset_iter = dataset_slice.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()
            dataset_iter = dataset_iter.map(map_func=img_class.image_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=True)
            dataset_iter = dataset_iter.prefetch(buffer_size=AUTOTUNE)
            dataset_iter = self.strategy.experimental_distribute_dataset(dataset_iter)

            img_slice = dataset_slice.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True, seed=777)
            img_slice = img_slice.map(map_func=inception_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=False)
            img_slice = img_slice.prefetch(buffer_size=AUTOTUNE)
            self.fid_img_slice = self.strategy.experimental_distribute_dataset(img_slice)

            self.dataset_iter = iter(dataset_iter)


            """ Network """
            self.generator = Generator(self.g_params, name='Generator')
            self.discriminator = Discriminator(self.d_params, name='Discriminator')
            self.g_clone = Generator(self.g_params, name='Generator')
            self.inception_model = Inception_V3()

            """ Finalize model (build) """
            test_latent = np.ones((1, self.g_params['z_dim']), dtype=np.float32)
            test_labels = np.ones((1, self.g_params['labels_dim']), dtype=np.float32)
            test_images = np.ones((1, 3, self.img_size, self.img_size), dtype=np.float32)
            test_images_inception = np.ones((1, 299, 299, 3), dtype=np.float32)

            _, __ = self.generator([test_latent, test_labels], training=False)
            _ = self.discriminator([test_images, test_labels], training=False)
            _, __ = self.g_clone([test_latent, test_labels], training=False)
            _ = self.inception_model(test_images_inception)

            # Copying g_clone
            self.g_clone.set_weights(self.generator.get_weights())

            """ Optimizer """
            self.d_optimizer = tf.keras.optimizers.Adam(self.d_opt['learning_rate'],
                                                        beta_1=self.d_opt['beta1'],
                                                        beta_2=self.d_opt['beta2'],
                                                        epsilon=self.d_opt['epsilon'])
            self.g_optimizer = tf.keras.optimizers.Adam(self.g_opt['learning_rate'],
                                                        beta_1=self.g_opt['beta1'],
                                                        beta_2=self.g_opt['beta2'],
                                                        epsilon=self.g_opt['epsilon'])

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(generator=self.generator,
                                            g_clone=self.g_clone,
                                            discriminator=self.discriminator,
                                            g_optimizer=self.g_optimizer,
                                            d_optimizer=self.d_optimizer)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)
            self.start_iteration = 0

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
                print('Latest checkpoint restored!!')
                print('start iteration : ', self.start_iteration)
            else:
                print('Not restoring from saved checkpoint')

        else:
            """ Test """
            """ Network """
            self.g_clone = Generator(self.g_params, name='Generator')
            self.discriminator = Discriminator(self.d_params, name='Discriminator')

            """ Finalize model (build) """
            test_latent = np.ones((1, self.g_params['z_dim']), dtype=np.float32)
            test_labels = np.ones((1, self.g_params['labels_dim']), dtype=np.float32)
            test_images = np.ones((1, 3, self.img_size, self.img_size), dtype=np.float32)
            _ = self.discriminator([test_images, test_labels], training=False)
            _, __ = self.g_clone([test_latent, test_labels], training=False)

            """ Checkpoint """
            self.ckpt = tf.train.Checkpoint(g_clone=self.g_clone, discriminator=self.discriminator)
            self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=2)

            if self.manager.latest_checkpoint:
                self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
                print('Latest checkpoint restored!!')
            else:
                print('Not restoring from saved checkpoint')

    def d_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            real_scores = self.discriminator([real_images, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            d_adv_loss = tf.math.softplus(fake_scores)
            d_adv_loss += tf.math.softplus(-real_scores)
            d_adv_loss = multi_gpu_loss(d_adv_loss, global_batch_size=self.batch_size)

            d_loss = d_adv_loss

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        return d_loss, d_adv_loss

    def d_reg_train_step(self, z, real_images, labels):
        with tf.GradientTape() as d_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            real_scores = self.discriminator([real_images, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            d_adv_loss = tf.math.softplus(fake_scores)
            d_adv_loss += tf.math.softplus(-real_scores)

            # simple GP
            with tf.GradientTape() as p_tape:
                p_tape.watch([real_images, labels])
                real_loss = tf.reduce_sum(self.discriminator([real_images, labels], training=True))

            real_grads = p_tape.gradient(real_loss, real_images)
            r1_penalty = tf.reduce_sum(tf.math.square(real_grads), axis=[1, 2, 3])
            r1_penalty = tf.expand_dims(r1_penalty, axis=1)
            r1_penalty = r1_penalty * self.d_opt['reg_interval']

            # combine
            d_adv_loss += r1_penalty * (0.5 * self.r1_gamma)
            d_adv_loss = multi_gpu_loss(d_adv_loss, global_batch_size=self.batch_size)

            d_loss = d_adv_loss

        d_gradients = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        r1_penalty = multi_gpu_loss(r1_penalty, global_batch_size=self.batch_size)

        return d_loss, d_adv_loss, r1_penalty

    def g_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, _ = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_adv_loss = tf.math.softplus(-fake_scores)
            g_adv_loss = multi_gpu_loss(g_adv_loss, global_batch_size=self.batch_size)

            g_loss = g_adv_loss

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return g_loss, g_adv_loss

    def g_reg_train_step(self, z, labels):
        with tf.GradientTape() as g_tape:
            # forward pass
            fake_images, w_broadcasted = self.generator([z, labels], training=True)
            fake_scores = self.discriminator([fake_images, labels], training=True)

            # gan loss
            g_adv_loss = tf.math.softplus(-fake_scores)

            # path length regularization
            pl_reg = self.path_regularization(pl_minibatch_shrink=self.pl_minibatch_shrink)

            # combine
            g_adv_loss += pl_reg
            g_adv_loss = multi_gpu_loss(g_adv_loss, global_batch_size=self.batch_size)

            g_loss = g_adv_loss

        g_gradients = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        pl_reg = multi_gpu_loss(pl_reg, global_batch_size=self.batch_size)

        return g_loss, g_adv_loss, pl_reg

    def path_regularization(self, pl_minibatch_shrink=2):
        # path length regularization
        # Compute |J*y|.

        pl_minibatch = tf.maximum(1, tf.math.floordiv(self.each_batch_size, pl_minibatch_shrink))
        pl_z = tf.random.normal(shape=[pl_minibatch, self.g_params['z_dim']], dtype=tf.float32)
        pl_labels = tf.random.normal(shape=[pl_minibatch, self.g_params['labels_dim']], dtype=tf.float32)

        with tf.GradientTape() as pl_tape:
            pl_tape.watch([pl_z, pl_labels])
            pl_fake_images, pl_w_broadcasted = self.generator([pl_z, pl_labels], training=True)

            pl_noise = tf.random.normal(tf.shape(pl_fake_images), mean=0.0, stddev=1.0, dtype=tf.float32) * self.pl_denorm
            pl_noise_applied = tf.reduce_sum(pl_fake_images * pl_noise)

        pl_grads = pl_tape.gradient(pl_noise_applied, pl_w_broadcasted)
        pl_lengths = tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(tf.math.square(pl_grads), axis=2), axis=1))

        # Track exponential moving average of |J*y|.
        pl_mean_val = self.pl_mean + self.pl_decay * (tf.reduce_mean(pl_lengths) - self.pl_mean)
        self.pl_mean.assign(pl_mean_val)

        # Calculate (|J*y|-a)^2.
        pl_penalty = tf.square(pl_lengths - self.pl_mean) * self.g_opt['reg_interval']

        # compute
        pl_reg = pl_penalty * self.pl_weight

        return pl_reg

    """ Distribute Train """
    @tf.function
    def distribute_d_train_step(self, z, real_images, labels):
        d_loss, d_adv_loss = self.strategy.run(self.d_train_step, args=(z, real_images, labels))

        d_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss, axis=None)
        d_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, d_adv_loss, axis=None)

        return d_loss, d_adv_loss

    @tf.function
    def distribute_d_reg_train_step(self, z, real_images, labels):
        d_loss, d_adv_loss, r1_penalty = self.strategy.run(self.d_reg_train_step, args=(z, real_images, labels))

        d_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, d_loss, axis=None)
        d_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, d_adv_loss, axis=None)
        r1_penalty = self.strategy.reduce(tf.distribute.ReduceOp.SUM, r1_penalty, axis=None)

        return d_loss, d_adv_loss, r1_penalty

    @tf.function
    def distribute_g_train_step(self, z, labels):
        g_loss, g_adv_loss = self.strategy.run(self.g_train_step, args=(z, labels))

        g_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss, axis=None)
        g_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, g_adv_loss, axis=None)

        return g_loss, g_adv_loss

    @tf.function
    def distribute_g_reg_train_step(self, z, labels):
        g_loss, g_adv_loss, pl_reg = self.strategy.run(self.g_reg_train_step, args=(z, labels))

        g_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, g_loss, axis=None)
        g_adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, g_adv_loss, axis=None)
        pl_reg = self.strategy.reduce(tf.distribute.ReduceOp.SUM, pl_reg, axis=None)

        return g_loss, g_adv_loss, pl_reg

    def train(self):

        start_time = time.time()

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)

        # start training
        print('max_steps: {}'.format(self.iteration))
        losses = {'g/loss': 0.0, 'd/loss': 0.0, 'r1_reg': 0.0, 'pl_reg': 0.0,
                  'g/adv_loss': 0.0,
                  'd/adv_loss': 0.0,
                  'fid': 0.0, 'best_fid': 0.0, 'best_fid_iter': 0}
        fid = 0
        best_fid = 1000
        best_fid_iter = 0
        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            x_real, z, labels = next(self.dataset_iter)

            if idx == 0:
                g_params = self.generator.count_params()
                d_params = self.discriminator.count_params()
                print("G network parameters : ", format(g_params, ','))
                print("D network parameters : ", format(d_params, ','))
                print("Total network parameters : ", format(g_params + d_params, ','))

            # update discriminator
            # At first time, each function takes 1~2 min to make the graph.
            if (idx + 1) % self.d_opt['reg_interval'] == 0:
                d_loss, d_adv_loss, r1_reg = self.distribute_d_reg_train_step(z, x_real, labels)
                losses['r1_reg'] = np.float64(r1_reg)
            else:
                d_loss, d_adv_loss = self.distribute_d_train_step(z, x_real, labels)

            losses['d/loss'] = np.float64(d_loss)
            losses['d/adv_loss'] = np.float64(d_adv_loss)

            # update generator
            # At first time, each function takes 1~2 min to make the graph.
            if (idx + 1) % self.g_opt['reg_interval'] == 0:
                g_loss, g_adv_loss, pl_reg = self.distribute_g_reg_train_step(z, labels)
                losses['pl_reg'] = np.float64(pl_reg)
            else:
                g_loss, g_adv_loss = self.distribute_g_train_step(z, labels)

            losses['g/loss'] = np.float64(g_loss)
            losses['g/adv_loss'] = np.float64(g_adv_loss)


            # update g_clone
            self.g_clone.set_as_moving_average_of(self.generator)

            if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1 :
                fid = self.calculate_FID()
                if fid < best_fid:
                    print("BEST FID UPDATED")
                    best_fid = fid
                    best_fid_iter = idx
                    self.manager.save(checkpoint_number=idx)
                losses['fid'] = np.float64(fid)


            # save to tensorboard

            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', losses['g/loss'], step=idx)
                tf.summary.scalar('g_adv_loss', losses['g/adv_loss'], step=idx)

                tf.summary.scalar('d_loss', losses['d/loss'], step=idx)
                tf.summary.scalar('d_adv_loss', losses['d/adv_loss'], step=idx)

                tf.summary.scalar('r1_reg', losses['r1_reg'], step=idx)
                tf.summary.scalar('pl_reg', losses['pl_reg'], step=idx)
                # tf.summary.histogram('w_avg', self.generator.w_avg, step=idx)

                if np.mod(idx, self.save_freq) == 0 or idx == self.iteration - 1:
                    tf.summary.scalar('fid', losses['fid'], step=idx)

            # save every self.save_freq
            # if np.mod(idx + 1, self.save_freq) == 0:
            #     self.manager.save(checkpoint_number=idx + 1)

            # save every self.print_freq
            if np.mod(idx + 1, self.print_freq) == 0:
                total_num_samples = min(self.n_samples, self.batch_size)
                partial_size = int(np.floor(np.sqrt(total_num_samples)))

                # prepare inputs
                latents = tf.random.normal(shape=(self.n_samples, self.g_params['z_dim']), dtype=tf.dtypes.float32)
                dummy_labels = tf.random.normal((self.n_samples, self.g_params['labels_dim']), dtype=tf.dtypes.float32)

                # run networks
                fake_img, _ = self.g_clone([latents, dummy_labels], truncation_psi=1.0, training=False)

                save_images(images=fake_img[:partial_size * partial_size, :, :, :],
                            size=[partial_size, partial_size],
                            image_path='./{}/fake_{:06d}.png'.format(self.sample_dir, idx + 1))

                x_real_concat = tf.concat(self.strategy.experimental_local_results(x_real), axis=0)
                self.truncation_psi_canvas(x_real_concat, path='./{}/fake_psi_{:06d}.png'.format(self.sample_dir, idx + 1))

            elapsed = time.time() - iter_start_time
            print(self.log_template.format(idx, self.iteration, elapsed,
                                           losses['d/loss'], losses['g/loss'], losses['r1_reg'], losses['pl_reg'], fid, best_fid, best_fid_iter))
        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

        print("LAST FID: ", fid)
        print("BEST FID: {}, {}".format(best_fid, best_fid_iter))
        print("Total train time: %4.4f" % (time.time() - start_time))

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.model_name, self.dataset_name, self.img_size, self.config)


    def calculate_FID(self):
        @tf.function
        def gen_samples_feats(test_z, test_labels, g_clone, inception_model):
            # run networks
            fake_img, _ = g_clone([test_z, test_labels], training=False)
            fake_img = adjust_dynamic_range(fake_img, range_in=(-1.0, 1.0), range_out=(0.0, 255.0), out_dtype=tf.float32)
            fake_img = tf.transpose(fake_img, [0, 2, 3, 1])
            fake_img = tf.image.resize(fake_img, [299, 299], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

            fake_img = torch_normalization(fake_img)

            feats = inception_model(fake_img)

            return feats

        @tf.function
        def get_inception_features(img, inception_model):
            feats = inception_model(img)
            return feats

        @tf.function
        def get_real_features(img, inception_model):
            feats = self.strategy.run(get_inception_features, args=(img, inception_model))
            feats = tf.concat(self.strategy.experimental_local_results(feats), axis=0)

            return feats

        @tf.function
        def get_fake_features(z, dummy_labels, g_clone, inception_model):

            feats = self.strategy.run(gen_samples_feats, args=(z, dummy_labels, g_clone, inception_model))
            feats = tf.concat(self.strategy.experimental_local_results(feats), axis=0)

            return feats

        @tf.function
        def convert_per_replica_image(nchw_per_replica_images, strategy):
            as_tensor = tf.concat(strategy.experimental_local_results(nchw_per_replica_images), axis=0)
            as_tensor = tf.transpose(as_tensor, perm=[0, 2, 3, 1])
            as_tensor = (tf.clip_by_value(as_tensor, -1.0, 1.0) + 1.0) * 127.5
            as_tensor = tf.cast(as_tensor, tf.uint8)
            as_tensor = tf.image.resize(as_tensor, [299, 299], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

            return as_tensor

        if not self.real_cache:
            real_feats = tf.zeros([0, 2048])
            """ Input Image"""
            # img_class = Image_data(self.img_size, self.g_params['z_dim'], self.g_params['labels_dim'],
            #                        self.dataset_path)
            # img_class.preprocess()
            # dataset_num = len(img_class.train_images)
            # img_slice = tf.data.Dataset.from_tensor_slices(img_class.train_images)
            #
            # img_slice = img_slice.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True, seed=777)
            # img_slice = img_slice.map(map_func=inception_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size,
            #                                                                                             drop_remainder=False)
            # img_slice = img_slice.prefetch(buffer_size=AUTOTUNE)
            # img_slice = self.strategy.experimental_distribute_dataset(img_slice)

            for img in self.fid_img_slice:
                feats = get_real_features(img, self.inception_model)
                real_feats = tf.concat([real_feats, feats], axis=0)
                print('real feats:', np.shape(real_feats)[0])

            self.real_mu = np.mean(real_feats, axis=0)
            self.real_cov = np.cov(real_feats, rowvar=False)

            with open('{}_mu_cov.pickle'.format(self.dataset_name), 'wb') as f:
                pickle.dump((self.real_mu, self.real_cov), f, protocol=pickle.HIGHEST_PROTOCOL)

            print('{} real pickle save !!!'.format(self.dataset_name))

            self.real_cache = True
            del real_feats

        fake_feats = tf.zeros([0, 2048])
        from tqdm import tqdm
        for begin in tqdm(range(0, self.fid_samples_num, self.batch_size)):
            z = tf.random.normal(shape=[self.each_batch_size, self.g_params['z_dim']])
            dummy_labels = tf.random.normal((self.each_batch_size, self.g_params['labels_dim']), dtype=tf.float32)

            feats = get_fake_features(z, dummy_labels, self.g_clone, self.inception_model)

            fake_feats = tf.concat([fake_feats, feats], axis=0)
            # print('fake feats:', np.shape(fake_feats)[0])

        fake_mu = np.mean(fake_feats, axis=0)
        fake_cov = np.cov(fake_feats, rowvar=False)
        del fake_feats

        # Calculate FID.
        m = np.square(fake_mu - self.real_mu).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(fake_cov, self.real_cov), disp=False)  # pylint: disable=no-member
        dist = m + np.trace(fake_cov + self.real_cov - 2 * s)

        return dist


    def truncation_psi_canvas(self, real_images, path):
        # prepare inputs
        reals = real_images[:self.n_samples, :, :, :]
        latents = tf.random.normal(shape=(self.n_samples, self.g_params['z_dim']), dtype=tf.dtypes.float32)
        dummy_labels = tf.random.normal((self.n_samples, self.g_params['labels_dim']), dtype=tf.dtypes.float32)

        # run networks
        fake_images_00, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.0, training=False)
        fake_images_05, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.5, training=False)
        fake_images_07, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.7, training=False)
        fake_images_10, _ = self.g_clone([latents, dummy_labels], truncation_psi=1.0, training=False)

        # merge on batch dimension: [4 * n_samples, 3, img_size, img_size]
        out = tf.concat([fake_images_00, fake_images_05, fake_images_07, fake_images_10], axis=0)

        # prepare for image saving: [4 * n_samples, img_size, img_size, 3]
        out = postprocess_images(out)

        # resize to save disk spaces: [4 * n_samples, size, size, 3]
        size = min(self.img_size, 256)
        out = tf.image.resize(out, size=[size, size], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

        # make single image and add batch dimension for tensorboard: [1, 4 * size, n_samples * size, 3]
        out = merge_batch_images(out, size, rows=4, cols=self.n_samples)

        images = cv2.cvtColor(out.astype('uint8'), cv2.COLOR_RGB2BGR)

        return cv2.imwrite(path, images)


    def test(self):
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        total_num_samples = min(self.n_samples, self.batch_size)
        partial_size = int(np.floor(np.sqrt(total_num_samples)))

        from tqdm import tqdm
        for i in tqdm(range(self.n_test)):
            z = tf.random.normal(shape=[self.batch_size, self.g_params['z_dim']])
            dummy_labels = tf.random.normal((self.batch_size, self.g_params['labels_dim']), dtype=tf.float32)
            fake_img, _ = self.g_clone([z, dummy_labels], training=False)

            save_images(images=fake_img[:partial_size * partial_size, :, :, :],
                        size=[partial_size, partial_size],
                        image_path='./{}/fake_{:01d}.png'.format(result_dir, i))

    def test_70000(self):
        result_dir = os.path.join(self.result_dir, self.model_dir)
        check_folder(result_dir)

        total_num_samples = 1
        partial_size = int(np.floor(np.sqrt(total_num_samples)))

        from tqdm import tqdm
        for i in tqdm(range(70000)):
            z = tf.random.normal(shape=[1, self.g_params['z_dim']])
            dummy_labels = tf.random.normal((1, self.g_params['labels_dim']), dtype=tf.float32)
            fake_img, _ = self.g_clone([z, dummy_labels], training=False)

            save_images(images=fake_img[:partial_size * partial_size, :, :, :],
                        size=[partial_size, partial_size],
                        image_path='./{}/fake_{:01d}.png'.format(result_dir, i))

    def draw_uncurated_result_figure(self):

        result_dir = os.path.join(self.result_dir, self.model_dir, 'paper_figure')
        check_folder(result_dir)

        seed_flag = True
        lods = [0, 1, 2, 2, 3, 3]
        seed = 3291
        rows = 3
        cx = 0
        cy = 0

        if seed_flag:
            latents = tf.cast(
                np.random.RandomState(seed).normal(size=[sum(rows * 2 ** lod for lod in lods), self.g_params['z_dim']]), tf.float32)
        else:
            latents = tf.cast(np.random.normal(size=[sum(rows * 2 ** lod for lod in lods), self.g_params['z_dim']]), tf.float32)

        dummy_labels = tf.random.normal((sum(rows * 2 ** lod for lod in lods), self.g_params['labels_dim']), dtype=tf.float32)

        images, _ = self.g_clone([latents, dummy_labels], training=False)
        images = postprocess_images(images)

        canvas = PIL.Image.new('RGB', (sum(self.img_size // 2 ** lod for lod in lods), self.img_size * rows), 'white')
        image_iter = iter(list(images))

        for col, lod in enumerate(lods):
            for row in range(rows * 2 ** lod):
                image = PIL.Image.fromarray(np.uint8(next(image_iter)), 'RGB')

                image = image.crop((cx, cy, cx + self.img_size, cy + self.img_size))
                image = image.resize((self.img_size // 2 ** lod, self.img_size // 2 ** lod), PIL.Image.ANTIALIAS)
                canvas.paste(image,
                             (sum(self.img_size // 2 ** lod for lod in lods[:col]), row * self.img_size // 2 ** lod))

        canvas.save('{}/figure02-uncurated.png'.format(result_dir))

    def draw_style_mixing_figure(self):
        result_dir = os.path.join(self.result_dir, self.model_dir, 'paper_figure')
        check_folder(result_dir)

        seed_flag = True
        src_seeds = [604, 8440, 7613, 6978, 3004]
        dst_seeds = [1336, 6968, 607, 728, 7036, 9010]

        truncation_psi = 0.7  # Style strength multiplier for the truncation trick
        truncation_cutoff = 8  # Number of layers for which to apply the truncation trick

        resolutions = self.g_params['resolutions']
        n_broadcast = len(resolutions) * 2

        style_ranges = [range(0, 4)] * 3 + [range(4, 8)] * 2 + [range(8, n_broadcast)]

        if seed_flag:
            src_latents = tf.cast(
                np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.g_params['z_dim']]) for seed in src_seeds), axis=0), tf.float32)
            dst_latents = tf.cast(
                np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.g_params['z_dim']]) for seed in dst_seeds), axis=0), tf.float32)

        else:
            src_latents = tf.cast(np.random.normal(size=[len(src_seeds), self.g_params['z_dim']]), tf.float32)
            dst_latents = tf.cast(np.random.normal(size=[len(dst_seeds), self.g_params['z_dim']]), tf.float32)

        dummy_labels = tf.random.normal((len(src_seeds), self.g_params['labels_dim']), dtype=tf.float32)

        src_images, src_dlatents = self.g_clone([src_latents, dummy_labels], truncation_cutoff=truncation_cutoff, truncation_psi=truncation_psi, training=False)
        dst_images, dst_dlatents = self.g_clone([dst_latents, dummy_labels], truncation_cutoff=truncation_cutoff, truncation_psi=truncation_psi, training=False)

        src_images = postprocess_images(src_images)
        dst_images = postprocess_images(dst_images)

        img_out_size = min(self.img_size, 256)

        src_images = tf.image.resize(src_images, size=[img_out_size, img_out_size], antialias=True, method=tf.image.ResizeMethod.BICUBIC)
        dst_images = tf.image.resize(dst_images, size=[img_out_size, img_out_size], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

        canvas = PIL.Image.new('RGB', (img_out_size * (len(src_seeds) + 1), img_out_size * (len(dst_seeds) + 1)), 'white')

        for col, src_image in enumerate(list(src_images)):
            canvas.paste(PIL.Image.fromarray(np.uint8(src_image), 'RGB'), ((col + 1) * img_out_size, 0))

        for row, dst_image in enumerate(list(dst_images)):
            canvas.paste(PIL.Image.fromarray(np.uint8(dst_image), 'RGB'), (0, (row + 1) * img_out_size))

            row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
            src_dlatents = np.asarray(src_dlatents, dtype=np.float32)
            row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]

            row_images, _ = self.g_clone([row_dlatents, dummy_labels], mapping=False, training=False)
            row_images = postprocess_images(row_images)


            for col, image in enumerate(list(row_images)):
                canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'), ((col + 1) * img_out_size, (row + 1) * img_out_size))

        canvas.save('{}/figure03-style-mixing.png'.format(result_dir))

    def draw_truncation_trick_figure(self):

        result_dir = os.path.join(self.result_dir, self.model_dir, 'paper_figure')
        check_folder(result_dir)

        seed_flag = True
        seeds = [1653, 4010]
        psis = [-1, -0.7, -0.5, 0, 0.5, 0.7, 1]

        if seed_flag:
            latents = tf.cast(
                np.concatenate(list(np.random.RandomState(seed).normal(size=[1, self.g_params['z_dim']]) for seed in seeds), axis=0), tf.float32)
        else:
            latents = tf.cast(np.random.normal(size=[len(seeds), self.g_params['z_dim']]), tf.float32)

        dummy_labels = tf.random.normal((len(seeds), self.g_params['labels_dim']), dtype=tf.float32)

        fake_images_10_, _ = self.g_clone([latents, dummy_labels], truncation_psi=-1.0, training=False)
        fake_images_05_, _ = self.g_clone([latents, dummy_labels], truncation_psi=-0.5, training=False)
        fake_images_07_, _ = self.g_clone([latents, dummy_labels], truncation_psi=-0.7, training=False)
        fake_images_00, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.0, training=False)
        fake_images_05, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.5, training=False)
        fake_images_07, _ = self.g_clone([latents, dummy_labels], truncation_psi=0.7, training=False)
        fake_images_10, _ = self.g_clone([latents, dummy_labels], truncation_psi=1.0, training=False)

        # merge on batch dimension: [7, 3, img_size, img_size]
        col_images = list([fake_images_10_, fake_images_05_, fake_images_07_, fake_images_00, fake_images_05, fake_images_07, fake_images_10])

        img_out_size = min(self.img_size, 256)

        for i in range(len(col_images)):
            col_images[i] = postprocess_images(col_images[i])
            col_images[i] = tf.image.resize(col_images[i], size=[img_out_size, img_out_size], antialias=True, method=tf.image.ResizeMethod.BICUBIC)

        canvas = PIL.Image.new('RGB', (img_out_size * len(psis), img_out_size * len(seeds)), 'white')

        for col, col_img in enumerate(col_images):
            for row, image in enumerate(col_img):
                canvas.paste(PIL.Image.fromarray(np.uint8(image), 'RGB'),
                             (col * img_out_size, row * img_out_size))

        canvas.save('{}/figure08-truncation-trick.png'.format(result_dir))
