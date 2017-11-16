"""Keras based implementation of Autoencoding beyond pixels using a learned similarity metric

References:

Autoencoding beyond pixels using a learned similarity metric
by: Anders Boesen Lindbo Larsen, Soren Kaae Sonderby, Hugo Larochelle, Ole Winther
https://arxiv.org/abs/1512.09300

Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Alec Radford, Luke Metz, Soumith Chintala
https://arxiv.org/abs/1511.06434
"""
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import cv2
import tensorflow as tf
# from utils import load, save
from  models.utils import load, save
# from models.layers import Deconv2D
from keras.layers import Deconv2D, Conv2D
# from layers import Deconv2D
from keras import backend as K
from keras.layers import Input, Dense, Reshape, Activation, Conv2D, LeakyReLU, Flatten, BatchNormalization as BN
from keras.models import Sequential, Model
# from keras import initializations
import keras.initializers as initializers
from functools import partial
#for Spherical interpolation
from models.interpolate import slerp
learning_rate = .0002
beta1 = .5
z_dim = 512
normal = initializers.RandomNormal


def cleanup(data):
    X = data[0][:64, -1]
    X = np.asarray([cv2.resize(x.transpose(1, 2, 0), (160, 80)) for x in X], dtype=np.float32)
    X = X / 127.5 - 1.
    Z = np.random.normal(0, 1, (X.shape[0], z_dim))
    return Z, X


def generator(batch_size, gf_dim, ch, rows, cols):
    model = Sequential()

    model.add(Dense(gf_dim * 8 * rows[0] * cols[0], batch_input_shape=(batch_size, z_dim), name="g_h0_lin",
                    kernel_initializer=normal(mean=0.0, stddev=0.02)))  # output size 25600
    model.add(Reshape((rows[0], cols[0], gf_dim * 8)))  # 5 , 10 , 512
    model.add(BN(axis=3, name="g_bn0", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim * 4, (5, 5), strides=(2, 2), padding='same', name="g_h1",
                       kernel_initializer=normal(mean=0.0, stddev=0.02)))
    model.add(BN(axis=3, name="g_bn1", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim * 2, (5, 5), strides=(2, 2), padding='same', name="g_h2",
                       kernel_initializer=normal(mean=0.0, stddev=0.02)))
    model.add(BN(axis=3, name="g_bn2", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(gf_dim, (5, 5), strides=(2, 2), padding='same', name="g_h3",
                       kernel_initializer=normal(mean=0.0, stddev=0.02)))
    model.add(BN(axis=3, name="g_bn3", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5))
    model.add(Activation("relu"))

    model.add(Deconv2D(ch, (5, 5), strides=(2, 2), padding='same', name="g_h4",
                       kernel_initializer=normal(mean=0.0, stddev=0.02)))
    model.add(Activation("tanh"))
    print(model.summary())

    return model


def encoder(batch_size, df_dim, ch, rows, cols):
    model = Sequential()
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch), name='encoder_x')
    model = Conv2D(df_dim, (5, 5), strides=(2, 2), padding="same",
                   name="e_h0_conv", data_format="channels_last", kernel_initializer=normal(mean=0.0, stddev=0.02))(X)
    model = LeakyReLU(.2)(model)

    model = Conv2D(df_dim * 2, (5, 5), strides=(2, 2), padding="same",
                   name="e_h1_conv", data_format="channels_last")(model)
    model = BN(axis=3, name="e_bn1", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5)(model)

    model = LeakyReLU(.2)(model)

    model = Conv2D(df_dim * 4, (5, 5), strides=(2, 2), name="e_h2_conv", padding="same",
                   data_format="channels_last", kernel_initializer=normal(mean=0.0, stddev=0.02))(model)
    model = BN(axis=3, name="e_bn2", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Conv2D(df_dim * 8, (5, 5), strides=(2, 2), padding="same",
                   name="e_h3_conv", data_format="channels_last", kernel_initializer=normal(mean=0.0, stddev=0.02))(
        model)
    model = BN(axis=3, name="e_bn3", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)
    model = Flatten()(model)

    mean = Dense(z_dim, name="e_h3_lin", kernel_initializer=normal(mean=0.0, stddev=0.02))(model)
    logsigma = Dense(z_dim, name="e_h4_lin", activation="tanh", kernel_initializer=normal(mean=0.0, stddev=0.02))(model)
    meansigma = Model([X], [mean, logsigma])
    print(meansigma.summary())
    return meansigma


def discriminator(batch_size, df_dim, ch, rows, cols):
    X = Input(batch_shape=(batch_size, rows[-1], cols[-1], ch), name='discriminator_x')
    model = Conv2D(df_dim, (5, 5), strides=(2, 2), padding="same",
                   batch_input_shape=(batch_size, rows[-1], cols[-1], ch),
                   name="d_h0_conv", data_format="channels_last", kernel_initializer=normal(mean=0.0, stddev=0.02))(X)
    model = LeakyReLU(.2)(model)

    model = Conv2D(df_dim * 2, (5, 5), strides=(2, 2), padding="same",
                   name="d_h1_conv", data_format="channels_last", kernel_initializer=normal(mean=0.0, stddev=0.02))(
        model)
    model = BN(axis=3, name="d_bn1", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Conv2D(df_dim * 4, (5, 5), strides=(2, 2), padding="same",
                   name="d_h2_conv", data_format="channels_last", kernel_initializer=normal(mean=0.0, stddev=0.02))(
        model)
    model = BN(axis=3, name="d_bn2", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5)(model)
    model = LeakyReLU(.2)(model)

    model = Conv2D(df_dim * 8, (5, 5), strides=(2, 2), padding="same",
                   name="d_h3_conv", data_format="channels_last", kernel_initializer=normal(mean=0.0, stddev=0.02))(
        model)

    dec = BN(axis=3, name="d_bn3", gamma_initializer=normal(mean=1., stddev=0.02), epsilon=1e-5)(model)
    dec = LeakyReLU(.2)(dec)
    dec = Flatten()(dec)
    dec = Dense(1, name="d_h3_lin", kernel_initializer=normal(mean=0.0, stddev=0.02))(dec)

    output = Model([X], [dec, model])
    print(output.summary())

    return output


def get_model(sess, image_shape=(80, 160, 3), gf_dim=64, df_dim=64, batch_size=64,
              name="autoencoder", gpu=0):
    K.set_session(sess)
    checkpoint_dir = './outputs/results_' + name
    with tf.variable_scope(name), tf.device("/gpu:{}".format(gpu)):

        # sizes
        ch = image_shape[2]
        rows = [int(image_shape[0] / i) for i in [16, 8, 4, 2, 1]]
        cols = [int(image_shape[1] / i) for i in [16, 8, 4, 2, 1]]

        # build
        G = generator(batch_size, gf_dim, ch, rows, cols)
        E = encoder(batch_size, df_dim, ch, rows, cols)
        D = discriminator(batch_size, df_dim, ch, rows, cols)
        Z2 = Input(batch_shape=(batch_size, z_dim), name='more_noise')
        # init
        # initialize bug
        # def init_updates(M):
        #   for t in M.updates:
        #     t.initialized_value()
        #
        # # sess.run(tf. initialize_all_variables())
        # [init_updates(m) for m in [G ,E,D]]

        G.compile("sgd", "mse")
        g_vars = G.trainable_weights
        print("G.shape: ", G.output_shape)

        E.compile("sgd", "mse")
        e_vars = E.trainable_weights
        print("E.shape: ", E.output_shape)

        D.compile("sgd", "mse")
        d_vars = D.trainable_weights
        print("D.shape: ", D.output_shape)

        Z = G.input
        Img = D.input
        G_train = G(Z)
        E_mean, E_logsigma = E(Img)
        G_dec = G(E_mean + Z2 * E_logsigma)
        D_fake, F_fake = D(G_train)
        D_dec_fake, F_dec_fake = D(G_dec)
        D_legit, F_legit = D(Img)

        # costs
        recon_vs_gan = 1e-6
        like_loss = tf.reduce_mean(tf.square(F_legit - F_dec_fake)) / 2.
        kl_loss = tf.reduce_mean(-E_logsigma + .5 * (-1 + tf.exp(2. * E_logsigma) + tf.square(E_mean)))

        d_loss_legit = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_legit, labels=tf.ones_like(D_legit)))
        d_loss_fake1 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
        d_loss_fake2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_dec_fake, labels=tf.zeros_like(D_dec_fake)))
        d_loss_fake = d_loss_fake1 + d_loss_fake2
        d_loss = d_loss_legit + d_loss_fake

        g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
        g_loss2 = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_dec_fake, labels=tf.ones_like(D_dec_fake)))
        g_loss = g_loss1 + g_loss2 + recon_vs_gan * like_loss
        e_loss = kl_loss + like_loss

        # optimizers
        print("Generator variables:")
        for v in g_vars:
            print(v.name)
        print("Discriminator variables:")
        for v in d_vars:
            print(v.name)
        print("Encoder variables:")
        for v in e_vars:
            print(v.name)

        e_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(e_loss, var_list=e_vars)
        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    # summaries
    sum_d_loss_legit = tf.summary.scalar("d_loss_legit", d_loss_legit)
    sum_d_loss_fake = tf.summary.scalar("d_loss_fake", d_loss_fake)
    sum_d_loss = tf.summary.scalar("d_loss", d_loss)
    sum_g_loss = tf.summary.scalar("g_loss", g_loss)
    sum_e_loss = tf.summary.scalar("e_loss", e_loss)
    sum_e_mean = tf.summary.histogram("e_mean", E_mean)
    sum_e_sigma = tf.summary.histogram("e_sigma", tf.exp(E_logsigma))
    sum_Z = tf.summary.histogram("Z", Z)
    sum_gen = tf.summary.image("G", G_train)
    sum_dec = tf.summary.image("E", G_dec)

    # saver
    saver = tf.train.Saver()
    g_sum = tf.summary.merge([sum_Z, sum_gen, sum_d_loss_fake, sum_g_loss])
    e_sum = tf.summary.merge([sum_dec, sum_e_loss, sum_e_mean, sum_e_sigma])
    d_sum = tf.summary.merge([sum_d_loss_legit, sum_d_loss])
    writer = tf.summary.FileWriter("/tmp/logs/" + name, sess.graph)

    #interplo
    def get_interpolation(codes, N, z_dim):
        shape = codes.shape
        code_a = codes[:int(shape[0] / 2)]
        code_a = np.reshape(code_a, [-1, 1, z_dim])
        code_b = codes[int(shape[0] / 2):]
        code_b = np.reshape(code_b, [-1, 1, z_dim])
        alpha = np.reshape(np.linspace(0., 1., N), [1, N, 1])
        z_inter = alpha * code_a + (1 - alpha) * code_b
        z_inter = np.reshape(z_inter, [-1, z_dim])

        return z_inter
    def get_slerp(codes, N, z_dim):
        shape = codes.shape
        code_a = codes[:int(shape[0] / 2)]
        code_a = np.reshape(code_a, [-1,  z_dim])
        code_b = codes[int(shape[0] / 2):]
        code_b = np.reshape(code_b, [-1,  z_dim])
        z_inter = []
        for low ,high in zip(code_a , code_b):
            for alpha in np.linspace(0., 1., N):
                interp_point = slerp(alpha, low , high)
                z_inter.append(interp_point)
        z_inter = np.asarray(z_inter)
        z_inter = np.reshape(z_inter, [-1, z_dim])
        return z_inter




    # functions
    def train_d(images, z, counter, sess=sess):
        z2 = np.random.normal(0., 1., z.shape)
        outputs = [d_loss, d_loss_fake, d_loss_legit, d_sum, d_optim]
        # with tf.control_dependencies(outputs):
        #   updates = [tf.assign(p, new_p) for p, new_p in zip(D.updates[::2] , D.updates[1::2])]
        # updates = [tf.assign(p, new_p) for (p, new_p) in D.updates]
        outs = sess.run(outputs, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        # outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        dl, dlf, dll, sums = outs[:4]
        writer.add_summary(sums, counter)
        return dl, dlf, dll

    def train_g(images, z, counter, sess=sess, mode = None):
        #inter
        if mode == 'inter':
            N = 20
            code_mean, code_mean_sigma = sess.run([E_mean, E_logsigma],feed_dict={Img: images,  K.learning_phase(): 1})
            noise = np.random.normal(0., 1., code_mean_sigma.shape)
            z = code_mean + noise*code_mean_sigma
            # z_int = get_interpolation(z, N, z_dim)
            z_int = get_slerp(z, N, z_dim)
            out_img  = sess.run(G_train , feed_dict = {Z: z_int, K.learning_phase(): 1} )
            return out_img
        # generator
        z2 = np.random.normal(0., 1., z.shape)
        outputs = [g_loss, G_train, g_sum, g_optim]
        # with tf.control_dependencies(outputs):
        #   updates = [tf.assign(p, new_p) for p, new_p in zip(G.updates[::2] , G.updates[1::2])]
        # outs = sess.run(outputs + updates, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        outs = sess.run(outputs, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        gl, samples, sums = outs[:3]
        writer.add_summary(sums, counter)
        # encoder
        outputs = [e_loss, G_dec, e_sum, e_optim]
        # with tf.control_dependencies(outputs):
        #   updates = [tf.assign(p, new_p) for p, new_p in zip(E.updates[::2] , E.updates[1::2])]
        try:
            outs = sess.run(outputs, feed_dict={Img: images, Z: z, Z2: z2, K.learning_phase(): 1})
        except:
            import IPython as IP;
            IP.embed()
        gl, samples, sums = outs[:3]
        writer.add_summary(sums, counter)

        return gl, samples, images

    def f_load():
        try:
            load(sess, saver, checkpoint_dir, name)



        except Exception as e:
            print(e)
            load_keras_model()


    def load_keras_model():
        print('load keras')
        G.load_weights(checkpoint_dir + "/G_weights.keras")
        D.load_weights(checkpoint_dir + "/D_weights.keras")
        E.load_weights(checkpoint_dir + "/E_weights.keras")
    def f_save(step):
        save(sess, saver, checkpoint_dir, step, name)
        G.save_weights(checkpoint_dir + "/G_weights.keras", True)
        D.save_weights(checkpoint_dir + "/D_weights.keras", True)
        E.save_weights(checkpoint_dir + "/E_weights.keras", True)

    def sampler(z, x):
        code = E.predict(x, )[0]
        out = G.predict(code, )
        return out, x

    return train_g, train_d, sampler, f_save, f_load, [G, D, E]
