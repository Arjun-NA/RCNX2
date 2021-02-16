import numpy as np
import random as rn
import os

import tensorflow as tf
from keras import backend as K

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
session_conf.gpu_options.allow_growth = True

tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
K.set_image_data_format('channels_last')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras import layers, models, optimizers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images, plot_log
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from capslayers import CapsuleLayer, Mask
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from resnext_keras import __create_res_next as ResNext


def mish(inputs):
    return inputs * K.tanh(K.softplus(inputs))

def level(input_shape, n_class, routings):
    """
    A 3-level Residual Capsule Network on CIFAR-10.

    """

    x = layers.Input(shape=input_shape)
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    ########################### Level 1 Capsules ###########################
    # Creating a residual block with 8 layers having 32 filters and 32 growth rate.
    #
    conv = ResNext(nb_classes=10, depth = [1], cardinality=2, width=32, include_top=False, img_input=x)

    # conv, nb_filter = resnet.ResBlock(conv, growth_rate=32, nb_layers=2, nb_filter=32, bottleneck=True)
    # Batch Normalization
    ResBlockOutput = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(conv)

    # Creating Primary Capsules (Level 1)
    # Here PrimaryCapsConv2D is the Conv2D output which is used as the primary capsules by reshaping and squashing (squash activation).
    # primarycaps_1 (size: [None, num_capsule, dim_capsule]) is the "reshaped and sqashed output" which will be further passed to the dynamic routing protocol.
    primarycaps_1, PrimaryCapsConv2D = PrimaryCap(ResBlockOutput, dim_capsule=2, n_channels=12, kernel_size=5,
                                                  strides=2, padding='valid')

    # Applying ReLU Activation to primary capsules 
    conv = layers.Activation('relu')(PrimaryCapsConv2D)

    ########################### Level 2 Capsules ###########################
    # Creating a residual block with 8 layers having 32 filters and 32 growth rate.
    conv = ResNext(nb_classes=10, depth = [1], cardinality=2, width=32, include_top=False, img_input=conv)
    # Batch Normalization
    ResBlockOutput = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(conv)

    # Creating Primary Capsules (Level 2)
    primarycaps_2, PrimaryCapsConv2D = PrimaryCap(ResBlockOutput, dim_capsule=2, n_channels=12, kernel_size=5,
                                                  strides=2, padding='valid')

    # Applying ReLU Activation to primary capsules 
    conv = layers.Activation('relu')(PrimaryCapsConv2D)

    ########################### Level 3 Capsules ###########################
    # Creating a residual block with 8 layers having 32 filters and 32 growth rate.
    conv = ResNext(nb_classes=10, depth = [1], cardinality=2, width=32, include_top=False, img_input=conv)
    # Batch Normalization
    ResBlockOutput = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(conv)

    # Creating Primary Capsules (Level 3)
    primarycaps_3, PrimaryCapsConv2D = PrimaryCap(ResBlockOutput, dim_capsule=2, n_channels=12, kernel_size=3,
                                                  strides=2, padding='valid')

    # Merging Primary Capsules for the Merged DigitCaps (CapsuleLayer formed by combining all levels of primary capsules)
    mergedLayer = layers.merge([primarycaps_1, primarycaps_2, primarycaps_3], mode='concat', concat_axis=1)

    ########################### Separate DigitCaps Outputs (used for training) ###########################
    # Merged DigitCaps
    digitcaps_0 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, channels=0,
                               name='digitcaps0')(mergedLayer)
    out_caps_0 = Length(name='capsnet_0')(digitcaps_0)

    # First Level DigitCaps
    digitcaps_1 = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, channels=0,
                               name='digitcaps1')(primarycaps_1)
    out_caps_1 = Length(name='capsnet_1')(digitcaps_1)

    # Second Level DigitCaps
    digitcaps_2 = CapsuleLayer(num_capsule=n_class, dim_capsule=12, routings=routings, channels=0,
                               name='digitcaps2')(primarycaps_2)
    out_caps_2 = Length(name='capsnet_2')(digitcaps_2)

    # Third Level DigitCaps
    digitcaps_3 = CapsuleLayer(num_capsule=n_class, dim_capsule=10, routings=routings, channels=0,
                               name='digitcaps3')(primarycaps_3)
    out_caps_3 = Length(name='capsnet_3')(digitcaps_3)

    ########################### Combined DigitCaps Output (used for evaluation) ###########################
    digitcaps = layers.merge([digitcaps_1, digitcaps_2, digitcaps_3, digitcaps_0], mode='concat', concat_axis=2,
                             name='digitcaps')
    out_caps = Length(name='capsnet')(digitcaps)

    # Reconstruction (decoder) network
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(600, activation='relu', input_dim=int(digitcaps.shape[2] * n_class), name='zero_layer'))
    decoder.add(layers.Dense(600, activation='relu', name='one_layer'))
    decoderFinal = models.Sequential(name='decoderFinal')
    # Concatenating two layers
    decoderFinal.add(layers.Merge([decoder.get_layer('zero_layer'), decoder.get_layer('one_layer')], mode='concat'))
    decoderFinal.add(layers.Dense(1200, activation='relu'))
    decoderFinal.add(layers.Dense(np.prod([32, 32, 3]), activation='sigmoid'))
    decoderFinal.add(layers.Reshape(target_shape=[32, 32, 3], name='out_recon'))

    # Model for training
    train_model = models.Model([x, y], [out_caps, out_caps_0, out_caps_1, out_caps_2, out_caps_3, decoderFinal(masked_by_y)])

    # Model for evaluation (prediction)
    # Note that out_caps is the final prediction. Other predictions could be used for analysing separate-level predictions. 
    eval_model = models.Model(x, out_caps)

    return train_model, eval_model


def margin_loss(y_true, y_pred):
    """
    Margin loss, as introduced for Capsule Networks.
    
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a 3-level Residual Capsule Network
    
    """

    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    row = x_train.shape[1]
    col = x_train.shape[2]
    channel = x_train.shape[3]

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_capsnet_acc',
                                           verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    # Notice the four separate losses (for separate backpropagations)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, margin_loss, margin_loss, margin_loss, margin_loss, 'mse'],
                  loss_weights=[1., 1., 1., 1., 1., args.lam_recon],
                  metrics=['accuracy'])

    # Training with data augmentation
    def train_generator(x, y, batch_size, shift_fraction=0.):
        train_datagen = ImageDataGenerator(width_shift_range=shift_fraction,
                                           height_shift_range=shift_fraction)  # shift up to 2 pixel for MNIST
        generator = train_datagen.flow(x, y, batch_size=batch_size)
        while 1:
            x_batch, y_batch = generator.next()
            yield ([x_batch, y_batch], [y_batch, y_batch, y_batch, y_batch, y_batch, x_batch[:, :, :, :]])

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    model.fit_generator(generator=train_generator(x_train, y_train, args.batch_size, args.shift_fraction),
                        steps_per_epoch=int(y_train.shape[0] / args.batch_size),
                        epochs=args.epochs,
                        validation_data=[[x_test, y_test], [y_test, y_test, y_test, y_test, y_test, x_test[:, :, :, :]]],
                        callbacks=[log, tb, checkpoint, lr_decay])

    # Save model weights
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data, args):
    x_test, y_test = data
    print('Testing the model...')
    y_pred = model.predict(x_test, batch_size=100)

    print('Test Accuracy (All DigitCaps): ',
          100.0 * np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1)) / (1.0 * y_test.shape[0]))
    model.save('try_save/new.h5')
    


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.
    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.
    # Returns
        Preprocessed tensor.
    """
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        if x.ndim == 3:
            # 'RGB'->'BGR'
            x = x[::-1, ...]
            # Zero-center by mean pixel
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x = x[:, ::-1, ...]
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[..., ::-1]
        # Zero-center by mean pixel
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    x *= 0.017  # scale values

    return x


def load_dataset():
    # Load the dataset from Keras
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Preprocessing the dataset
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = preprocess_input(x_train)
    x_test = preprocess_input(x_test)
    x_train = x_train.reshape(-1, 32, 32, 3).astype('float32')
    x_test = x_test.reshape(-1, 32, 32, 3).astype('float32')
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import argparse
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="3-level Residual Capsule Network on CIFAR-10.")
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.512, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    (x_train, y_train), (x_test, y_test) = load_dataset()

    # define model
    model, eval_model = level(input_shape=x_train.shape[1:],
                              n_class=len(np.unique(np.argmax(y_train, 1))),
                              routings=args.routings)
    model.summary()
    eval_model.summary()
    # train or test
    # args.weights = 'result/trained_model.h5'
    if args.weights is not None:  # init the model weights with provided one
        model.load_weights(args.weights)
    if not args.testing:
        train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    else:  # as long as weights are given, will run testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')
        test(model=eval_model, data=(x_test, y_test), args=args)
