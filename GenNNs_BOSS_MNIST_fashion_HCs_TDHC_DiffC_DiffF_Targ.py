import tensorflow as tf
from keras.utils import np_utils
import glob
#import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
#import PIL
from tensorflow.keras import layers
import timeit
import time

#from skimage.measure import compare_ssim
from skimage.measure import compare_ssim

import pickle

import pydot
import graphviz

from numpy import linalg as LA

from IPython import display

from tensorflow.keras import Input, Model

from keras.models import load_model



#####################################################################
################### get_S_function
#####################################################################
def get_S_function(M,r):
    '''

    :param M: is the number of labels
    :param r: the labels we want to take out
    :return: [M]\r, where [M] := {0,1,2,...,M-1}
    '''
    all_M = np.arange(0, M)
    return np.setdiff1d(all_M,r)
#########################################################################################################################################


#####################################################################
################### get_S_function
#####################################################################
def sup_lbl_from_lbl_2(input_lbl):


    if input_lbl==0 or input_lbl==2 or  input_lbl==6:
        sup_lbl_from_lbl = 0
    elif input_lbl==1 or input_lbl==5  or input_lbl==7 or input_lbl==9:
        sup_lbl_from_lbl = 1
    elif input_lbl == 3 or input_lbl == 4 or input_lbl == 8:
        sup_lbl_from_lbl = 2

    return sup_lbl_from_lbl
#########################################################################################################################################


#########################################################################################################################################
############################### relu_scaler_Ismail  ######
#########################################################################################################################################
def relu_scaler_Ismail(x):
    '''
    :param x: scaler
    :return: y=relu(x)
    '''
    y=0
    if x >= 0:
        y=x
    else:
        y=0
    return y
#########################################################################################################################################


#########################################################################################################################################
############################### SSIM fucntion  ######
#########################################################################################################################################
def SSIM_index(imageA, imageB):

    imageA = imageA.reshape(28, 28)
    imageB = imageB.reshape(28, 28)

    # rho_inf = LA.norm(input_image.reshape(784, 1) - X_test_pert[idx].reshape(784, 1) , np.inf)
    (D_s, diff) = compare_ssim(imageA, imageB, full=True)
    return D_s
#########################################################################################################################################



#########################################################################################################################################
############################### target coarser  ######
#########################################################################################################################################

targeted_super_label_confer            = pickle.load(open("/home/ismail/pycharmProjects/SSLTL_project/RL_adv_attacks_LP/targeted_super_label_confer.p","rb"))

#########################################################################################################################################


##########################################################################################
############################### jensen shannon divergence fucntion -  ######
##########################################################################################
"""
it is a normalized and stable version of the KL divergence and return values between [0,1] where 0 is two identical distributions
"""

from scipy.spatial.distance import jensenshannon

from math import log2
def D_JS_PMFs(p, q):
    # D_JS_PMFs(p,q) = D_JS_PMFs(q,p)
    return jensenshannon(p, q, base=2)


####################################################################################
################################ some dataset - MNIST fashion
####################################################################################

# download mnist data and split into train and test sets
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# reshape data to fit model
X_train = train_images.reshape(train_images.shape[0], 28, 28, 1)
X_test = test_images.reshape(test_images.shape[0], 28, 28, 1)
X_train, X_test = X_train/255, X_test/255
# normalization:
train_images = train_images / 255
test_images = test_images / 255
y_test = np_utils.to_categorical(test_labels,10)
# ###############################################


####################################################################################
################################ load trained model(s) and freeze
####################################################################################

# trained model for the MNIST fashion used for the FHC with 1D input and \in [0,1]
#trained_model = tf.keras.models.load_model('my_model_1d_last_dense_activation_seperate')
trained_model_coarser = tf.keras.models.load_model('coarser_model_1d')

# test the CA of the model = check in IJCNN paper
#results = trained_model.evaluate(X_test, y_test)
#print("test loss, test acc:", results)

######## freeze trained_model
for layer in trained_model_coarser.layers:
    layer.trainable = False

number_of_observations = 3
succ=0
D_ssim_images_save = []
D_JS_save          = []
run_time_save      = []

BOSS_images = []

for idx in range(number_of_observations):
    start = timeit.default_timer()
    ########################################################################################
    ######## Get the trained finer which is here based on the PREDCTED COARSER
    ########################################################################################

    ### define X_d as the desired
    X_desired = X_test[idx]

    finer_1_model_1d = tf.keras.models.load_model('finer_1_model_1d')
    finer_2_model_1d = tf.keras.models.load_model('finer_2_model_1d')
    finer_3_model_1d = tf.keras.models.load_model('finer_3_model_1d')

    #predicted_coarser_of_clean_image = np.argmax(trained_model_coarser(X_desired.reshape(1,784,1)).numpy())


    ### the choice is made here as: if the coarser is different, then any finer will suffice
    if targeted_super_label_confer[idx] == 0:
        trained_model_finer = finer_1_model_1d
        nnumber_of_classes_finer = 3
        target_lbl_finer = np.random.choice(nnumber_of_classes_finer)
    if targeted_super_label_confer[idx] == 1:
        trained_model_finer = finer_2_model_1d
        nnumber_of_classes_finer = 4
        target_lbl_finer = np.random.choice(nnumber_of_classes_finer)
    if targeted_super_label_confer[idx] == 2:
        trained_model_finer = finer_3_model_1d
        nnumber_of_classes_finer = 3
        target_lbl_finer = np.random.choice(nnumber_of_classes_finer)

    for layer in trained_model_finer.layers:
        layer.trainable = False
        layer._name = layer.name + str("_finer")


    ###########################################################################
    #################################### BUILDING THE gen model g(z,\phi)
    ###########################################################################

    gen_NN = tf.keras.Sequential()
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=102)
    ## ADDING THE GEN MODEL layers that will be trained

    layer = layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,), name='dense_gen', kernel_initializer=initializer)
    layer.trainable=True
    gen_NN.add(layer)

    layer = layers.BatchNormalization()
    layer.trainable=True
    gen_NN.add(layer)

    layer = layers.LeakyReLU()
    layer.trainable=True
    gen_NN.add(layer)

    layer = layers.Reshape((7, 7, 256),name='reshape_gen')
    layer.trainable=True
    gen_NN.add(layer)
    #assert combined_NN.output_shape == (None, 7, 7, 256)

    layer = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=initializer)
    layer.trainable=True
    gen_NN.add(layer)
    #assert gen_NN.output_shape == (None, 14, 14, 64)

    layer = layers.BatchNormalization()
    layer.trainable=True
    gen_NN.add(layer)

    layer = layers.LeakyReLU()
    layer.trainable=True
    gen_NN.add(layer)

    layer = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer)
    layer.trainable=True
    gen_NN.add(layer)
    #assert gen_NN.output_shape == (None, 14, 14, 64)

    layer = layers.BatchNormalization()
    layer.trainable=True
    gen_NN.add(layer)

    layer = layers.LeakyReLU()
    layer.trainable=True
    gen_NN.add(layer)

    layer = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid', kernel_initializer=initializer)
    layer.trainable=True
    gen_NN.add(layer)


    # below is added for the 1D modification
    layer = layers.Reshape((784, 1, 1),name='reshape_gen_final')
    layer.trainable=True
    gen_NN.add(layer)




    #########################################################################################################################################
    ############################### this is NOT sequentiail traning (traning two loss fucntions from different heads of the NN) ######
    #########################################################################################################################################


    # these two need to be have the same values as of now since X_train is the same for both
    batch_size_gen = 80
    batch_size_2 = 80


    #################### training steps and stopping thresholds
    delta_s  = 0.25

    delta_js = 0.35
    delta_ssim = 0.80

    delta_c = 0.25

    traning_steps = 20

    ############################################################
    ###  automated desired for confidence reduction y_d (desired PMF)
    ################################################################

    nnumber_of_classes_corser  = 3

    #target_class_coarser = np.argmax(trained_model_coarser(X_desired.reshape(1, 784, 1)).numpy()[0])
    target_class_coarser = targeted_super_label_confer[idx]

    #target_class_finer   = np.argmax(trained_model_finer(X_desired.reshape(1, 784, 1)).numpy()[0])
    target_class_finer   = target_lbl_finer

    desired_confidence_coarser =  1

    desired_confidence_finer   =  1

    #1 code the confidence only
    desired_PMF_confidence_coarser = np.zeros(shape=(1, nnumber_of_classes_corser ))

    desired_PMF_confidence_finer   = np.zeros(shape=(1, nnumber_of_classes_finer  ))

    for i in range(nnumber_of_classes_corser):
        if i == target_class_coarser:
            desired_PMF_confidence_coarser[:,i] = desired_confidence_coarser
        else:
            desired_PMF_confidence_coarser[:,i] = (1-desired_confidence_coarser) / (nnumber_of_classes_corser-1)

    for i in range(nnumber_of_classes_finer):
        if i == target_class_finer:
            desired_PMF_confidence_finer[:,i] = desired_confidence_finer
        else:
            desired_PMF_confidence_finer[:,i] = (1-desired_confidence_finer) / (nnumber_of_classes_finer-1)


    ################################################################
    ##### X_train is the same for both gen and combined models #####
    ################################################################
    # build x_train as some random input and y_train to be the desired image
    # X_train is the same as z in the paper

    # create one vector and repeat
    X_train_one = tf.random.uniform(shape=[1,100], minval=0., maxval=1., seed=101)
    X_train_one_np = X_train_one.numpy()
    X_train_np = np.zeros(shape=(batch_size_gen,100))
    for i in range(batch_size_gen):
        X_train_np[i,:] = X_train_one_np
    X_train = tf.convert_to_tensor(X_train_np, dtype=tf.float32)
    X_val_np = X_train_one_np
    X_val = tf.convert_to_tensor(X_val_np, dtype=tf.float32)


    ############################################################
    ### Y_train_gen for the gen model (whcih is the image)
    ################################################################

    # below is for the 1D image
    Y_train_np_gen = np.zeros(shape=(batch_size_gen,784,1,1))
    Y_val_gen = X_desired.reshape(1,784,1,1)
    for i in range(batch_size_gen):
        Y_train_np_gen[i,:,:,:] = Y_val_gen.reshape(784,1,1)
    # convert Y_train to tf eager tensor
    Y_train_gen = tf.convert_to_tensor(Y_train_np_gen, dtype=tf.float32)


    ######################################################################################
    ### Y_train_combined is the y_d (desired PMF) - HERE WE NEED FOR COARSER AND FINER
    ######################################################################################

    Y_train_combined_coarser = np.zeros(shape=(batch_size_2,nnumber_of_classes_corser))
    Y_val_combined_coarser = desired_PMF_confidence_coarser

    for i in range(batch_size_2):
        Y_train_combined_coarser[i,:] = Y_val_combined_coarser

    Y_desired_coarser = Y_val_combined_coarser[0]


    Y_train_combined_finer = np.zeros(shape=(batch_size_2, nnumber_of_classes_finer))
    Y_val_combined_finer = desired_PMF_confidence_finer

    for i in range(batch_size_2):
        Y_train_combined_finer[i, :] = Y_val_combined_finer

    Y_desired_finer = Y_val_combined_finer[0]



    print('break')


    ####################################################################################################################
    ### defining the combined model such that its the concatenation of g, then f ==> this is defing model h in the paper
    #############################################################################################################################

    input = Input(shape=100)

    x = gen_NN.layers[0](input)
    for lay in range(len(gen_NN.layers) - 1):
        layer = gen_NN.layers[lay+1]
        layer.trainable = True
        x = layer(x)
    out_1 = x


    x_2 = trained_model_coarser.layers[0](x)
    for lay in range(len(trained_model_coarser.layers) - 1):
        layer = trained_model_coarser.layers[lay + 1]
        layer.trainable = False
        x_2 = layer(x_2)
    out_2 = x_2

    x_3 = trained_model_finer.layers[0](x)
    for lay in range(len(trained_model_finer.layers) - 1):
        layer = trained_model_finer.layers[lay + 1]
        layer.trainable = False
        #layer._name = layer.name + str("_finer")
        x_3 = layer(x_3)
    out_3 = x_3




    ### defining the model: this is h(z,\psi)
    combined_NN = Model(input, [out_1, out_2 , out_3])

    ### defning the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

    loss_1 = tf.keras.losses.MeanSquaredError(name='LOSS_1')

    loss_2 = tf.keras.losses.CategoricalCrossentropy(name='LOSS_2', from_logits=False,label_smoothing=0)

    loss_3 = tf.keras.losses.CategoricalCrossentropy(name='LOSS_3', from_logits=False, label_smoothing=0)

    dynamic_weights_selection = True
    # initial losses functions weights
    lambda_gen = 1
    lambda_pmf_c = 0.01
    lambda_pmf_f = 0.01

    ############# trainING LOOP
    for i in range(traning_steps):


        combined_NN.compile(optimizer=optimizer, loss=[loss_1, loss_2, loss_3], loss_weights=[lambda_gen, lambda_pmf_c, lambda_pmf_f])
        # for lay in range(18):
        #     if lay >= 12:
        #         layer = combined_NN.layers[lay]
        #         layer.trainable = False

        # traning
        combined_NN.fit(X_train,[Y_train_gen, Y_train_combined_coarser, Y_train_combined_finer],
                        epochs=1,
                        batch_size=1,
                        validation_data=(X_val, [Y_val_gen,Y_val_combined_coarser, Y_val_combined_finer]),
                        verbose=0 )
        #combined_NN.train_on_batch(X_train, [Y_train_gen, Y_train_combined])
        # fake image at step i ==> this is X in the paper and X_val is z in the paper
        fake_image = combined_NN(X_val)[0].numpy().reshape(28, 28)


        #trained_model = load_model("MNIST_digits_trained_model_3.h5")
        trained_model_coarser = tf.keras.models.load_model('coarser_model_1d')

        output_vector_probabilities_c = trained_model_coarser(fake_image.reshape(1, 28, 28, 1)).numpy()[0]
        output_vector_probabilities_f = trained_model_finer(fake_image.reshape(1, 28, 28, 1)).numpy()[0]
        #output_vector_probabilities = combined_NN(X_val)[1].numpy().reshape(10,)

        # D_2 distance between real image and fake image at step i==> this is equation (9)
        #D_2_s = LA.norm(X_desired.reshape(784,) - fake_image.reshape(784,),          2)
        # SSIM distance between real image and fake image at step i ==>
        D_ssim_images = SSIM_index(X_desired, fake_image)
        # D_2 distance between desired PMF and the PMF returned by the fake image ==> this is equation (3)
        #D_2 = LA.norm(output_vector_probabilities_c-Y_desired,                  2       )
        # D_JS: JS divergence distance between desired and actual PMFs (it uses KL divergence)
        D_JS_c = D_JS_PMFs(output_vector_probabilities_c, Y_desired_coarser)
        D_JS_f = D_JS_PMFs(output_vector_probabilities_f,   Y_desired_finer)

        ### THE STOPPING EXIT CRITERIA
        if D_ssim_images >= delta_ssim  and D_JS_c <= delta_js and D_JS_f <= delta_js:
            #print('BREAKING FOR IS USED with Distance SSIM = ', D_ssim_images, ' and D_JS = ', D_JS)
            break

        ### logger:
        print('training step = ', i, '; image SSIM = ', D_ssim_images, ' ; PMF_JS_Distance = ', [D_JS_c,D_JS_f], ' ; current loss weights = ', lambda_gen,' , ', [lambda_pmf_c,lambda_pmf_f] )

        ##### dynamic weight selection option in training
        if dynamic_weights_selection is True:
            lambda_gen   = relu_scaler_Ismail(lambda_gen       -   0.01 * 1    * ((D_ssim_images/delta_ssim)) * np.sign((D_ssim_images/delta_ssim)-1))
            lambda_pmf_c = relu_scaler_Ismail(lambda_pmf_c     -   0.01 * 0.01 * ((delta_js/D_JS_c))          * np.sign((delta_js/D_JS_c         )-1))
            lambda_pmf_f = relu_scaler_Ismail(
                lambda_pmf_f - 0.01 * 0.01 * ((delta_js / D_JS_f)) * np.sign((delta_js / D_JS_f) - 1))

        else:
            lambda_gen = 1
            lambda_pmf = 0.01


        ### SAVE THE DISTNCE AND PERTURBED IMAGE SO AS TO TAKE THE MINIMUM AT THE END OF THE TRAINING STEP (THIS IS TO OVER COME OVERFITTING DURING TRAINING)



    fake_image = combined_NN(X_val)[0].numpy().reshape(28,28)

    ### below is the same thing (just for sanity check)
    #trained_model = load_model("MNIST_digits_trained_range_1to1_1d_input.h5")
    trained_model_coarser = tf.keras.models.load_model('coarser_model_1d')
    output_vector_probabilities_c   = trained_model_coarser(fake_image.reshape(1,28,28,1)).numpy()[0]
    output_vector_probabilities_f   = trained_model_finer(fake_image.reshape(1, 28, 28, 1)).numpy()[0]

    real_image = X_desired.reshape(28,28)

    confidence_score      = np.max(trained_model_coarser(X_desired.reshape(1,28,28,1)).numpy()[0])

    confidence_score_BOSS_coarser = np.max(trained_model_coarser(fake_image.reshape(1,28,28,1)).numpy()[0])

    confidence_score_BOSS_finer   = np.max(trained_model_finer(fake_image.reshape(1, 28, 28, 1)).numpy()[0])

    #### #### passing criteria: prediction_coarser(X_BOSS) = target coarser AND
    #### #### prediction_finer(X_BOSS) = target_finer
    if np.argmax(trained_model_coarser(fake_image.reshape(1, 784, 1)).numpy()[0]) == target_class_coarser and np.argmax(trained_model_finer(fake_image.reshape(1, 784, 1)).numpy()[0]) == target_lbl_finer:
        succ = succ + 1

    #### outer loop logger:
    print('[index,succ] = ', [idx, succ], ' - [prediction C,F] = ', [np.argmax(trained_model_coarser(fake_image.reshape(1, 784, 1)).numpy()[0]), target_class_coarser], [np.argmax(trained_model_finer(fake_image.reshape(1, 784, 1)).numpy()[0]),target_lbl_finer],
          ' - [SSIM,JS] = ', [D_ssim_images, D_JS_c, D_JS_f])


    D_ssim_images_save.append(D_ssim_images)
    D_JS_save.append([D_JS_c,D_JS_f])
    BOSS_images.append(fake_image)
    stop = timeit.default_timer()
    run_time_save.append(stop - start)

    #print('Time: ', stop - start)


print('OVERALL PERFORMANCE: ', 'succ = ' , 100*(succ/number_of_observations) , 'AVG_SSIM_JS = ',[np.mean(D_ssim_images_save),np.mean(D_JS_save)], ' AVG runtime = ',np.mean(run_time_save))



# plt.figure()
# plt.subplot(3,2,1)
# plt.title('Desired example')
# plt.imshow(X_desired,cmap='gray',vmin=0, vmax=1)
# plt.colorbar()
# plt.axis('off')
#
# plt.subplot(3,2,2)
# plt.title('Generated example')
# plt.imshow(fake_image,cmap='gray',vmin=0, vmax=1)
# plt.colorbar()
# plt.axis('off')
#
# plt.subplot(3,2,3)
# plt.title('PMF_coarser')
# plt.stem(output_vector_probabilities_c)
# plt.ylim(top=1.2)
#
# plt.subplot(3,2,4)
# plt.title('desired PMF_coarser')
# plt.stem(desired_PMF_confidence_coarser[0])
# plt.ylim(top=1.2)
#
# plt.subplot(3,2,5)
# plt.title('PMF_finer')
# plt.stem(output_vector_probabilities_f)
# plt.ylim(top=1.2)
#
# plt.subplot(3,2,6)
# plt.title('desired PMF_finer')
# plt.stem(desired_PMF_confidence_finer[0])
# plt.ylim(top=1.2)
#

print('break')



