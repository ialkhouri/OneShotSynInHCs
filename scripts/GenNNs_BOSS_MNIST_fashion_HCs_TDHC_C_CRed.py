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



#########################################################################################
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

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# ###############################################


####################################################################################
################################ load trained model(s) and freeze
####################################################################################

# load NDC binary classifiers
C_Other_Not   = load_model("MNIST_fashion_BOSS_ND_C_Other_Not_2.h5"  )
C_top_bottom  = load_model("MNIST_fashion_BOSS_ND_C_top_bottom_2.h5" )
C_Bag_Not     = load_model("MNIST_fashion_BOSS_ND_C_Bag_Not_2.h5"    )
C_dress_coat  = load_model("MNIST_fashion_BOSS_ND_C_dress_coat_2.h5" )
C_Touser_Not  = load_model("MNIST_fashion_BOSS_ND_C_Touser_Not_2.h5" )
C_Sandal_Not  = load_model("MNIST_fashion_BOSS_ND_C_Sandal_Not_2.h5" )
C_Tshirt_Not  = load_model("MNIST_fashion_BOSS_ND_C_Tshirt_Not_2.h5" )
C_Pull_Shirt  = load_model("MNIST_fashion_BOSS_ND_C_Pull_Shirt_2.h5" )
C_Ank_Sneaker = load_model("MNIST_fashion_BOSS_ND_C_Ank_Sneaker_2.h5")

######## freeze trained_models
for layer in C_Other_Not.layers:
    layer.trainable = False
for layer in C_top_bottom.layers:
    layer.trainable = False
for layer in C_Bag_Not.layers:
    layer.trainable = False
for layer in C_dress_coat.layers:
    layer.trainable = False
for layer in C_Touser_Not.layers:
    layer.trainable = False
for layer in C_Sandal_Not.layers:
    layer.trainable = False
for layer in C_Tshirt_Not.layers:
    layer.trainable = False
for layer in C_Pull_Shirt.layers:
    layer.trainable = False
for layer in C_Ank_Sneaker.layers:
    layer.trainable = False


########################################################################################
################################ NDC CLASSIFICATION FUCNTION MNIST FASHION - FIXED ND:
########################################################################################
def NDC_prediction(image):
    # input  : image
    # output : 1- list of string labels reflecting the route the image took
    #          2- final label to compare against the true one
    # MAKE THE FUNCTION RETURNS A PROB VECTOR BY TRYING EVERY ROUTE AND BASE THE PREDICTION ON THE ARGMAX OF PROBs Q


    # root prediction:
    root_pred = np.argmax(C_Other_Not(image.reshape(1,28,28,1)).numpy()[0])
    final_prediction = []
    lbls_route       = []
    #print(root_pred)
    if root_pred == 0:
        lbls_route.append('other')
        # use C_Bag_Not
        C_Bag_Not_pred = np.argmax(C_Bag_Not(image.reshape(1,28,28,1)).numpy()[0])
        if C_Bag_Not_pred == 0:
            lbls_route.append('Bag')
            # leaf node is reached
            final_prediction.append([8])
        else:
            lbls_route.append('Not Bag')
            # use C_dress_coat
            C_dress_coat_pred = np.argmax(C_dress_coat(image.reshape(1, 28, 28, 1)).numpy()[0])
            if C_dress_coat_pred == 0:
                lbls_route.append('Dress')
                # leaf node is reached
                final_prediction.append([3])
            else:
                lbls_route.append('Coat')
                # leaf node is reached
                final_prediction.append([4])
    else:
        lbls_route.append('Not other')
        # use C_top_bottom
        C_top_bottom_pred = np.argmax(C_top_bottom(image.reshape(1, 28, 28, 1)).numpy()[0])
        if C_top_bottom_pred == 0:
            lbls_route.append('Top')
            # use C_Tshirt_Not
            C_Tshirt_Not_pred = np.argmax(C_Tshirt_Not(image.reshape(1, 28, 28, 1)).numpy()[0])
            if C_Tshirt_Not_pred == 0:
                lbls_route.append('T-shirt')
                # leaf node is reached
                final_prediction.append([0])
            else:
                lbls_route.append('Not T-shirt')
                # use C_Pull_Shirt
                C_Pull_Shirt_pred = np.argmax(C_Pull_Shirt(image.reshape(1, 28, 28, 1)).numpy()[0])
                if C_Pull_Shirt_pred == 0:
                    lbls_route.append('Shirt')
                    # leaf node is reached
                    final_prediction.append([6])
                else:
                    lbls_route.append('Pullover')
                    # leaf node is reached
                    final_prediction.append([2])
        else:
            lbls_route.append('Bottom')
            # use C_Touser_Not
            C_Touser_Not_pred = np.argmax(C_Touser_Not(image.reshape(1, 28, 28, 1)).numpy()[0])
            if C_Touser_Not_pred == 0:
                lbls_route.append('Trouser')
                # leaf node is reached
                final_prediction.append([1])
            else:
                lbls_route.append('Not Trouser')
                # use C_Sandal_Not
                C_Sandal_Not_pred = np.argmax(C_Sandal_Not(image.reshape(1, 28, 28, 1)).numpy()[0])
                if C_Sandal_Not_pred == 0:
                    lbls_route.append('Sandal')
                    # leaf node is reached
                    final_prediction.append([5])
                else:
                    lbls_route.append('Not Sandal')
                    # use C_Ank_Sneaker
                    C_Ank_Sneaker_pred = np.argmax(C_Ank_Sneaker(image.reshape(1, 28, 28, 1)).numpy()[0])
                    if C_Ank_Sneaker_pred == 0:
                        lbls_route.append('Ankle Boot')
                        # leaf node is reached
                        final_prediction.append([7])
                    else:
                        lbls_route.append('Sneaker')
                        # leaf node is reached
                        final_prediction.append([9])
    return final_prediction, lbls_route


#test fucntion:
pred = []

succ = 0
for i in range(10000):
    pred_temp = NDC_prediction(X_test[i])[0]
    if pred_temp[0] == test_labels[i]:
        succ = succ + 1
    pred.append(pred_temp)




print('break')















# number_of_observations = 10
# succ=0
# D_ssim_images_save = []
# D_JS_save          = []
# run_time_save      = []
#
# for idx in range(number_of_observations):
#     start = timeit.default_timer()
#     ########################################################################################
#     ###########################################################################
#     #################################### BUILDING THE gen model g(z,\phi)
#     ###########################################################################
#
#     gen_NN = tf.keras.Sequential()
#     initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=102)
#     ## ADDING THE GEN MODEL layers that will be trained
#
#     layer = layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,), name='dense_gen', kernel_initializer=initializer)
#     layer.trainable=True
#     gen_NN.add(layer)
#
#     layer = layers.BatchNormalization()
#     layer.trainable=True
#     gen_NN.add(layer)
#
#     layer = layers.LeakyReLU()
#     layer.trainable=True
#     gen_NN.add(layer)
#
#     layer = layers.Reshape((7, 7, 256),name='reshape_gen')
#     layer.trainable=True
#     gen_NN.add(layer)
#     #assert combined_NN.output_shape == (None, 7, 7, 256)
#
#     layer = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=initializer)
#     layer.trainable=True
#     gen_NN.add(layer)
#     #assert gen_NN.output_shape == (None, 14, 14, 64)
#
#     layer = layers.BatchNormalization()
#     layer.trainable=True
#     gen_NN.add(layer)
#
#     layer = layers.LeakyReLU()
#     layer.trainable=True
#     gen_NN.add(layer)
#
#     layer = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=initializer)
#     layer.trainable=True
#     gen_NN.add(layer)
#     #assert gen_NN.output_shape == (None, 14, 14, 64)
#
#     layer = layers.BatchNormalization()
#     layer.trainable=True
#     gen_NN.add(layer)
#
#     layer = layers.LeakyReLU()
#     layer.trainable=True
#     gen_NN.add(layer)
#
#     layer = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid', kernel_initializer=initializer)
#     layer.trainable=True
#     gen_NN.add(layer)
#
#
#     # below is added for the 1D modification
#     layer = layers.Reshape((784, 1, 1),name='reshape_gen_final')
#     layer.trainable=True
#     gen_NN.add(layer)
#
#
#
#
#     #########################################################################################################################################
#     ############################### this is NOT sequentiail traning (traning two loss fucntions from different heads of the NN) ######
#     #########################################################################################################################################
#
#     ### define X_d as the desired
#     X_desired = X_test[idx]
#
#     # these two need to be have the same values as of now since X_train is the same for both
#     batch_size_gen = 80
#     batch_size_2 = 80
#
#
#     #################### training steps and stopping thresholds
#     delta_s  = 0.25
#
#     delta_js = 0.20
#     delta_ssim = 0.80
#
#     delta_c = 0.25
#
#     traning_steps = 10
#
#     ############################################################
#     ###  automated desired for confidence reduction y_d (desired PMF)
#     ################################################################
#     # number_of_classes_corser  = 3
#     # # take the predicted class instead of the true label
#     # #target_class       =  test_labels[idx]
#     # target_class = np.argmax(trained_model_coarser(X_desired.reshape(1, 784, 1)).numpy()[0])
#     #
#     # target_super_label = targeted_super_label_confer[idx]
#     #
#     # desired_PMF = np.zeros(shape=(1, 3))
#     # desired_PMF[0][target_super_label] = 1
#
#
#     nnumber_of_classes_corser  = 3
#     # take the predicted class instead of the true label
#     #target_class       =  test_labels[idx]
#     target_class = np.argmax(trained_model_coarser(X_desired.reshape(1, 784, 1)).numpy()[0])
#
#
#     desired_confidence =  0.6
#
#     #1 code the confidence only
#     desired_PMF_confidence = np.zeros(shape=(1,nnumber_of_classes_corser))
#     for i in range(nnumber_of_classes_corser):
#         if i == target_class:
#             desired_PMF_confidence[:,i] = desired_confidence
#         else:
#             desired_PMF_confidence[:,i] = (1-desired_confidence) / (nnumber_of_classes_corser-1)
#
#
#
#     ################################################################
#     ##### X_train is the same for both gen and combined models #####
#     ################################################################
#     # build x_train as some random input and y_train to be the desired image
#     # X_train is the same as z in the paper
#
#     # create one vector and repeat
#     X_train_one = tf.random.uniform(shape=[1,100], minval=0., maxval=1., seed=101)
#     X_train_one_np = X_train_one.numpy()
#     X_train_np = np.zeros(shape=(batch_size_gen,100))
#     for i in range(batch_size_gen):
#         X_train_np[i,:] = X_train_one_np
#     X_train = tf.convert_to_tensor(X_train_np, dtype=tf.float32)
#     X_val_np = X_train_one_np
#     X_val = tf.convert_to_tensor(X_val_np, dtype=tf.float32)
#
#
#     ############################################################
#     ### Y_train_gen for the gen model (whcih is the image)
#     ################################################################
#
#     # below is for the 1D image
#     Y_train_np_gen = np.zeros(shape=(batch_size_gen,784,1,1))
#     Y_val_gen = X_desired.reshape(1,784,1,1)
#     for i in range(batch_size_gen):
#         Y_train_np_gen[i,:,:,:] = Y_val_gen.reshape(784,1,1)
#     # convert Y_train to tf eager tensor
#     Y_train_gen = tf.convert_to_tensor(Y_train_np_gen, dtype=tf.float32)
#
#
#     ############################################################
#     ### Y_train_combined is the y_d (desired PMF)
#     ################################################################
#
#     Y_train_combined = np.zeros(shape=(batch_size_2,3))
#
#     Y_val_combined = desired_PMF_confidence
#
#     for i in range(batch_size_2):
#         Y_train_combined[i,:] = Y_val_combined
#
#     Y_desired = Y_val_combined[0]
#
#     print('break')
#
#
#     ####################################################################################################################
#     ### defining the combined model such that its the concatenation of g, then f ==> this is defing model h in the paper
#     #############################################################################################################################
#
#     input = Input(shape=100)
#
#     x = gen_NN.layers[0](input)
#     for lay in range(len(gen_NN.layers) - 1):
#         layer = gen_NN.layers[lay+1]
#         layer.trainable = True
#         x = layer(x)
#     out_1 = x
#
#
#     x_2 = trained_model_coarser.layers[0](x)
#     for lay in range(len(trained_model_coarser.layers) - 1):
#         layer = trained_model_coarser.layers[lay + 1]
#         layer.trainable = False
#         x_2 = layer(x_2)
#     out_2 = x_2
#
#
#     ### defining the model: this is h(z,\psi)
#     combined_NN = Model(input, [out_1, out_2])
#
#     ### defning the optimizer
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
#
#     loss_1 = tf.keras.losses.MeanSquaredError(name='LOSS_1')
#
#     loss_2 = tf.keras.losses.CategoricalCrossentropy(name='LOSS_2', from_logits=False,label_smoothing=0)
#
#     dynamic_weights_selection = True
#     # initial losses functions weights
#     lambda_gen = 1
#     lambda_pmf = 0.01
#
#     ############# trainING LOOP
#     for i in range(traning_steps):
#
#
#         combined_NN.compile(optimizer=optimizer, loss=[loss_1, loss_2], loss_weights=[lambda_gen, lambda_pmf])
#         # for lay in range(18):
#         #     if lay >= 12:
#         #         layer = combined_NN.layers[lay]
#         #         layer.trainable = False
#
#         # traning
#         combined_NN.fit(X_train,[Y_train_gen, Y_train_combined], epochs=1, batch_size=1, validation_data=(X_val, [Y_val_gen,Y_val_combined]), verbose=0 )
#         #combined_NN.train_on_batch(X_train, [Y_train_gen, Y_train_combined])
#         # fake image at step i ==> this is X in the paper and X_val is z in the paper
#         fake_image = combined_NN(X_val)[0].numpy().reshape(28, 28)
#
#
#         #trained_model = load_model("MNIST_digits_trained_model_3.h5")
#         trained_model_coarser = tf.keras.models.load_model('coarser_model_1d')
#         output_vector_probabilities = trained_model_coarser(fake_image.reshape(1, 28, 28, 1)).numpy()[0]
#         #output_vector_probabilities = combined_NN(X_val)[1].numpy().reshape(10,)
#
#         # D_2 distance between real image and fake image at step i==> this is equation (9)
#         D_2_s = LA.norm(X_desired.reshape(784,) - fake_image.reshape(784,),          2)
#         # SSIM distance between real image and fake image at step i ==>
#         D_ssim_images = SSIM_index(X_desired, fake_image)
#         # D_2 distance between desired PMF and the PMF returned by the fake image ==> this is equation (3)
#         D_2 = LA.norm(output_vector_probabilities-Y_desired,                  2       )
#         # D_JS: JS divergence distance between desired and actual PMFs (it uses KL divergence)
#         D_JS = D_JS_PMFs(output_vector_probabilities, Y_desired)
#
#
#         ### THE STOPPING EXIT CRITERIA
#         if D_ssim_images >= delta_ssim  and D_JS <= delta_js:
#             #print('BREAKING FOR IS USED with Distance SSIM = ', D_ssim_images, ' and D_JS = ', D_JS)
#             break
#
#         ### logger:
#         #print('training step = ', i, '; image SSIM = ', D_ssim_images, ' ; PMF_JS_Distance = ', D_JS, ' ; current loss weights = ', lambda_gen,' , ', lambda_pmf )
#
#         ##### dynamic weight selection option in training
#         if dynamic_weights_selection is True:
#             lambda_gen = relu_scaler_Ismail(lambda_gen       -   0.01 * 1    * ((D_ssim_images/delta_ssim)) * np.sign((D_ssim_images/delta_ssim)-1))
#             lambda_pmf = relu_scaler_Ismail(lambda_pmf       -   0.05 * 0.01 * ((delta_js/D_JS))            * np.sign((delta_js/D_JS           )-1))
#         else:
#             lambda_gen = 1
#             lambda_pmf = 0.01
#
#
#         ### SAVE THE DISTNCE AND PERTURBED IMAGE SO AS TO TAKE THE MINIMUM AT THE END OF THE TRAINING STEP (THIS IS TO OVER COME OVERFITTING DURING TRAINING)
#
#
#
#     fake_image = combined_NN(X_val)[0].numpy().reshape(28,28)
#
#     ### below is the same thing (just for sanity check)
#     #trained_model = load_model("MNIST_digits_trained_range_1to1_1d_input.h5")
#     trained_model_coarser = tf.keras.models.load_model('coarser_model_1d')
#     output_vector_probabilities   = trained_model_coarser(fake_image.reshape(1,28,28,1)).numpy()[0]
#
#
#
#     real_image = X_desired.reshape(28,28)
#
#     confidence_score      = np.max(trained_model_coarser(X_desired.reshape(1,28,28,1)).numpy()[0])
#
#     confidence_score_BOSS = np.max(trained_model_coarser(fake_image.reshape(1,28,28,1)).numpy()[0])
#
#     #### #### passing criteria: prediction(X) = prediction(X_BOSS)
#     if np.argmax(trained_model_coarser(fake_image.reshape(1, 784, 1)).numpy()[0]) == np.argmax(trained_model_coarser(X_desired.reshape(1, 784, 1)).numpy()[0]):
#         succ = succ + 1
#
#     #### outer loop logger:
#     print('[index,succ] = ', [idx, succ], ' - [confidence] = ', [confidence_score, confidence_score_BOSS],
#           ' - [SSIM,JS] = ', [D_ssim_images, D_JS])
#
#
#     D_ssim_images_save.append(D_ssim_images)
#     D_JS_save.append(D_JS)
#
#     stop = timeit.default_timer()
#     run_time_save.append(stop - start)
#
#     #print('Time: ', stop - start)
#
#
# print('OVERALL PERFORMANCE: ', 'succ = ' , 100*(succ/number_of_observations) , 'AVG_SSIM_JS = ',[np.mean(D_ssim_images_save),np.mean(D_JS_save)], ' AVG runtime = ',np.mean(run_time_save))
#
#
#
# # plt.figure()
# # plt.subplot(2,2,1)
# # plt.title('Desired example')
# # plt.imshow(real_image,cmap='gray',vmin=0, vmax=1)
# # plt.colorbar()
# # plt.axis('off')
# # plt.subplot(2,2,2)
# # plt.title('Generated example')
# # plt.imshow(fake_image,cmap='gray',vmin=0, vmax=1)
# # plt.colorbar()
# # plt.axis('off')
# # plt.subplot(2,2,4)
# # plt.title('Generated example PMF')
# # plt.stem(output_vector_probabilities)
# # plt.ylim(top=1.2)
# # plt.subplot(2,2,3)
# # plt.title('Desired PMF')
# # plt.stem(Y_val_combined[0])
# # plt.ylim(top=1.2)
#
#

print('break')



