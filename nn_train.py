import torch as t
import torchvision as tv
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import math
import pickle

from data_loader import data_container
from nn_model import segnet_model1

#flag to indicate debug mode
debug = True
#number of training samples to use in debug mode
#the same set of samples will be used repeatedly in each epoch
debug_train_count = 50
#number of validation samples to use in debug mode
debug_val_count = 5

#setup matplotlib
#set interactive mode on
plt.ion()
plt.show()

#hyper parameters
#percentage of training data
percent_training_data = 70
#batch size
batch_size = 16
#learning rate
eta = 0.5
#number of epochs
num_epochs = 100
#create a data container, with 70% as training data and batch_size=16
data = data_container(percent_training_data, batch_size, debug)
#neural network model
net = segnet_model1()
#optimizer
optimizer = optim.Adadelta(net.parameters(), lr=eta, rho=0.9, eps=1e-6, weight_decay=0.00001)

#dictionary to store the error values
error_dict = {'train':[], 'val':[]}
#padding required for the input and target images
image_padding = t.nn.ZeroPad2d((1,2,1,1))
#transform to Normalize the input data
normalize = tv.transforms.Compose([tv.transforms.Normalize(5.9,21.54)])

for epoch in range (num_epochs):
    #set data container mode to train
    data.set_mode('train')
    train_error = 0
    train_count = 0
    #iterate over the training samples
    for batch_id, (input_image_tensor, target_image_tensor) in enumerate(data):
        #TODO : implement normalization of input images
        #normalize the input images
        #input_image_tensor = normalize(input_image_tensor)
        #pad the input images and convert to Variable
        #spatial dimensions of the images before padding is (94x93)
        #spatial dimensions of the images after padding is (96x96)
        input_image_tensor = image_padding(input_image_tensor)
        #zero out the gradients of the parameters
        optimizer.zero_grad()
        #obtaint the output of the network
        output = net(input_image_tensor)
        #add the required padding to the labels and convert to Variable
        target_image_tensor = image_padding(target_image_tensor)
        #compute the error
        #convert type of "target_image_tensor" from FloatTensor to LongTensor,
        #since nll_loss expects the exptected target to be of type LongTensor having 3 dimensions
        err = F.nll_loss(output, target_image_tensor.type(t.LongTensor)[:,0,:,:])
        #backpropagate the error
        err.backward()
        #compute the summation of the error for the batch (average error for batch * batch size)
        train_error += (err.data[0] * input_image_tensor.size()[0])
        train_count += input_image_tensor.size()[0]
        #if in debug mode, stop training once we have looked at the required number of samples
        if (debug):
            if (train_count >= debug_train_count):
                break
    #save the training error
    error_dict['train'].append(train_error/train_count)

    #set mode of data container to 'val'
    data.set_mode('val')
    #reset the vlues of count and error
    val_error = 0
    val_count = 0
    #iterate over the validation samples
    for index, (input_image_tensor, target_image_tensor) in enumerate(data):
        #pad the input images and convert to Variable
        input_image_tensor = image_padding(input_image_tensor)
        #obtaint the output of the network
        output = net(input_image_tensor)
        #add the required padding to the labels and convert to Variable
        target_image_tensor = image_padding(target_image_tensor)
        #compute the error
        err = F.nll_loss(output, target_image_tensor.type(t.LongTensor)[:,0,:,:])
        #compute the summation of the error for each sample
        val_error += (err.data[0])
        val_count += 1
        #if in debug mode, stop validating once we have looked at the required number of samples
        if (debug):
            if (val_count >= debug_val_count):
                break
    #save the validation error
    error_dict['val'].append(val_error/val_count)

    #update the weights
    optimizer.step()

    #update the plot
    if (debug):
        #plot the error values
        #TODO: add a title to the graph and also a legend for the two plots
        plt.plot(error_dict['train'], 'xb-')
        plt.plot(error_dict['val'], 'r-')
        plt.pause(0.001)

#save state only when not in debug mode
if (not(debug)):
    #save the error values
    pickle.dump(error_dict, open('saved_errors.p', 'wb'))
    #save the trained model
    t.save(net, 'saved_model.pt')

#if in debug mode, hold the plot open
if (debug):
    plt.ioff()
    plt.show()
