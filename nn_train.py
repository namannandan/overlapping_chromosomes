import torch as t
import torchvision as tv
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import math
import pickle
import shutil

from data_loader import data_container
from nn_model import *

#seed
t.manual_seed(1)
#flag to indicate debug mode
debug = False
#class name of the model being used
model_name = 'nn_model'
#flag to resume training
resume_training = True
resume_checkpoint_file = model_name+'_checkpoint.pth.tar'
#number of training samples to use in debug mode
#the same set of samples will be used repeatedly in each epoch
debug_train_count = 100
#number of validation samples to use in debug mode
debug_val_count = 50

#setup matplotlib
#set interactive mode on
plt.ion()
plt.show()

#load the saved state if resuming training
if(resume_training):
    checkpoint = t.load(resume_checkpoint_file)

#neural network model
net = nn_model()
if(resume_training):
    net.load_state_dict(checkpoint['model'])
#verify that we are using the correct model
if (type(net).__name__ != model_name):
    print("The intended neural net model is not being used")
    exit()
#set mode of the model to training
net.train()

#hyper parameters
#percentage of data to be used as training data
percent_training_data = 70
#batch size
batch_size = 128
#learning rate
#(0.5 works) (0.002 for adam)
eta = 10.0
#number of epochs
num_epochs = 100
#optimizer
optimizer = optim.Adadelta(net.parameters(), lr=eta, weight_decay=0.0001)
#weight assigned to the errors
error_weights = t.Tensor([1.0,1.5,1.5,1.5,1.0])

if (resume_training):
    optimizer.load_state_dict(checkpoint['optimizer'])

#dataset
#create a data container, with 70% as training data and batch_size=16
data = data_container(percent_training_data, batch_size, debug)

#dictionary to store the error values
if (resume_training):
    error_dict = checkpoint['errors']
    #variable to track the best validation error
    best_val_error = checkpoint['best_val_error']
else:
    error_dict = {'train':[], 'val':[]}
    best_val_error = math.inf


#padding required for the input and target images
image_padding = t.nn.ZeroPad2d((1,2,1,1))
#transform to Normalize the input data (subtract mean from each pixel)
normalize = tv.transforms.Compose([tv.transforms.Normalize((5.9,),(1.0,))])

#method to save checkpoints of the trained model and the optimizer state
def save_checkpoint(state, is_best, model):
    t.save(state, model+'_checkpoint.pth.tar')
    #update the best model obtained so far
    if is_best:
        shutil.copyfile(model+'_checkpoint.pth.tar', model+'_best.pth.tar')

for epoch in range (num_epochs):
    #set data container mode to train
    data.set_mode('train')
    train_error = 0
    train_count = 0
    #iterate over the training samples
    for batch_id, (input_image_tensor, target_image_tensor) in enumerate(data):
        #normalize the input images
        for i in range(input_image_tensor.size()[0]):
            #normalize each of the input images in the batch
            input_image_tensor[i] = normalize(input_image_tensor[i])
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
        err = F.nll_loss(output, target_image_tensor.type(t.LongTensor)[:,0,:,:], weight=error_weights)
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
    #reset the values of count and error
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
        err = F.nll_loss(output, target_image_tensor.type(t.LongTensor)[:,0,:,:], weight=error_weights)
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

    #update best_val_error if required and save checkpoint
    if(error_dict['val'][-1] < best_val_error):
        is_best = True
        best_val_error = error_dict['val'][-1]
    else:
        is_best = False
    save_checkpoint({'epoch': epoch,
                     'model': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'errors': error_dict,
                     'best_val_error': best_val_error}, is_best, model_name)

    #print the training and test error every 10 epochs
    if (epoch % 10 == 0):
        print ('Epoch: ', epoch)
        print ('Training error: ', error_dict['train'][-1])
        print ('Validation error: ', error_dict['val'][-1], '\n')

    #update the plot
    if (debug):
        #plot the error values
        #Title of the plot
        plt.title('Training and Validation error plots')
        #axis labels
        plt.xlabel('Number of Epochs')
        plt.ylabel('Error')
        #plot the training and validation errors
        train_error_plot, = plt.plot(error_dict['train'], 'xb-')
        val_error_plot, = plt.plot(error_dict['val'], 'r-')
        #add legends for the plots
        plt.legend([train_error_plot, val_error_plot], ["Training Error", "validation Error"])
        plt.pause(0.001)

#if in debug mode, hold the plot open
if (debug):
    plt.ioff()
    plt.show()
