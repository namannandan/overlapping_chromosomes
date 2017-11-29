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
from nn_model import segnet_model1

#seed
t.manual_seed(1)
#flag to indicate debug mode
debug = False
#flag to resume training
resume_training = False
resume_checkpoint_file = 'checkpoint.pth.tar'
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
net = segnet_model1()
if(resume_training):
    net.load_state_dict(checkpoint['model'])

#hyper parameters
#percentage of data to be used as training data
percent_training_data = 70
#batch size
batch_size = 64
#learning rate
#(0.5 works) (0.002 for adam)
eta = 0.3
#number of epochs
num_epochs = 100
#TODO:choose correct optimizer
#optimizer
#working !
#optimizer = optim.Adadelta(net.parameters(), lr=eta, rho=0.9, eps=1e-6, weight_decay=0.00001)
#experiments !
optimizer = optim.Adadelta(net.parameters(), lr=eta)
#optimizer = optim.Adam(net.parameters(), lr=eta, weight_decay=0.00005)

#optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
#optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.3, weight_decay=0.1)
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
#TODO:fix normalization
#transform to Normalize the input data (subtract mean from each pixel)
normalize = tv.transforms.Compose([tv.transforms.Normalize((5.9,),(1.0,))])

#method to save checkpoints of the trained model and the optimizer state
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    t.save(state, filename)
    #update the best model obtained so far
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

for epoch in range (num_epochs):
    #set data container mode to train
    data.set_mode('train')
    train_error = 0
    train_count = 0
    #iterate over the training samples
    for batch_id, (input_image_tensor, target_image_tensor) in enumerate(data):
        #TODO:enable normalization
        #normalize the input images
        # for i in range(input_image_tensor.size()[0]):
        #     #normalize each of the input images in the batch
        #     input_image_tensor[i] = normalize(input_image_tensor[i])
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
        #weight=t.Tensor([0.5,1.5,1.5,2.0,2.0])
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
                     'best_val_error': best_val_error}, is_best)

    #print the training and test error every 10 epochs
    if (epoch % 10 == 0):
        print ('Epoch: ', epoch)
        print ('Training error: ', error_dict['train'][-1])
        print ('Validation error: ', error_dict['val'][-1], '\n')


    #TODO: check updation of learning rate
    #update learning rate if required
    # if (epoch > 0 and epoch%10==0):
    #     avg_val_error = sum(error_dict['val'][-10:len(error_dict['val'])])/10
    #     diff = abs(avg_val_error - error_dict['val'][-1])
    #     if (diff < 0.01 and eta > 0.0001):
    #         eta = eta*0.1
    #         optimizer = optim.SGD(net.parameters(), lr=eta)

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
