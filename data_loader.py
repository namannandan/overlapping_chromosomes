import h5py
import numpy as np
import torch as t

class data_container():
    '''a class that handles loading data, shuffling it and providing training, validation and test data'''
    def __init__(self, train_data_percentage=70, batch_size=1, debug=False):
        '''constructor'''
        #input data is a numpy array with 4 dimensions
        #(images(13,434), rows(94), columns(93), input image / segmentation label(2))
        #Mean and Standard Deviation of pixel values of in put images
        #Mean = 5.8991370727534225
        #Standard Deviation = 21.539881799673775
        h5f = h5py.File('overlapping-chromosomes/LowRes_13434_overlapping_pairs.h5','r')
        self.data_ndarray = h5f['dataset_1'][:]
        h5f.close()
        #get a handle for tensor representation of data
        #tensor is of type "Long Tensor" by default
        #data_tensor and data_ndarray will point to the same data in memory
        self.data_tensor = t.from_numpy(self.data_ndarray)
        #permute the dimensions of the tensor so that it uses the following format
        #original : [sample, row, column, channel(input/output)]
        #required : [sample, channel(input/output), row, column]
        self.data_tensor = self.data_tensor.permute(0, 3, 1, 2)
        #number of data samples
        self.num_data_samples = self.data_tensor.size()[0]
        #compute the indices for the training, validation and test data
        #these indices are used when specifying the range of values (the value at the last index is not included in a given set)
        #check if train_data_percentage is valid
        if not((train_data_percentage > 0) and (train_data_percentage <= 100)):
            print ('Invalid value received for train_data_percentage : ' + str(train_data_percentage))
            print ('exiting')
            exit()
        self.training_data_indices = [0, int((train_data_percentage/100)*self.num_data_samples)]
        self.validation_data_indices = [self.training_data_indices[1], int((self.training_data_indices[1]+self.num_data_samples)/2)]
        self.test_data_indices = [self.validation_data_indices[1], self.num_data_samples]
        #by default, set the mode to train
        self.mode = 'train'
        #record the batch size
        self.batch_size = batch_size
        #create iterator variables to track the current state of the iterator
        self.iterator_index = 0
        self.iterator_max = 0
        #store the image width and height
        self.image_width = self.data_tensor.size()[3]
        self.image_height = self.data_tensor.size()[2]
        #store debug state
        self.debug = debug

    def set_mode(self, mode):
        '''function used to set the mode of the data container'''
        if (mode == 'train' or mode == 'val' or mode == 'test'):
            self.mode = mode
            #if the mode is changed, reset the iterator variables
            self.iterator_index = 0
            self.iterator_max = 0

        else:
            print ('set_mode() received invalid mode : ' + mode_str)
            print ('use one of : train, val or test')
            print ('exiting')
            exit()

    def shuffle_data(self):
        '''function to shuffle specific sections of the dataset'''
        #based on the current mode being used, shuffle only a specific portion of the dataset
        if (self.mode == 'train'):
            np.random.shuffle(self.data_ndarray[self.training_data_indices[0]:self.training_data_indices[1], :, :, :])
        elif (self.mode == 'val'):
            np.random.shuffle(self.data_ndarray[self.validation_data_indices[0]:self.validation_data_indices[1], :, :, :])
        elif (self.mode == 'test'):
            np.random.shuffle(self.data_ndarray[self.test_data_indices[0]:self.test_data_indices[1], :, :, :])

    def __iter__(self):
        '''function to return the Iterable object'''
        #shuffle the data that is going to be accessed, provided we are not in debug mode
        #Note: In debug mode, we would like to access a smaller subset of the data in the same order,
        #in order to compare performace of different networks / techniques
        if(not(self.debug)):
            self.shuffle_data()
        #setup the iterator variables
        #Note: value at the index self.iterator_max is not included for each class (train / val / test)
        if (self.mode == 'train'):
            self.iterator_index = self.training_data_indices[0]
            self.iterator_max = self.training_data_indices[1]
        elif (self.mode == 'val'):
            self.iterator_index = self.validation_data_indices[0]
            self.iterator_max = self.validation_data_indices[1]
        elif (self.mode == 'test'):
            self.iterator_index = self.test_data_indices[0]
            self.iterator_max = self.test_data_indices[1]
        else:
            self.iterator_index = 0
            self.iterator_max = 0

        return (self)

    def __next__(self):
        '''function to return the next item in the Iterable'''
        if (self.mode == 'train'):
            #check if we have enough data to return an entire batch
            if ((self.iterator_index + self.batch_size) < self.training_data_indices[1]):
                batch_data_tensor = self.data_tensor[self.iterator_index:(self.iterator_index+self.batch_size), :, :, :]
                #obtain the input and expected output data and conver to FloatTensor
                batch_data_tensor_input = (batch_data_tensor[:, 0:1, :, :]).type(t.FloatTensor)
                batch_data_tensor_output = (batch_data_tensor[:, 1:2, :, :]).type(t.FloatTensor)
                #update iterator_index
                self.iterator_index += self.batch_size
                return ((batch_data_tensor_input, batch_data_tensor_output))
            elif (self.iterator_index < self.iterator_max):
                batch_data_tensor = self.data_tensor[self.iterator_index:self.iterator_max, :, :, :]
                batch_data_tensor_input = (batch_data_tensor[:, 0:1, :, :]).type(t.FloatTensor)
                batch_data_tensor_output = (batch_data_tensor[:, 1:2, :, :]).type(t.FloatTensor)
                self.iterator_index = self.iterator_max
                return ((batch_data_tensor_input, batch_data_tensor_output))
            else:
                raise StopIteration
        else:
            #each batch for val or test data has only one sample
            if (self.iterator_index < self.iterator_max):
                batch_data_tensor = self.data_tensor[self.iterator_index:(self.iterator_index+1), :, :, :]
                batch_data_tensor_input = (batch_data_tensor[:, 0:1, :, :]).type(t.FloatTensor)
                batch_data_tensor_output = (batch_data_tensor[:, 1:2, :, :]).type(t.FloatTensor)
                self.iterator_index += 1
                return ((batch_data_tensor_input, batch_data_tensor_output))
            else:
                raise StopIteration
