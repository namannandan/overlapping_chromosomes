import torch as t
import torchvision as tv
from data_loader import data_container
from matplotlib import pyplot as plt

#debug flag
debug = True
#create a data container, with 70% (same as that used while training) as training data
#batch size doesn't matter here, since we will only look at test data
data = data_container(70, 1, debug)
#set the mode of the data container to 'test'
data.set_mode('val')
#load the saved network
net = t.load(open('saved_model.pt', 'rb'))

#padding required for the input images given to the network
image_padding = t.nn.ZeroPad2d((1,2,1,1))
#transform to Normalize the input image
normalize = tv.transforms.Compose([tv.transforms.Normalize((5.9,),(21.54,))])

#configure matplotlib
#enable interactive mode
plt.ion()
plt.show()

#function to compute accuracy of the segmentation output
def get_accuracy(tensor1, tensor2):
    eq_tensor = t.eq(tensor1.type(t.FloatTensor), tensor2.type(t.FloatTensor))
    num_pixels = tensor1.size()[2] * tensor1.size()[3]
    return ((sum(sum(eq_tensor[0, 0, :, :])) * 100) / num_pixels)

#variables to track the summation of the accuracy for each image and count
sum_accuracy = 0
count = 0
#iterate over the test samples
for batch_id, (input_image, target_image) in enumerate(data):
    #TODO:fix normalization
    #normalize the input image
    #input_image[0] = normalize(input_image[0])
    #add padding to the input image
    input_image = image_padding(input_image)
    #get the output of the network
    output = net(input_image)
    #extract the data contained in the Variable
    output = output.data
    #get the index of the feature map that has the max value at each pixel location
    ret, output = t.max(output, 1, keepdim=True)
    #add padding to the target image
    target_image = image_padding(target_image)
    #extract the data contained in the Variable
    target_image = target_image.data
    #determine accuracy
    accuracy = get_accuracy(output, target_image)
    #accumulate the computed accuracy and count
    sum_accuracy += accuracy
    count += 1
    #if in debug mode, display the plots
    if (debug):
        print('Accuracy : ' + str(accuracy))
        #plot input image
        input_image = (input_image.data).numpy()
        plt.subplot(131)
        plt.title('Input')
        plt.imshow(input_image[0,0,:,:], cmap='gray')
        #plot the target image
        target_image = target_image.numpy()
        plt.subplot(132)
        plt.title('Expected')
        plt.imshow(target_image[0,0,:,:])
        #plot the output image
        output = output.numpy()
        plt.subplot(133)
        plt.title('Output')
        plt.imshow(output[0, 0, :, :])
        plt.show()
        plt.pause(0.001)
        #wait for user input to continue
        input_str = input("Press [enter] to continue or 'e' to exit: ")
        if (len(input_str)>0 and input_str[0]=='e'):
            break

if (not debug):
    print ('Average accuracy is : ' + str(sum_accuracy/count))
