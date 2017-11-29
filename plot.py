import torch as t
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

y_val = []
#class name of the model being used
model_name = 'segnet_model1'
checkpoint_file = model_name+'_model_best.pth.tar'
checkpoint = t.load(checkpoint_file)
errors = checkpoint['errors']

y_val = errors['train']

x_val = []
for x in range(0, len(y_val)):
	x_val.append(x)

training_error, = plt.plot(x_val, y_val, 'xb-')
plt.title('Training and Validation error plots')
plt.xlabel('Number of Epochs')
plt.ylabel('Error')

y_val = errors['val']

val_error, = plt.plot(x_val, y_val, '.r-')

plt.legend([training_error, val_error], ["Training Error", "validation Error"])

plt.savefig(model_name+'_train_val_error.jpg')
