import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import scipy
import scipy.misc
import imageio
from PIL import Image
from scipy import ndimage
import PySimpleGUI as sg
import json,pickle

# ------------------------------------------------------------------------

def load_dataset(choice):
        numbers = [ # Used to insert to get file directory
            "one","two","three","four","five","six","seven","eight","nine","ten"
        ]
        #datanum = numbers[choice-1] # 0 based, back by one
        train_dataset = h5py.File(
            r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5', "r")
        train_set_x_orig = np.array(train_dataset["kanji_" + choice + "_x"][:]) # Original dataset
        #  train set features , the file of _train_set_x , the feature of RGB channel values from 0->64 since it is 64x64
        train_set_y_orig = np.array(train_dataset["kanji_" + choice  +"_y"][:])  # train set labels, whether it is 1(true) or 0(not)
        test_dataset = h5py.File(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5', "r")
        test_set_x_orig = np.array(test_dataset["kanji_" + "one" + "_x"][:])  #  test set features
        test_set_y_orig = np.array(test_dataset["kanji_" +"one"+ "_y"][:])  #  test set labels
        # The test data, to compare how accurate the algorithm is, random kanji featuring all
        classes = np.array(
            test_dataset["kanji_" + "one" +"_class"][:])  # the list of classes, which is just a string of 'kanji' and 'non-kanji'
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



# Loading the data (kanji/non-kanji)

train_set_x_orig_one, train_set_y_one, test_set_x_orig_one, test_set_y_one, classes_one = load_dataset("one")
train_set_x_orig_two, train_set_y_two, test_set_x_orig_two, test_set_y_two, classes_two = load_dataset("two")


# Reshape the training and test examples
train_set_x_flatten_one = train_set_x_orig_one.reshape(train_set_x_orig_one.shape[0], -1).T
test_set_x_flatten_one = test_set_x_orig_one.reshape(test_set_x_orig_one.shape[0], -1).T
train_set_x_flatten_two = train_set_x_orig_two.reshape(train_set_x_orig_two.shape[0], -1).T
test_set_x_flatten_two = test_set_x_orig_two.reshape(test_set_x_orig_two.shape[0], -1).T

#  Standardize the arrays to give lower range during gradient descent
train_set_x_one = train_set_x_flatten_one / 255.
test_set_x_one = test_set_x_flatten_one / 255.
train_set_x_two = train_set_x_flatten_two / 255.
test_set_x_two = test_set_x_flatten_two / 255.
def sigmoid(z):
    """
    z - a scalar or numpy array of any size.
    """
    s = 1 / (1 + np.exp(-z))
    return s
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    Arguments
    dim - size of the w vector we want (or number of parameters in this case)
    Returns:
    w - initialized vector of shape (dim, 1)
    b - initialized scalar (corresponds to the bias)
    """
    w = np.zeros((dim, 1))
    b = 0
    #assert (w.shape == (dim, 1))
    #assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def propagate(w, b, X, Y):
    """
    Cost function and its gradient for the propagation
    Arguments:
    w - weights, a numpy array of size (num_px * num_px * 3, 1)
    b - bias, a scalar
    X - data of size (num_px * num_px * 3, number of examples)
    Y - true "label" vector (containing 0 if non-kanji, 1 if kanji) of size (1, number of examples)
    Return:
    cost - negative log-likelihood cost for logistic regression
    dw - gradient of the loss with respect to w, thus same shape as w
    db - gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1] # num examples

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation

    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=1)  # compute cost
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1 / m) * np.dot(X, (A - Y).T) # Derivatives
    db = (1 / m) * np.sum(A - Y, axis=1)
    cost = np.squeeze(cost)
    print(cost)


    grads = {"dw": dw, # Dictionary to access derivatives
             "db": db}
    return cost

w,b = initialize_with_zeros(train_set_x_flatten_one.shape[0])
propagate(w,b,train_set_x_flatten_one,train_set_y_one)

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=True):
    """
    This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    w - weights, a numpy array of size (num_px * num_px * 3, 1)
    b - bias, a scalar
    X - data of shape (num_px * num_px * 3, number of examples)
    Y - true "label" vector (containing 0 if non-kanji, 1 if kanji), of shape (1, number of examples)
    num_iterations - number of iterations of the optimization loop
    learning_rate - learning rate of the gradient descent update rule
    print_cost - True to print the loss every 100 steps
    Returns:
    params - dictionary containing the weights w and bias b
    grads - dictionary containing the gradients of the weights and bias with respect to the cost function
    costs - list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    costs = []
    for i in range(num_iterations):
        m = X.shape[1] # Num examples
        A = sigmoid(np.dot(w.T, X) + b)
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y, axis=1)
        cost = (-1 / m) * np.sum((Y * np.log(A)) + (1 - Y) * np.log(1 - A), axis=1) # compute cost
        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs/error
        if i % 100 == 0:
            costs.append(cost)
        # Print loss every 100 training iterations
        if print_cost and i % 400 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs

# w,b = initialize_with_zeros(train_set_x_flatten_one.shape[0])
# learning_rates = [0.1]
# for i in learning_rates: # for each index
#     print ("learning rate is: " + str(i))
#     params,grads,costs = optimize(w, b, train_set_x_flatten_one, train_set_y_one, num_iterations=1500, learning_rate=i, print_cost=True)
#     print ("-------------------------------------------------------")
#     plt.plot(np.squeeze(costs), label= str(i)) # For each, plot the data
# plt.ylabel('cost') # Plot the axis on board to display
# plt.xlabel('iterations/hundreds)')
# legend = plt.legend(loc='upper center')
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()
def predict(w, b, X):  # prediction used to train model
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters w and b
    Arguments:
    w - weights: a numpy array of size (num_px * num_px * 3, 1)
    b - bias
    X - data of size : num_px * num_px * 3, number of examples
    Returns:
    Y_prediction - a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1] # 210
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1) # 210 by 1 vector
    # Compute vector "A" predicting the probabilities
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[0, i] <= 0.5: # If less than .5 then unlikely to be correct kanji image
            Y_prediction[0, i] = 0

        else:
            Y_prediction[0, i] = A[0, i] # If > .5 keep as likely


    return Y_prediction



def predict_single(w, b, X):  # prediction used to train model
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters w and b
    Arguments:
    w - weights: a numpy array of size (num_px * num_px * 3, 1)
    b - bias
    X - data of size : num_px * num_px * 3, number of examples
    Returns:
    Y_prediction - a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1] # 210
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1) # 210 by 1 vector
    # Compute vector "A" predicting the probabilities
    A = sigmoid(np.dot(w.T, X) + b)
    print(A)

    for i in range(A.shape[1]):
        # if A[0, i] <= 0.5: # If less than .5 then unlikely to be correct kanji image
        #     Y_prediction = 0
        Y_prediction= A[0, 0] # If > .5 keep, as likely


    return  Y_prediction



def model_kanji(X_train, Y_train, X_test, Y_test, num_iterations=5000, learning_rate=0.01, print_cost=False):
    """
    Builds the logistic regression model
    Arguments:
    X_train - training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train - training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test - test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test - test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations - hyperparameter representing the number of iterations to optimize the parameters
    learning_rate - hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    Returns:
    d - dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print(Y_prediction_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    # dictionary containg the values needed for the model

    model_data = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return model_data  # returns a dictionary which has data for model


# dict = {'P':123,'PP':12}
# # Save
# np.save(r"C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\model_data\data.npy", dict)
# # Load
# read_dictionary = np.load(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\model_data\data.npy',allow_pickle='TRUE').item()
# print(read_dictionary['P']) # displays "world"

# Train the models individually using the data provided
model_one = model_kanji(train_set_x_one, train_set_y_one, test_set_x_one, test_set_y_one, num_iterations=2500,
                        learning_rate=0.01, print_cost=True)
np.save(r"C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\model_data\k1.npy", model_one)
model_two = model_kanji(train_set_x_two, train_set_y_two, test_set_x_two, test_set_y_two, num_iterations=2500,
                        learning_rate=0.01, print_cost=True)
np.save(r"C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\model_data\k2.npy", model_two)

# # Convert the drawn image into an array of size 64 by 64, of 3 channels, which is 3D array
# fname = r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\image.jpeg'
# image = np.array(ndimage.imread(fname, flatten=False))
# my_image = scipy.misc.imresize(image, size=(64, 64)).reshape(
#     (1, 64 * 64 * 3)).T  # Change image to vector, THEN APPLY / 255.
# my_image = my_image / 255.


# # Note is that the images drawn should be mid-large rather than small as it struggles to predict
# # with smaller images
#
#
# """
# Mini algorithm: Each image has a prediction value, the actual images finds the largest probability
# the largest probability is most likely the image that has been drawn from the multiple prediction
# models, therefore find which this is and execute further instructions, original uses classes
# to classify which this is, however we only need to know which has been found, then allow for
# user to select if this is correct since this may not always be correct, then append to the corresponding table

# kanji_one = predict(model_one["w"], model_one["b"],my_image)  # CONTAINS THE VALUE OF WHETHER IT IS 0 OR 1, EVERYTHING ELSE IS STRINGS ADDED ON
# kanji_two = predict(model_two["w"], model_two["b"], my_image)
# actual_image = max(kanji_one, kanji_two)
# if actual_image == 0:
#     print('No kanji found')
# elif actual_image == kanji_one:
#     print("Predicted kanji is 'day-kanji' ")
#     # If message is found then display a message along with it's image and english meaning
#     # Ask if this is correct, if yes then append to corresponding table with y value as 1
#     # else append y value as 0 and append an image that is drawn,
#     # this is assuming that the user is honest
#     msg = " You have drawn a 'day-kanji', this is   "
# elif actual_image == kanji_two:
#     print("Predicted kanji is 'week-kanji' ")

# learning_rates = [0.1, 0.00001,0.01]
# models = {} # Dictionary to access each learning rate example
# for i in learning_rates: # for each index
#     print ("learning rate is: " + str(i))
#     models[str(i)] = model_kanji(train_set_x_one, train_set_y_one, test_set_x_one, test_set_y_one, num_iterations = 1500, learning_rate = i, print_cost = True)
#     print ('\n' + "-------------------------------------------------------" + '\n')
#     # store each one as dictionary element
# for i in learning_rates:
#     plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"])) # For each, plot the data
# plt.ylabel('cost')
# plt.xlabel('iterations (hundreds)')
# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()


#    -------------------------------------------------------------------------------------------------------------------------------CANVAS

from tkinter import *
import time
import pyautogui

canvas_width = 700
canvas_height = 450
from PIL import Image, ImageTk, ImageGrab, ImageDraw
import PIL

"""
PIL is a seperate image that is created, the tkinter drawing is on top of it to
simulate that it is being drawn on the PIL image, which is actually being drawn into PIL,
when drawing a black line, you give it info for the tkinter drawing (which does not save)
but the PIL drawing on top does, as proven by drawing in (paint_in) with a black line,
yet when erasing it (no white circle to draw PIL), the image is still the same.

"""


def paint_in(event):
    color = 'black'  # The colour to be drawn on
    size = 9  # Radius
    x1, y1 = (event.x - size), (event.y - size)  # Setting start ordinates
    x2, y2 = (event.x + size), (event.y + size)  # Setting end ordinates
    Canvas.create_oval(x1, y1, x2, y2, fill=color, outline=color)  # Create oval, when clicking
    # PIL
    draw.ellipse((x1, y1, x2, y2), fill='black', width=1)  # Draws same values onto the background


def clear(event):
    color = '#FFFFFF'  # Hex colour for white
    size = 1000  # Sets radius to 1000 to fully clear the screen
    x1, y1 = (event.x - size), (event.y - size)  # Set start ordinates
    x2, y2 = (event.x + size), (event.y + size)  # Set end ordinates
    Canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)  # Draws rectangle to fit screen to clear all
    # PIL
    draw.rectangle(
        (Canvas.winfo_rootx(), Canvas.winfo_y(),  # Sets starts to end vertices to fit onto screen, within background
         Canvas.winfo_rootx() + Canvas.winfo_width(),
         Canvas.winfo_rooty() + Canvas.winfo_height()), fill=color)


master = Tk()
master.title('Painting For Kanji')
Canvas = Canvas(master, width=canvas_width, height=canvas_height, bg='white')  # Create Canvas
Canvas.pack(expand=YES, fill=BOTH)  # Insert onto the screen

global counter
counter = 1
def save_data_true():
    # global counter
    fnamebackup = r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\image.jpeg'
    image1.save(fnamebackup)  # Save image as a screenshot where each time, to add new image classification
    fname = r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\image.jpeg'
    image = np.array(ndimage.imread(fname, flatten=False))
    my_image = scipy.misc.imresize(image, size=(64, 64)).reshape((1, 64 * 64 * 3)).T  # Change image to vector, THEN APPLY / 255.
    my_image = my_image / 255.
    kanji_one = predict_single(model_one["w"], model_one["b"],my_image)  # CONTAINS THE VALUE OF WHETHER IT IS 0 OR 1, EVERYTHING ELSE IS STRINGS ADDED ON
    kanji_two = predict_single(model_two["w"], model_two["b"], my_image)
    actual_image = max(kanji_one,kanji_two)
    print(kanji_one)
    print(kanji_two)
    if actual_image == 0:
        print('No kanji found')
    elif actual_image == kanji_one:
        print("Predicted kanji is 'day-kanji' ",kanji_one)
        # If message is found then display a message along with it's image and english meaning
        # Ask if this is correct, if yes then append to corresponding table with y value as 1
        # else append y value as 0 and append an image that is drawn,
        # this is assuming that the user is honest
        msg = " You have drawn a 'day-kanji', this is   "
    elif actual_image == kanji_two:
        print("Predicted kanji is 'week-kanji' ",kanji_two)





    # fname = r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\backupimages\kanji_3\image.jpeg'  # image_number increments by 1 at every save
    # image1.save(fname)  # save image
    # image = np.array(ndimage.imread(fname, flaten=False))  # extract image
    # my_image = scipy.misc.imresize(image, size=(64, 64)).reshape( (1, 64 * 64 * 3))  # resize to matrix and image size of 64x64
    # with h5py.File(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5',
    #                "r") as hdf: #Read the file which holds the data
    #     existing_size_x = np.array(hdf['kanji_one_x'][:])  # Extract the entire dataset for images data
    #
    #     existing_size_y = np.array(hdf['kanji_one_y'][:])  # Extract entire dataset for classification
    #
    # size = existing_size_x.shape  # Extract tuple which represents current size
    # current_row = size[0]  # Columns by rows, extract how many rows there are
    # new_row = current_row + 1  # The new row amount
    # existing_size_x = existing_size_x.reshape((1, 64 * 64 * 3 * current_row))  # Reshaping to keep same shape
    # existing_size_x = np.append(existing_size_x, my_image)  # Append the data onto already existing dataset
    # existing_size_x = existing_size_x.reshape((new_row, 64 * 64 * 3)) # Convert to keep h5 file format
    # existing_size_y.reshape(1, len(existing_size_y)) # 1 by how many there are numbers
    #
    # one = [1] #One for true
    # one = np.array(one) # Convert to fit numpy array
    # existing_size_y = np.append(existing_size_y, one)  # True value, therefore true
    # existing_size_y = existing_size_y.reshape(len(existing_size_y), 1)  # A column vector representing all image classifications
    #
    # with h5py.File(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5',
    #                "a") as hdf: # Appending images
    #     del hdf['kanji_one_x']  # Image data for images, delete as you cannot add new files
    #     del hdf['kanji_one_y']  # Whether each image is an accurate image or not, using binary classification
    #     hdf.create_dataset('kanji_one_x', data=existing_size_x, chunks=True, maxshape=(None, None))# No maxshape so you can append
    #     hdf.create_dataset('kanji_one_y', data=existing_size_y, chunks=True, maxshape=(None, None))# Keep column vector


def save_data_false():
    fname = r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\image.jpeg'  # image_number increments by 1 at every save
    image1.save(fname)  # save image

    image = np.array(ndimage.imread(fname, flatten=False))  # extract image
    my_image = scipy.misc.imresize(image, size=(64, 64)).reshape(
        (1, 64 * 64 * 3))  # resize to matrix and image size of 64x64

    with h5py.File(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5',
                   "r") as hdf:
        existing_size_x = np.array(hdf['kanji_one_x'][:])
        print(existing_size_x, "---------------------")
        existing_size_y = np.array(hdf['kanji_one_y'][:])
        print(existing_size_y, "---------------------")

    size = existing_size_x.shape
    current_row = size[0]
    new_row = current_row + 1
    existing_size_x = existing_size_x.reshape((1, 64 * 64 * 3 * current_row))
    existing_size_x = np.append(existing_size_x, my_image)
    existing_size_x = existing_size_x.reshape((new_row, 64 * 64 * 3))
    print(existing_size_x, "-------------")

    existing_size_y.reshape(1, len(existing_size_y))
    print(existing_size_y, "------------------------")
    zero = np.random.randint(1, size=1)
    existing_size_y = np.append(existing_size_y, zero)  # False value, therefore 0
    existing_size_y = existing_size_y.reshape(len(existing_size_y), 1)
    print(existing_size_y, "--------------------------------------")

    with h5py.File(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5',
                   "a") as hdf:
        del hdf['kanji_one_x']
        del hdf['kanji_one_y']

        hdf.create_dataset('kanji_one_x', data=existing_size_x, maxshape=(None, None))
        hdf.create_dataset('kanji_one_y', data=existing_size_y, maxshape=(None, None))


def displayInstructions():
    layout = [  # 2D List to contain data on instructional screen
        [sg.Text('Here is a list of Kanji characters available for drawing, in this section \n '
                 'you can draw any of these available 10 characters and the algorithm will \n '
                 'make a best guess of what you have drawn to practice')],  # Text information
        [sg.Image(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\kanjiList.gif')]
    ]
    window = sg.Window('Kanji List', layout)
    while True: #While window is available to be run
        event_start, value_start = window.read()
        if event_start in ('Exit', None): break #If user clicks on 'X' button to quit, stop loop and exits






# --- PIL
image1 = PIL.Image.new('RGB', (canvas_width, canvas_height), 'white')  # Variable which creates a background
draw = ImageDraw.Draw(image1)
Canvas.bind('<B1-Motion>', paint_in)  # Clicking left will call function 'paint_in'
Canvas.bind('<B3-Motion>', clear)  # Clcking right click will call function 'clear'
message = Label(master, text='Press and drag to draw; left click to draw')
message.pack(side=BOTTOM)
button_submit_true = Button(master, text="Submit Data True", command=save_data_true)
buttom_instructions_display = Button(master,text="Kanji Available",command=displayInstructions)
button_submit_true.pack(side=LEFT)
buttom_instructions_display.pack(side=LEFT)





#
# for i in range(1,161):
#     fname = r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\backupimages' + str(i) +'.jpeg'  # image_number increments by 1 at every save
#     image = np.array(ndimage.imread(fname, flaten=False))  # extract image
#     my_image = scipy.misc.imresize(image, size=(64, 64)).reshape(
#         (1, 64 * 64 * 3))  # resize to matrix and image size of 64x64
#     with h5py.File(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5',
#                    "r") as hdf:  # Read the file which holds the data
#         existing_size_x = np.array(hdf['kanji_one_x'][:])  # Extract the entire dataset for images data
#
#         existing_size_y = np.array(hdf['kanji_one_y'][:])  # Extract entire dataset for classification
#
#     size = existing_size_x.shape  # Extract tuple which represents current size
#     current_row = size[0]  # Columns by rows, extract how many rows there are
#     new_row = current_row + 1  # The new row amount
#     existing_size_x = existing_size_x.reshape((1, 64 * 64 * 3 * current_row))  # Reshaping to keep same shape
#     existing_size_x = np.append(existing_size_x, my_image)  # Append the data onto already existing dataset
#     existing_size_x = existing_size_x.reshape((new_row, 64 * 64 * 3))  # Convert to keep h5 file format
#     existing_size_y.reshape(1, len(existing_size_y))  # 1 by how many there are numbers
#
#     one = [1]  # One for true, for first 160
#     one = np.array(one)  # Convert to fit numpy array
#     existing_size_y = np.append(existing_size_y, one)  # True value, therefore true
#     existing_size_y = existing_size_y.reshape(len(existing_size_y),1)  # A column vector representing all image classifications
#
#     with h5py.File(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\Kanji.h5',
#                    "a") as hdf:  # Appending images
#         del hdf['kanji_one_x']  # Image data for images, delete as you cannot add new files
#         del hdf['kanji_one_y']  # Whether each image is an accurate image or not, using binary classification
#         hdf.create_dataset('kanji_one_x', data=existing_size_x, chunks=True,
#                            maxshape=(None, None))  # No maxshape so you can append
#         hdf.create_dataset('kanji_one_y', data=existing_size_y, chunks=True, maxshape=(None, None))  # Keep column vector

master.mainloop()


def send_data():
    # when update, it creates new, therefore drawing not saved, on clicking
    # button it refreshes the menu, therefore gone
    x, y = Canvas.winfo_rootx(), Canvas.winfo_y()  # get the top left coordinates of window
    im = pyautogui.screenshot(region=(x, y,  # fetch the image location
                                      Canvas.winfo_rootx() + Canvas.winfo_width(),
                                      Canvas.winfo_rooty() + Canvas.winfo_height()))
    box = (Canvas.winfo_rootx(), Canvas.winfo_rooty(), Canvas.winfo_rootx() + Canvas.winfo_width(),
           Canvas.winfo_rooty() + Canvas.winfo_height())  # coordinate of (x,y,image_width,image_height)
    grab = ImageGrab.grab(bbox=box)  # Take the image
    grab.save(r'C:\Users\Eimantas\Desktop\A Level CS Stuff\Project\NeuralNetWrk\datasets\week2\image1.png')
