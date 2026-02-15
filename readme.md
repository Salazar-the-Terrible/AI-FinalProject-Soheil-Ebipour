This project aims to produce a model that can tell apart cat images from dog images. To achieve this the kaggle "cat vs dog" dataset was used. The project uses a 5 layer convolutional neural network(cnn) to achieve its goal.
To download the dataset, use the following link:
https://www.microsoft.com/en-us/download/details.aspx?id=54765

Training images should be placed in the data folder and be divided into 2 subfolders named "cat" and "dog".

To run the training, simply run the cnn_train script.
There is an alternative classic-cnn-train script that uses canny transform.
To test, run the ui.py script.