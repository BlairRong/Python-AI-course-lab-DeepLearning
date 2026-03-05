# Lab Requirements

In this lab, you will build a Convolutional Neural Network that classifies images as either cats or dogs. You will design, train, evaluate, and improve your own model without being given starter code. The goal is to understand how CNNs work in practice, not just to obtain high accuracy.

You will use the Dogs vs Cats dataset:
https://www.microsoft.com/en-us/download/details.aspx?id=54765
Download the dataset and organize the images into two folders, one for cats and one for dogs.

Before writing any code, inspect several images from each class. Pay attention to resolution differences, background complexity, lighting conditions, and variation in breeds and poses. Write a short paragraph explaining why this is not a trivial classification task despite having only two classes.

Next, preprocess the data. Resize all images to a fixed size and normalize the pixel values. Split the dataset into training and validation sets, for example 80 percent for training and 20 percent for validation. Explain in your report why normalization and dataset splitting are necessary for stable and fair training.

Design a CNN architecture with multiple convolutional layers followed by pooling layers. Increase the number of filters as the network gets deeper. After feature extraction, convert the feature maps into a format suitable for classification and add one or more fully connected layers. Include at least one regularization method such as dropout or batch normalization. The final layer must output a single probability value for binary classification. Explain each architectural choice in writing.

Compile the model using a suitable loss function for binary classification and train it for multiple epochs. Track both training and validation accuracy and loss. After training, analyze the learning curves and explain whether your model shows signs of underfitting or overfitting.

Evaluate the model on the validation set and inspect several correct and incorrect predictions. Discuss why certain images may have been misclassified. Reflect on whether dataset difficulty is driven more by visual ambiguity, background noise, or intra-class variation.

Finally, propose and implement at least one improvement, such as data augmentation or architectural changes. Retrain the model and compare results with your original baseline. Conclude with a short reflection on what you learned about CNNs and image classification from this experiment.

- **Answer please see the lab_report.md**
