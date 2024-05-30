# Machine-Learning-Engineering_An-ASL-Ensemble

# 1 Introduction

This project pertains to the multi-classification problem of static American Sign Language (ASL) alphabet signs.
While this project has been done before, we are utilizing an ensemble approach with a CNN, transfer learning with
MobileNet, KNN, and VGG. This approach should allow us to achieve a high accuracy regardless of the individual
performance of the models.

The classification of ASL signs has been a common project for various machine learning uses as it has both a
seemingly positive social impact and is, in reality, a simple multi-classification problem.


# 2 Approach: Dataset(s) & Pipeline(s)

To solve our problem of multi-classification, we decided to utilize an ensemble approach which we had not seen
previously in the literature. We decided on using four models together: a Convulational Neural Network (CNN),
a MobileNet (via transfer learning), a K-Nearest Neighbor model, and a Visual Geometry Group (VGG) model.
These models, once trained, would be put into an ensemble that would utilize majority voting for classification.

The start of our pipeline included the loading and pre-processing of our data. We opted to only due pixel
normalization for the data and flattening of the data that needed it. The neural networks used the image data
rather than the flattened data.

The dataset we utilized was a dataset compiled by Kaggle user Akash. This dataset included 87,000 200x200
RGB images of the ASL alphabet as well as the signs for delete, space, and photos of nothing at all for training
models [2]. This created 29 classes for images.

For the data, we split it into a 80%, 10%, 10% ratio for training, testing, and validation. We had 69,600 training
samples, 8,700 validation samples, and 8,728 testing samples. Our data is equally distributed among all classes.

Once each directory was set up with the images, we created three data generators that allowed us to control
the size of the image and implement any data augmentation if we wanted to do so later. Since each image was
originally 200x200 pixels, we had to turn them into 32x32 due to memory constraints. We did this in the generator
along with setting our class mode to categorical as we had mutually exclusive labels.

Our pipeline also included the creation of multiple models to be done in an ensemble. We will discuss each of the
models in the following subsections including their results and architecture. The models were made independently,
tested, and then moved into the ensemble so that we could track their individual performance as well as the
ensemble performance.

## 2.1 CNN
Our CNN had the architecture seen in Table 1. This allowed us to have 359,581 total parameters with 359,133 of
those being trainable. We utilized the categorical crossentropy loss function with the adam optimizer.

## 2.2 Transfer Learning with MobileNet
MobileNets are an architecture type developed by Howard et al. as ”light weight deep neural networks.” This
architecture consists of multiple convulational depthwise and pointwise layers with average pooling and the softmax
function at the end. The full architecture can be found in their paper [1]. We added two more layers on top of the
MobileNet architecture. These were a global average pooling 2D later and a dense layer with softmax activation.

## 2.3 KNN
The KNN model required the data to be flattened. To do this, we looped through the batches from the generators
and appended each flattened batch to a list while keeping the labels the same. The model itself had a grid search
implemented for tuning the number of neighbors hyperparameter. The grid search presented the best n neighbors
as 1 using the validation set.

## 2.4 Transfer Learning with VGG19
We utilized the Visual Geometry Group (VGG) model with 19 layers for our last model. This model is a deep
convulational neural network generally used for image classification. This model was developed by Simonyan et
al. (2015) [5] with an architecture of maxpools between groups of convulational layers, three hidden layers and a
softmax activation. Our VGG also had some custom dense and dropout layers as well as early stopping implemented
to help prevent overfitting.


# 3 Evaluation Methodology

Our evaluation consists of finding the F1-Score, Recall, and accuracy of each of our models separately and our
ensemble as a whole. The F1-Score was chosen for its usefulness in classification problems and ability to give
insight into precision and recall. Recall was chosen to find the rate at which true positives were chosen. Accuracy
was chosen as a general performance measurement against other models.

As a baseline, we defer to Pathan et al. (2023) in their gathering of various models and their accuracies for this
type of task. The first baseline we chose were various models from Kaggle users on the same dataset which ranged
from 95.81% - 96.96%. The second baseline we chose was Pathan et al.’s multi-headed CNN as it performed better
than the other models that they had referenced in their paper at 98.98% [3].

The splitting of the data was 80%, 10%, 10%. The overall distribution of this data ended up being 69,600 images
for training, 8700 images for validation, and 8728 images for testing. There was an additional testing dataset that
we created ourselves. For this test set, one image of each of the signs (and ’nothing’) was taken via a USB web
camera and placed in its own test folder. Overall, this custom test dataset had 29 images in it (one per class).


# 4 Results

## 4.1 Individual Models
Of the individual models, the transfer learning with VGG19 and the CNN performed the best at both above 99.1%
accuracy while the MobileNet transfer learning performed the worst at 85.71%, which was well below our baselines

## 4.2 The Ensemble
Overall, our results were very promising in terms of f1-score, recall, and accuracy. We scored roughly the same
on all 3 (see table 3 for results) which seems to indicate that our data was well balanced and well-performing.
Interestingly, when introducing a test set of one of us making the symbols, the ensemble and individual models
performed poorly. With this test set, our ensemble achieved an accuracy of 10.34%.

Our baselines of 95.81% - 96.96% and 98.98% were both improved upon with our ensemble method. The first
baseline came from the same dataset with a KNN model, ORB model [4], and an MLP model from Kaggle users
[3]. The second baseline was the accuracy from a multi-headed CNN network which was trained with a different
dataset, but for a similar task [3].


# 5 Conclusions

Our first key takeaway is that our ensemble was too complex for very little gain. By this, we mean that we had
already achieved 99.15% accuracy with the Transfer Learning VGG19 model and 99.10% with the CNN, but our
ensemble was only slightly above that at 99.34% with more work and computational resources behind it.

Second, we saw a problem with generalization in a different test dataset that we created ourselves. This could
be a problem with the data that we created (such as performing the sign incorrectly) or an issue with the model
itself. More work would need to be done to figure out the actual cause for the lack of generalization.

Finally, it was also difficult to train the various models on images that were too large (we had to scale them
down to 32x32) due to various memory issues. We utilized a machine with a dedicated GPU and still had to scale
down the data and utilize batches in order to be able to train the models, specifically the KNN which had to utilize
flattened data for its training. It was simpler to train the neural networks as the data could come directly from the
directory instead of loading the entire dataset into memory.


# References

[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, and Hartwig Adam. Mobilenets: Efficient convolutional neural networks for mobile vision applications,
2017.

[2] Akash Nagaraj. Asl alphabet, 2018. Available at https://www.kaggle.com/datasets/grassknoted/asl-alphabet.

[3] Refat Khan Pathan, Munmun Biswas, Suraiya Yasmin, Mayeen Uddin Khandaker, Mohammad Salman, and
Ahmed A. F. Youssef. Sign language recognition using the fusion of image and hand landmarks through multiheaded convolutional neural network. Scientific Reports, 2023.

[4] Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. Orb: An efficient alternative to sift or surf.
In 2011 International Conference on Computer Vision, pages 2564–2571, 2011.

[5] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition,
2015.
