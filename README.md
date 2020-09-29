# Deep Learning with Amazon SageMaker Studio

In this workshop, you’ll learn how to use Amazon SageMaker to build, train and tune deep learning models using built-in algorithms as well as custom Tensorflow code. 

![SageMakerImage](https://sagemaker-workshop.com/images/sm-overview.png)

Amazon SageMaker is a fully managed service that provides machine learning (ML) developers and data scientists with the ability to build, train, and deploy ML models quickly. SageMaker provides you with everything you need to train and tune models at scale without the need to manage infrastructure. You can use Amazon SageMaker Studio, the first integrated development environment (IDE) for machine learning, to quickly visualize experiments and track training progress without ever leaving the familiar Jupyter Notebook interface. Within SageMaker Studio, you can use SageMaker Experiments to track, evaluate, and organize
experiments easily.


### In this workshop, you’ll go through the following steps:
1. Set up Amazon SageMaker Studio to build and train your deep learning models. 
2. Download a public dataset using Amazon SageMaker Studio Notebook and upload it to Amazon S3.
3. Create an Amazon SageMaker Experiment to track and manage training jobs
4. Run Amazon SageMaker large-scale training and model tuning:
	a. Using Built-in Algorithms. 
	b. Using custom code for training with Tensorflow
5. Improve accuracy by running a large-scale Amazon SageMaker Automatic Model Tuning job to find the best model hyperparameters

### Dataset: 
You’ll be using the CIFAR-10 dataset to train a model with a SageMaker Built-in algorithm as well as with TensorFlow to classify images into 10 classes. This dataset consists of 60000 32x32 color images, split into 40000 images for training, 10000 images for validation
and 10000 images for testing.
![Cifar10Image](https://cdn.analyticsvidhya.com/wp-content/uploads/2020/02/1_sGochNLZ-qfesdyjadgXNw.png)


## Before you begin
Before you begin this tutorial, you must have an AWS account. If you do not already have an account, follow the instructions for eventengine. 

## Step 1
Follow instructions for setting a notebook in SageMaker Studio

## Step 2 - Setting up Permissions and environment variables
In this step we setup some prequisites so our notebook can authenticate and link to other AWS Services. There are 3 parts to this: 
- Define the S3 bucket that you want to use for training and model data.
- Create a role for permissions
- Specify the built-in algorithm that you want to use. 

Add the following code to your notebook:
```
%%time
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
print(role)

sess = sagemaker.Session()
bucket=sess.default_bucket()
prefix = 'ic-fulltraining'
```

```
from sagemaker.amazon.amazon_estimator import get_image_uri

training_image = get_image_uri(sess.boto_region_name, 'image-classification', repo_version="latest")
print (training_image)
```

## Step 3 Download the dataset and upload it to Amazon S3
In this step we'll be downloading the CIFAR-10 training dataset and upload that to Amazon S3. As mentioned above this dataset consists of 10 classes with 40,000 images for training, 10,000 for validation and another 10,000 for testing. We can download all of these images from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and then convert them into to the recommended input format for image classification which is Apache MXNet RecordIO. We can also just download the CIFAR-10 dataset already in the RecordIO, which is what we'll be doing for this workshop.

 

