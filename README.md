# Tweets Classification
## Tweets Sentiment Analysis

This repository contains code developed to classify the sentiment of tweets using various Machine Learning techniques. The Sentiment140 dataset used for training and evaluation is an extensive collection of tweets, each labeled with their sentiment: positive, negative, or neutral. 

The implemented processing pipeline includes: 

- [Data loading and preprocessing](#loading-data)
- [Data splitting](#data-splitting)
- [Sentiment Classification Model Training](#model-training)
- [Performance Evaluation](#evaluation)
- [New Tweet Prediction](#prediction)

The machine learning models provided in the code include popular algorithms, such as Naive Bayes, Support Vector Machines, and Random Forests.

You can leverage this codebase for various tasks such as identifying customer sentiment about a product, detecting hate speech, tracking misinformation spread, or understanding public opinion on current events.

## Getting Started

### Prerequisites

Before running the code, ensure that the following libraries are installed.

```
numpy
pandas
matplotlib
re
spacy
nltk
seaborn
string
textblob
emoji
sklearn
```

If not, you can install these prerequisites using pip:

```
pip install -r requirements.txt
```

### Usage

Clone this repository and install the necessary dependencies. You can then train the model using the provided dataset or with your custom dataset. The trained model can be used to classify new tweets.
Please refer to the following sections for details about each step of the processing pipeline.

#### Loading and preprocessing the data

The tweet dataset is loaded into memory and cleaned by removing noise, stop words and other irrelevant information. This ensures the dataset is ready for model training and testing.

```python
# Reading The DataSet
Data = pd.read_csv('Tweets.csv', encoding ='ISO-8859-1', header = None)
```

#### Splitting data

To ensure that the model is not overfitting the training data, the dataset is split into training and test sets.

#### Training the model

The code includes implementations of several popular machine learning models. This allows the user to select a model which best suits their needs.

#### Model Performance Evaluation

The classification model is evaluated on the test set to measure its accuracy on unseen data.

#### New tweet prediction

The trained model can predict the sentiment of a new tweet.

```python
Predi = Vector.transform(["your tweet here"])
prediction = Class.predict(Predi)
print(prediction)
```

## Authors

- Your Name - Complete work - [Your Github Username](https://github.com/Your Github Username)

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Acknowledge any articles, papers, or websites that have contributed to this project.

  
