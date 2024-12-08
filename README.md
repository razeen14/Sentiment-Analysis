# Sentiment Analysis on IMDB Reviews using LSTM

A deep learning project to classify movie reviews from the IMDB dataset as positive or negative. The model leverages an LSTM network to capture the sequential patterns in text data, ensuring accurate sentiment prediction.

# Overview

The Sentiment Analysis on IMDB Reviews using LSTM project aims to predict the sentiment of movie reviews as either positive or negative. Using the IMDB dataset, this project leverages the power of Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), to capture long-range dependencies and contextual meaning in text.

The model processes textual data by tokenizing the reviews, converting words into numerical representations through embedding layers, and padding the sequences to ensure uniform input size. The LSTM layer is used to learn sequential patterns and contextual relationships between words, while the dense layer outputs the sentiment classification.

# Features
Data Preprocessing: The IMDB reviews dataset is preprocessed through tokenization, padding, and word embeddings to prepare the text for the LSTM model.

Model Architecture: The model is built using an Embedding layer, an LSTM layer, and a Dense layer to predict sentiment.

Libraries: Built using TensorFlow/Keras for deep learning, Pandas and NumPy for data manipulation, and Scikit-learn for model evaluation.

# Workflow


1. Prepare the Data: The dataset contains a collection of movie reviews with labeled sentiments.
  
2. Model Building: The script ``` sentiment_analysis_on_imdb_reviews_with_lstm.py ``` trains the LSTM model on the IMDB reviews dataset.

3. Evaluate the Model: After training, the model’s performance can be evaluated using the validation or test data. The script outputs accuracy and other evaluation metrics.
   
4. Make Predictions: You can use the trained model to predict the sentiment of a new review.
   
5. Evaluate the model's performance on the test set.

# Model Architecture

The model uses the following layers:

1. Embedding Layer: Converts words into dense vectors of fixed size.

2. LSTM Layer: Learns sequential patterns in the review text.

3. Dense Layer: Outputs the sentiment classification (positive or negative).


# Evaluation

The model achieves high accuracy in predicting sentiments based on the IMDB reviews dataset, handling the sequential nature of text through the LSTM network.
