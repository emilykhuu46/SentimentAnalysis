# Sentiment Analysis
This repository contains an attempt to use the Keras NLP BERT model to train and predict data from the "Sentiment Analysis on Movie Reviews" Kaggle challenge. (https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/overview) 
# Overview
The task given by the Kaggle challenge was to generate a sentiment analysis for a given phrase. The sentiment labels ranged from 0 to 4 with 0 meaning negative, 1 meaning somewhat negative, 2 meaning neutral, 3 meaning somewhat positive, and 4 meaning positive. The task requires me to use classification model methods, and I approached the problem by using the Keras BERT model which is a well trained neural network for natural language processing model. The model succeeded and was able to predict sentiments with a 66% accuracy.
# Summary of Work Done
**Data**  
The data was 2.44 MB and given in two tab separated value (TSV) files, one for training and one for testing. The train.tsv had 156060 rows with four columns labeled: phrase IDs, sentence IDs, phrases, and a sentiment. The test.tsv had 66292 rows with only three columns labeled: phrase IDs, sentence IDs, and phrases. 

**Preprocessing and Clean up**  
I had to unzip the read the files before having to separate phrases into their own directories by sentiment labels via the text data loading function. The five resulting directories were then moved into one main directory. Also for text data to be read by the machine neural networks, they must be tokenized and hashed, so this was done to the data before loading the data into the model.   

**Data Visualization**  
Through the data visualization process, I learned that each phrase belongs to a sentence and some sentences had more instances of phrases than others, specifically the first several sentences. This is true for both the training and testing dataset. 
![image](https://github.com/emilykhuu46/SentimentAnalysis/assets/123412398/2a3f4991-6331-4fa9-9a15-2a402bb0e26b)

![image](https://github.com/emilykhuu46/SentimentAnalysis/assets/123412398/9110095c-bbc1-4256-b7f5-def9d04d372c)

Overall however, the training dataset seems to have an overall neutral sentiment as 51% of all the phrases have the label 2.
![image](https://github.com/emilykhuu46/SentimentAnalysis/assets/123412398/5fe11424-60b0-47bc-8b3d-e47a709e9419)

**Problem Formulation**
The Kaggle problem required me to have an input be a Phrase ID with a phrase and the resulting output be the predicted sentiment label. The main model that was used was the BERT NLP model from Keras. The loss function used was sparse_categorical_crossentropy which measures the difference between the predictions and actual values; it is common for multiclass classification problems like mines. The optimizer used was the adam optimizer which is used for governing update strategies for neural network weights while training. It is an effective optimization algorithm for deep learning tasks. The metric used to evaluate the performance of the bert model was accuracy.  

**Training**
The training portion encountered a lot of difficulties including the datasets having different shapes-due to the test data not having a sentiment column and having to account for that. The number of phrase instances also created issues during the training process as it was a lot of information to train, so it was shortened into a smaller subset. During the initial training for the train dataset there were three epochs and each epoch took about 30 minutes each for a total of 90 minutes of training. With each epoch, the loss decreased while the accuracy increased. The training for the test dataset took 50 minutes. Originally, the time it would have taken for one epoch of the full train dataset was 17 hours. I trained the model until I reached the highest accuracy it could produce, which in this case was 66%.  

**Performance Comparison**
Accuracy was the key performance metric for this task. The model had to predict a sentiment label and then compare it to the actual sentiment label.

This is an example of the kind of comparisons that were done:  
![image](https://github.com/emilykhuu46/SentimentAnalysis/assets/123412398/d4fbbe36-3770-48c3-bc9e-83fdc06c0801)

In the end, because the data for training the model was shortened and 51% of the overall sentiments were neutral, the model seems to be overpredicting 2s as well as 1s while underpredicting 0s, 3s and 4s. 0s and 4s actually did not get predicted at all. If the full dataset was able to be trained and predicted, the accuracy would most likely be higher.  

![image](https://github.com/emilykhuu46/SentimentAnalysis/assets/123412398/ac736ca5-99e1-420c-b4ed-19891823c996)


**Conclusion**
Overall, a BERT based model is suitable for the classification task. My first attempt at training the model and predicting sentiments ended up with all the predictions beings 2s. The first training attempt utilized the scikitlearn library more with functions such as cross_val_score etc. The second attempt was more accurate and it relied on manual data splitting before using the BERT tokenizer and hashing the tokenized sequences for training. In conclusion, the manual data splitting method was better for the task, but decreasing the training subset will greatly decrease the accuracy no matter what method you use.  

**Future Work**
I would like to try other LSTM methods in the future and possibly build my own model rather than using the prebuilt BERT model. ChatGPT is also known to be able to do sentiment analysis and other nlp features so I could try that in the future as well.  
# How to Reproduce Results
To reproduce my results you need to begin by downloading and unzipping the dataset. Afterwhich, you can visualize the data using pandas and matplotlib and see how many sentiments the train dataset has as well as how many instances of phrases you have to work with. Next, you separate your rows into directories based on their sentiment labels. You then have to pip install keras, tensorflow, and transformers to use the modules. Next, you have to decrease your dataset from the full length to a shortened length of 1000 or 50 or 20 instances. Next, you have to initialize the tokenizer and load the bert_model. Then, you would have to preprocess the data by tokenizing the phrases of both the train and test dataset via the tokenizer and then hash the tokenized data. Next, you would compile the model with the adam optimizer, the sparse_categorical_crossentropyloss function, the accuracy metric, and then fit it with three epochs each. Depending on the number of instances you use, the model will train quickly or take 2 hours. Then you use the trained model on the test dataset to predict the test data. You can see the accuracy of the predictions using scikitlearn accuracy_score. Then using confusion_matrix you can compare the true sentiment labels and the predicted sentiment labels to see what parts are accurate and what parts are inaccurate. The last thing you will do is to create a graph to visualize the true vs. predicted sentiment labels.   

**Overview of files in Repository**  
+ unzip.ipynb: unzipping and preparing the data so it can be manipulated.
+ visualization.ipynb: takes train.tsv and test.tsv and visualizes the information.
+ training-1.ipynb: contains some data preprocessing where phrase rows were sorted into directories based on their sentiment labels. Also contains the first attempt at using the BERT model to train the data and predict sentiments for the test dataset.
+ training-2.ipynb: contains the second attempt at using the BERT model to train and predict but it is modified a little to increase accuracy.

**Software Setup**  
Some packages required for this task are NumPy, Pandas, ScikitLearn, Tensorflow, Keras, and Matplotlib. Some packages such as NumPy and Matplotlib are already installed in python so simply importing them is all that is needed but other packages like Tensorflow must be installed first before importing.  

**Data**  
The data can be downloaded from the Sentiment Analysis on Movie Reviews kaggle challenge page. 
https://www.kaggle.com/competitions/sentiment-analysis-on-movie-reviews/data 

**Training**  
To train the model, I would recommend decreasing the dataset from the full length to a shortened length of 1000 or 50 or 20 instances. Next, you have to initialize the tokenizer and load the bert_model. Then, you would have to preprocess the data by tokenizing the phrases of both the train and test dataset via the tokenizer and then hash the tokenized data. Next, you would compile the model with the adam optimizer, the sparse_categorical_crossentropyloss function, the accuracy metric, and then fit it with three epochs each. Depending on the number of instances you use, the model will train quickly or take 2 hours. Then you use the trained model on the test dataset to predict the test data.  

**Performance Evaluation**  
To complete a performance evaluation for this task you can use the accuracy_score function from scikitlearn to view the accuracy of the model each time you train it. Then using confusion_matrix, which is also from scikitlearn, you can compare the true sentiment labels and the predicted sentiment labels to see what parts are accurate and what parts are inaccurate. The last thing you will do is to create a plot to visualize the true vs. predicted sentiment labels.  

# Citations
[1] BERT Tokenizer. (n.d.). Keras Documentation. https://keras.io/api/keras_nlp/models/bert/bert_tokenizer/

[2] Keras. (n.d.). IMDb Movie Reviews Dataset. Keras Documentation. https://keras.io/api/datasets/imdb/

[3] Pang and L. Lee. 2005. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In ACL, pages 115–124.

[4] Salva, G.(n.d.). An Easy Tutorial About Sentiment Analysis with Deep Learning and Keras. Towards Data Science. https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91

[5] Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C., Ng, A., & Potts, C. (2013). Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank. Conference on Empirical Methods in Natural Language Processing (EMNLP 2013).

[6] Will Cukierski. (2014). Sentiment Analysis on Movie Reviews. Kaggle. https://kaggle.com/competitions/sentiment-analysis-on-movie-reviews  
