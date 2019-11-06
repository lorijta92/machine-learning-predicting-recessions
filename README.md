# Predicting U.S. Recessions with Machine Learning

## Goal

Use machine learning to create a model to predict the likelihood of a U.S. recession within 1, 2, and 4 quarters. 


## Data Preparation

Data was collected from the Federal Reserve Economic Data (FRED) online database and cleaned in Python notebooks. Depending on the feature, data came in a daily, monthly, and quarterly frequency, some dating back to the 1940s, some only to the 1970s. The two crucial data sets (GDP and recession dates) came in quarterly format and only extended to the 1970s. Therefore, all other data was aggregated to the quarter level and because the model can not take null values, dates prior to the 1970s were dropped.

Now, each row represented one quarter, with the target column indicating 1 for a recession, and 0 for not a recession. However, as a recession is a function of what happened in the quarters prior, we needed to shift the target column so that the rest of the data would point ahead to whether or not there was a recession. Because we wanted to make predictions based on 1, 2, and 4 quarters, three different shifts were made using the same time frames with `pd.shift()`. 

The shift created more null values that needed to be dropped, but because the data set was already quite small, three different X matrices were created for each shift to keep as much of the data as possible. Once all variables were defined, the target columns were dropped, and the three data sets were split into training and testing sets, and normalized.


## Model Building

For this time series forecast, a recurrent neural network (RNN) was used, as this model would be able to consider preceding inputs in addition to the current input and therefore make predictions based on the entire context of the data set. Because RNNs require a 3-dimensional array, the data was reshaped before being fed into the model. 

We initialized our model using the `Sequential` model, and then proceeded to add `LSTM`, `Dropout`, and `BatchNormalization` layers. `return_sequences` was set to `True` in the LSTM layers so that the next LSTM layer would have a 3-dimensional input, and a `BatchNormalization` layer was added to normalize the hidden state outputs. Dropout layers were used to avoid training the model too heavily on any one feature, and we closed out the  model with `Dense` layers. 

The model was compiled and then trained on each of the three X matrices, using a validation split of 20%. It was at this point that two notebooks were created to train the model with shuffled data and non-shuffled data. Because recessions are so heavily tied to sequence, we wanted to maintain that sequence when training the model, but also wanted to compare the model performance to shuffled data. 


## Evaluating and Predicting

The model scored high for all time frames, and with both shuffled and non-shuffled data, with accuracy being 90% or higher. However, because recessions are rare events, the model would’ve scored high just by predicting “0” all the way down. What really mattered was how many times our model correctly predicted a recession (“1”). To evaluate the “true positives”, we used sklearn’s `confusion_matrix` and `classification_report`. 

To get a visual sense of this, a line chart was also made to show the average of both correctly predicted recessions and correctly predicted non-recessions. For correctly predicting recessions one quarter out, the non-shuffled model worked best, with 73% accuracy. Because a recession is technically defined as a fall in GDP for two consecutive quarters, to predict a recession one quarter out requires knowing what is happening in the immediate preceding quarter. However, models predicting recessions two and four quarters out had higher accuracy with shuffled data. 
![plot](https://github.com/lorijta92/machine-learning-predicting-recessions/blob/master/images/plots/shuffle_noShuffule_vs20.png?raw=true)
