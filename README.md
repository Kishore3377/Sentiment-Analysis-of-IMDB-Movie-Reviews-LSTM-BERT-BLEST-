A Streamlit web app that classifies IMDB movie reviews as positive or negative using three models and compares them side by side:

LSTM: a Bidirectional LSTM built with TensorFlow/Keras
BERT: a fine-tuned BERT transformer (Hugging Face Transformers + PyTorch)
BLEST (BERT-LSTM Ensemble Sentiment Technology): an ensemble that takes the probabilities from the LSTM and BERT models and passes them to a meta-classifier, which makes the final prediction
Features
Analyse Review: type or paste a review and see each model's prediction and confidence, plus a verdict showing whether the models agree
Batch analysis: upload a CSV with a review column to classify many reviews at once and download the results
Analytics dashboard: sentiment trend, most frequent words, model agreement rate and review history for the current session
Model comparison: accuracy, precision, recall and F1 score for LSTM, BERT and BLEST in a chart and a table
Movie recommendations: suggested films based on the predicted sentiment
How BLEST works
The LSTM and BERT models each output the probability that a review is positive.
Six features are built from those two probabilities: both probabilities, their mean, their absolute difference, their product and a weighted average (0.4 × LSTM + 0.6 × BERT).
A meta-classifier trained on these features produces the final label and confidence.
