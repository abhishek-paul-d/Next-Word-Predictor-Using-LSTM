# Next Word Predictor Using LSTM

This project uses an LSTM (Long Short-Term Memory) neural network to predict the next word in a sequence based on Shakespeare's Hamlet text.

## About LSTM

LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) architecture designed to remember long-term dependencies in sequence data. It is particularly useful for natural language processing tasks like next word prediction because it can capture the context of words over long sequences.

The model achieved an accuracy of 0.7894 after 221 epochs of training.

## Usage

1. Ensure you have the required dependencies installed:
   - TensorFlow
   - NLTK
   - NumPy
   - scikit-learn

2. Run the `Next_Word_Predictor_Using_LSTM.ipynb` notebook to train the model and save it as `next_word_lstm.h5`.

3. Use the `predict_next_word` function to predict the next word given an input text.

## Example

```python
input_text = "this is the most"
next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
print(f"Next Word Prediction: {next_word}")
```

## Dependencies

- TensorFlow
- NLTK
- NumPy
- scikit-learn

## Files

- `Next_Word_Predictor_Using_LSTM.ipynb`: Jupyter notebook containing the code for data collection, preprocessing, model building, training, and prediction.
- `hamlet.txt`: Text file containing Shakespeare's Hamlet.
- `next_word_lstm.h5`: Trained LSTM model.
- `tokenizer.pickle`: Saved tokenizer.
- `app.py`: Streamlit app (under development).
- `README.md`: This file.
