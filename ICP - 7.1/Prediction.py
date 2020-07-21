
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


text = ["A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump"]

# preprocessed to match saved model
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(text)
Y = tokenizer.texts_to_sequences(text)
Y = pad_sequences(Y, maxlen=28)

# call of saved model
model1 = tf.keras.models.load_model('model.h1')

# prediction for processed new text
pred = model1.predict_classes(Y)

# allow for named output instead of 0 or 1
classes = ['positive', 'negative']
print("The predicted result is ->", classes[pred[0]])


