import os

from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC

if __name__ == '__main__':

    rec = EmotionRecognizer(emotions=["neutral", "happy", "sad", "boredom"], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
    # train the model
    rec.train()
    # check the test accuracy for that model
    print("Test score:", rec.test_score())
    # check the train accuracy for that model
    print("Train score:", rec.train_score())

    file_name = os.path.join(os.getcwd(), '1.wav')

    print("Prediction:", rec.predict(file_name))
