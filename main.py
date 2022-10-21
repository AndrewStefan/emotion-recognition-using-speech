import os

from emotion_recognition import EmotionRecognizer

import logging
import time

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

rec = EmotionRecognizer(emotions=["neutral", "happy", "sad", "boredom"], n_rnn_layers=2, n_dense_layers=2,
                        rnn_units=128, dense_units=128)
# train the model
rec.train()
# check the test accuracy for that model
print("Test score:", rec.test_score())
# check the train accuracy for that model
print("Train score:", rec.train_score())


def analyse_audio(file_name):
    print("Prediction:", rec.predict(file_name))


class Event(LoggingEventHandler):
    def on_created(self, event):
        file_name = os.path.basename(event.src_path)
        path = os.path.join(os.getcwd(), '../study-buddy-agent/src/main/java/recording/done/')

        location = os.path.join(path, file_name)

        print(location)

        analyse_audio(location)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    path = '../study-buddy-agent/src/main/java/recording/done'
    event_handler = Event()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
