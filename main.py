import os

from emotion_recognition import EmotionRecognizer

import logging
import time

from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler

from fer import FER
import cv2
import zmq
import sys

rec = EmotionRecognizer(emotions=["neutral", "happy", "sad", "boredom"], n_rnn_layers=2, n_dense_layers=2,
                        rnn_units=128, dense_units=128)
# train the model
rec.train()
# check the test accuracy for that model
print("Test score:", rec.test_score())
# check the train accuracy for that model
print("Train score:", rec.train_score())

face_emotion = "none"
global_socket = None


def analyse_audio(location, question_id):
    voice_emotion = rec.predict(location)
    print("Prediction voice:", rec.predict(location))
    print("Prediction face:", face_emotion)

    global_socket.send_string("123 " + question_id + " face emotion " + face_emotion + " voice emotion " + voice_emotion)

    print("sent: 123 " + question_id + " face emotion " + face_emotion + " voice emotion " + voice_emotion)


class Event(LoggingEventHandler):
    def on_created(self, event):
        question_id = os.path.basename(event.src_path)
        path = os.path.join(os.getcwd(), '../study-buddy-agent/src/main/java/recording/done/')

        location = os.path.join(path, question_id)

        analyse_audio(location, question_id)


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

            port = "5657"
            if len(sys.argv) > 1:
                port = sys.argv[1]
                int(port)

            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind("tcp://127.0.0.1:" + port)

            global_socket = socket

            detector = FER()

            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                emotion, score = detector.top_emotion(frame)
                if emotion:
                    res = emotion + ', ' + str(score)
                    print(res)
                    face_emotion = res
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    cap.release()
    cv2.destroyAllWindows()
    socket.close()
