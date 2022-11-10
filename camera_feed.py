import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


def mediapipe_detection(imagen, modelo):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen.flags.writeable = False
    resultados = modelo.process(imagen)
    imagen.flags.writeable = True
    imagen = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    return imagen, resultados


def draw_styled_landmarks(imagen, resultados, mp_drawing, mp_holistic):
    mp_drawing.draw_landmarks(imagen, resultados.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    mp_drawing.draw_landmarks(imagen, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(imagen, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(imagen, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 30), thickness=2, circle_radius=2)
                              )


def extract_keypoints(resultados):
    pose = np.array([[resultado.x, resultado.y, resultado.z, resultado.visibility]
                    for resultado in resultados.pose_landmarks.landmark]).flatten() if resultados.pose_landmarks else np.zeros(132)
    face = np.array([[resultado.x, resultado.y, resultado.z] for resultado in resultados.face_landmarks.landmark]).flatten(
    ) if resultados.face_landmarks else np.zeros(1404)
    lh = np.array([[resultado.x, resultado.y, resultado.z] for resultado in resultados.left_hand_landmarks.landmark]).flatten(
    ) if resultados.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[resultado.x, resultado.y, resultado.z] for resultado in resultados.right_hand_landmarks.landmark]).flatten(
    ) if resultados.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame




# cap = cv2.VideoCapture(0)

def generate(cap,sentence):
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    
    actions = np.array(['Hola', '¿Cómo estás?', '¿Cuál?'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
        activation='relu', input_shape=(30, 1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.load_weights('action.h5')

    colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
    sequence = []
    
    threshold = 0.8
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            # print(results)
            # draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
            keypoints = extract_keypoints(results)
            # sequence.insert(0,keypoints)
            # sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])

                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            (flag, encodedImage) = cv2.imencode(".jpg",image)
            if not flag:
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                bytearray(encodedImage) +b'\r\n') 
