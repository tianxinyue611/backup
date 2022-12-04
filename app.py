import copy
import itertools
import pickle

import cv2 as cv
import numpy as np
import mediapipe as mp

def main():

    cap = cv.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    filename = 'rand_forest.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    while True:
        key = cv.waitKey(1)
        if key==27:
            break

        # camera capture
        ret, image = cap.read()

        if not ret:
            continue

        image = cv.flip(image, -1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                brect = cal_bounding_rect(debug_image, hand_landmarks)
                landmark_list = cal_landmark_list(debug_image, hand_landmarks)

                # normalize coordinates
                preprocess_landmark_list = preprocess_landmark(landmark_list)
                preprocess_landmark_list = np.array(preprocess_landmark_list).reshape(1,-1)
                predict_list = loaded_model.predict_proba(preprocess_landmark_list)[0]

                max_index = np.argmax(predict_list)

                if(max_index==0):
                    gesture_label = "stop"
                elif(max_index==1):
                    gesture_label = "move"
                elif(max_index==2):
                    gesture_label = "left"
                elif(max_index==3):
                    gesture_label = "right"

                print(gesture_label)

                debug_image=cv.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]),
                             (0, 255, 0), 4)

                text_position = (brect[0]+50, brect[1]+10)
                debug_image=cv.putText(debug_image, gesture_label, text_position, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 3)

                mp_drawing.draw_landmarks(
                    debug_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        cv.imshow('Hand Gesture Recognition', debug_image)


        #control robot


    cap.release()




def cal_bounding_rect(image, landmarks):
    img_width = image.shape[1]
    img_height = image.shape[0]

    landmark_array = np.empty((0,2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x*img_width), img_width-1)
        landmark_y = min(int(landmark.y*img_height), img_height-1)

        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x,y,w,h = cv.boundingRect(landmark_array)

    return [x,y,x+w, y+h]

def cal_landmark_list(image, landmarks):
    img_width = image.shape[1]
    img_height = image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x*img_width), img_width-1)
        landmark_y = min(int(landmark.y*img_height), img_height-1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def preprocess_landmark(landmark_list):
    temp_lm_list = copy.deepcopy(landmark_list)

    base_x=0
    base_y=0

    for index, landmark_point in enumerate(temp_lm_list):
        if index ==0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_lm_list[index][0] = temp_lm_list[index][0] - base_x
        temp_lm_list[index][1] = temp_lm_list[index][1] - base_y

    temp_lm_list = list(itertools.chain.from_iterable(temp_lm_list))

    max_value = max(list(map(abs, temp_lm_list)))

    def normalize_(n):
        return n/max_value

    temp_lm_list = list(map(normalize_, temp_lm_list))

    return temp_lm_list





if __name__ == '__main__':
    main()