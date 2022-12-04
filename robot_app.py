import copy
import itertools
import pickle

import cv2 as cv
import numpy as np
import mediapipe as mp

import RPi.GPIO as GPIO
import time
import RPi.GPIO as GPIO
import time
  
#设置 GPIO 模式为 BCM
GPIO.setmode(GPIO.BCM)
  
#定义 GPIO 引脚
GPIO_TRIGGER = 26
GPIO_ECHO = 19
  
#设置 GPIO 的工作方式 (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

def main():
    GPIO.setwarnings(False)
    AIN1 = 5
    AIN2 = 6
    PWMA = 13
    BIN1 = 20
    BIN2 = 21
    PWMB = 12
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PWMA, GPIO.OUT)
    GPIO.setup(AIN1, GPIO.OUT)
    GPIO.setup(AIN2, GPIO.OUT)
    GPIO.setup(PWMB, GPIO.OUT)
    GPIO.setup(BIN1, GPIO.OUT)
    GPIO.setup(BIN2, GPIO.OUT)
    frequency = 50
    stop_dc = 0
    half_dc = 20
    full_dc = 30
    pA = GPIO.PWM(PWMA, frequency)
    pA.start(0)
    pB = GPIO.PWM(PWMB, frequency)
    pB.start(0)

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

    gesture_label = 'stop'
    previous_label = 'stop'
    code_run = True
    while code_run:
        time.sleep(0.5)
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

        #print(gesture_label)

        cv.imshow('Hand Gesture Recognition', debug_image)

        if(gesture_label=='stop'and previous_label!='stop'):
            stop(pA)
            stop(pB)
            print('stop')

        elif(gesture_label=='move' and previous_label!='move'):
            clockwise(pA, full_dc, AIN1, AIN2)
            clockwise(pB, full_dc, BIN1, BIN2)
            print('move')    

        elif(gesture_label=='left' and previous_label!='left'):
            clockwise(pA, full_dc, AIN1, AIN2)
            clockwise(pB, half_dc, BIN1, BIN2)
            print('left')

        elif(gesture_label=='right' and previous_label!='right'):
            clockwise(pA, half_dc, AIN1, AIN2)
            clockwise(pB, full_dc, BIN1, BIN2)
            print('right')

        previous_label = gesture_label
        
        while gesture_label=='move' or gesture_label=='stop':
            dist = distance()
            print("Measured Distance = {:.2f} cm".format(dist))
            if(dist<50):
                clockwise(pA, full_dc, AIN1, AIN2)
                clockwise(pB, full_dc, BIN1, BIN2)
                print('distance less than 50, move')
            if(dist>60):
                counterclockwise(pA, full_dc, AIN1, AIN2)
                counterclockwise(pB, full_dc, BIN1, BIN2)
                print('distance more than 60, back')    
            time.sleep(1)   

    cap.release()

    pA.stop()
    pB.stop()

    GPIO.output(AIN2, GPIO.LOW)
    GPIO.output(AIN1, GPIO.LOW)
    GPIO.output(BIN2, GPIO.LOW)
    GPIO.output(BIN1, GPIO.LOW)

    GPIO.cleanup()

def distance():
    # 发送高电平信号到 Trig 引脚
    GPIO.output(GPIO_TRIGGER, True)
  
    # 持续 10 us 
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
  
    start_time = time.time()
    stop_time = time.time()
  
    # 记录发送超声波的时刻1
    while GPIO.input(GPIO_ECHO) == 0:
        start_time = time.time()
  
    # 记录接收到返回超声波的时刻2
    while GPIO.input(GPIO_ECHO) == 1:
        stop_time = time.time()
    # 计算超声波的往返时间 = 时刻2 - 时刻1
    time_elapsed = stop_time - start_time
    # 声波的速度为 343m/s， 转化为 34300cm/s。
    distance = (time_elapsed * 34300) / 2
  
    return distance


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

def clockwise(p, dc, in1, in2):
    p.ChangeDutyCycle(dc)
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    return None

def counterclockwise(p, dc, in1, in2):
    p.ChangeDutyCycle(dc)
    GPIO.output(in2, GPIO.HIGH)
    GPIO.output(in1, GPIO.LOW)
    return None

def stop(p):
    p.ChangeDutyCycle(0)
    return None




if __name__ == '__main__':
    main()

