import cv2
import mediapipe as mp
import pandas as pd

capture = cv2.VideoCapture(0)
mp_hand = mp.solutions.hands
hands = mp_hand.Hands()
mp_draw = mp.solutions.drawing_utils

columns = ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5', 'x6', 'y6', 'x7', 'y7', 'x8',
           'y8', 'x9', 'y9', 'x10', 'y10', 'x11', 'y11', 'x12', 'y12', 'x13', 'y13', 'x14', 'y14', 'x15', 'y15', 'x16',
           'y16', 'x17', 'y17', 'x18', 'y18', 'x19', 'y19', 'x20', 'y20']

df = pd.DataFrame(columns=columns)

number = 0
while number < 1500:
    number += 1
    print(number)
    ret, img = capture.read()

    # display img in every frame
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        img_height = img.shape[0]
        img_weight = img.shape[1]

        if result.multi_hand_landmarks:
            for handLmk in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, handLmk, mp_hand.HAND_CONNECTIONS)
                # print 21 points coordinates
                array = []
                for i, lm in enumerate(handLmk.landmark):
                    yPos = int(lm.y * img_height)
                    xPos = int(lm.x * img_weight)
                    array.append(xPos)
                    array.append(yPos)

                df.loc[-1] = array
                df.index = df.index + 1

        cv2.imshow('img', cv2.flip(img, -1))

        if cv2.waitKey(1) == ord('q'):
            break


# 0 stop
# 1 move
# 2 left
# 3 right
def creat_dataset(label, csv_name, df):
    length = len(df)
    if label=='stop':
        labels = [0] * length
        df['label'] = labels
    elif label == 'move':
        labels = [1] * length
        df['label'] = labels
    elif label == 'left':
        labels = [2] * length
        df['label'] = labels
    elif label == 'right':
        labels = [3] * length
        df['label'] = labels

    df.to_csv(str(csv_name), index = False)

    return

creat_dataset('left', 'left1.csv', df)