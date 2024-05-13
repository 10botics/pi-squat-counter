#!/usr/bin/env python
# coding: utf-8

# # Posture detection using tensorflow判斷人體姿態<br>Step 6

# Download `step5.py`, move it to the current folder, and import it.<br>
# 下載`step5.py`,將它移動到當前的文件夾下並導入

# In[1]:


from step5 import *
from step5 import _keypoints_and_edges_for_display

if __name__ != "__main__":
    print("====== Importing from step 6 ... ======")


# ### 3. Squat detection and counting 深蹲判斷與計數

# In squat exercises, though **knee and hip joint angles** vary by individual ability, flexibility, and squatting techniques, these can be used as effective features to determine squatting down or standing up.<br>深蹲運動中，雖然膝蓋與臀部關節角度可能因個人體能、柔軟度和深蹲技巧而有所不同，但一般來說，**膝蓋與臀部關節的角度**可以作爲我們判別蹲下或起立的有效特徵。<br>

# Above, we performed pose estimation on an image using MoveNet and defineed `get_keypoints_with_scores_from_image_with_movenet` to directly obtain MoveNet results from an image.<br>上面，我們可以通過MoveNet對圖像進行姿態辨識，並定義了`get_keypoints_with_scores_from_image_with_movenet`來直接從一張圖片獲取MoveNet結果<br>
# <br>
# How can we convert these results into angle?<br>
# 我們如何把這些結果轉換成角度信息呢？<br>
# ![img6-1](resource/img6-1.png)<br>

# ##### The `calculate_angle(a, b, c)` in the cell below obtains angle information using a such method.<br>下方單元格的`calculate_angle(a,b,c)`函數就是運用類似這樣的方法獲取角度信息。

# In[3]:


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle


# ##### Let's use the `calculate_angle(a,b,c)` previously defined to create a function that calculates the angle between the knee and hip joints.<br>讓我們利用上方定義的calculate_angle(a,b,c)封裝計算膝關節與臀部角度的函數
# The input of the function is the keypoints detected by the MoveNet model with scores.<br>
# 函數的輸入是MoveNet模型檢測到的帶有分數的關鍵點。<br>
# <br>
# `calculate_hip_joint_angle` calculates the angle of the left and right hip joints. It extracts positions of the hips, knees, and shoulders from keypoints, and calculates angles from the position. It finally returns the average of these two angles.<br>
# `calculate_hip_joint_angle`計算左右臀部的角度。它首先從關鍵點中提取左右臀部，膝蓋和肩膀的位置，然後從位置信息計算角度。最後返回這兩個角度的平均值。<br>
# <br>
# `calculate_hip_joint_angle` calculates the angle of the left and right knee joints in the same way<br>
# `calculate_hip_joint_angle` 函數用同樣的方法計算左右膝關節的角度。<br>

# In[2]:


def calculate_hip_joint_angle(keypoints_with_scores, keypoint_threshold = 0.2): 
    keypoints_with_scores = keypoints_with_scores[0][0]
    
    # get the position of keypoints from MoveNet output
    a1y, a1x, a1s = keypoints_with_scores[KEYPOINT_DICT["left_shoulder"]]
    a2y, a2x, a2s = keypoints_with_scores[KEYPOINT_DICT["right_shoulder"]]
    b1y, b1x, b1s = keypoints_with_scores[KEYPOINT_DICT["left_hip"]]
    b2y, b2x, b2s = keypoints_with_scores[KEYPOINT_DICT["right_hip"]]
    c1y, c1x, c1s = keypoints_with_scores[KEYPOINT_DICT["left_knee"]]
    c2y, c2x, c2s = keypoints_with_scores[KEYPOINT_DICT["right_knee"]]

    # calculate angle of left and right body respectively
    angle1 = calculate_angle( (a1y, a1x), (b1y, b1x), (c1y, c1x) )
    angle2 = calculate_angle( (a2y, a2x), (b2y, b2x), (c2y, c2x) )

    # return the midpoint of two angle
    return (angle1 + angle2) / 2
    

def calculate_knee_joint_angle(keypoints_with_scores, keypoint_threshold = 0.2): 
    keypoints_with_scores = keypoints_with_scores[0][0]

    # get the position of keypoints from MoveNet output
    a1y, a1x, a1s = keypoints_with_scores[KEYPOINT_DICT["left_hip"]]
    a2y, a2x, a2s = keypoints_with_scores[KEYPOINT_DICT["right_hip"]]
    b1y, b1x, b1s = keypoints_with_scores[KEYPOINT_DICT["left_knee"]]
    b2y, b2x, b2s = keypoints_with_scores[KEYPOINT_DICT["right_knee"]]
    c1y, c1x, c1s = keypoints_with_scores[KEYPOINT_DICT["left_ankle"]]
    c2y, c2x, c2s = keypoints_with_scores[KEYPOINT_DICT["right_ankle"]]

    # calculate angle of left and right body respectively
    angle1 = calculate_angle( (a1y, a1x), (b1y, b1x), (c1y, c1x) )
    angle2 = calculate_angle( (a2y, a2x), (b2y, b2x), (c2y, c2x) )

    # return the midpoint of two angle
    return (angle1 + angle2) / 2


# ##### Next, we will use OpenCV and functions above to detect body posture and calculate the number of squats.<br>接下來，我們利用OpenCV和上面的函數，檢測人的身體姿勢計算深蹲次數。
# <br>
# In each iteration, we pass the image frame to MoveNet and obtain the results, and then convert the results of MoveNet into angles of the knee joint and hip.<br>
# 在每次循環中，我們將圖像幀傳給MoveNet並獲取姿態檢測的結果，將MoveNet的結果轉換爲膝關節和臀部的角度。<br>
# <br>
# When the angle is below a certain value, the `squat` is recorded as 1 (indicates squatted down). Otherwise, otherwise the `squat` is recorded as 0 (indicates standed up). We add 1 to the squat count when transitioning from squat to stand. (i.e., when squat==1 and angle is greater than the threshold ).<br>
# 當角度小於一定值時，將`squat`記爲1（表示已蹲下）。否則，將`squat`記爲0（表示已起立）。我們將姿態從蹲下轉爲起立的瞬間（squat==1的同時角度大於判定值）將深蹲計數+1。

# In[6]:


def putText(frame, text, color = (0, 255, 0)):
    # Define the text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (50, 50)
    text_scale = 0.65
    text_color = color
    text_thickness = 2

    # Add text annotation on the frame
    cv2.putText(frame, text, text_position, font, text_scale, text_color, text_thickness)

repetition = 0
squat = 0
def frame_process(frame):
    global repetition, squat
    # calculate and display angle
    keypoints_with_scores = get_keypoints_with_scores_from_image_with_movenet(frame,input_size,input_size)
    hip_angle = calculate_hip_joint_angle(keypoints_with_scores)
    knee_angle = calculate_knee_joint_angle(keypoints_with_scores)


    # Gesture judgement from angle of hip-joint and knee-joint
    # if gesture is from squat (squat == 1) change to stand (squat == 0), repetition add by 1
    if hip_angle<135 and knee_angle<120:
        squat = 1
    elif hip_angle>135 and knee_angle>120:
        if squat == 1:
            repetition += 1
        squat = 0
    text = f"Reps: {repetition}  Knee: {int(knee_angle)}  Hip: {int(hip_angle)}"
    if squat:
        text_color = (0,255,0)
    else:
        text_color = (128, 192, 64)

    # draw visualized prediction from MoveNet on the frame. Attention: Not recommended, VERY LAGGY!
    #frame = draw_prediction_on_image( tf_image_to_model(frame,600,600)[0]/255, keypoints_with_scores)
    
    
    putText(frame, text, text_color)

    return frame


# Run
# ================
if __name__ == "__main__":
    display(stopButton)
    thread = threading.Thread(target=view, args=(stopButton, "Produce_2.mp4", 0, frame_process))
    thread.start()



# ##### <br><hr>Evidently, the output results of MoveNet are mostly reliable, but sometimes there may be some jitter, which can cause counting errors! <br>很明顯，MoveNet的輸出結果雖然大多是可靠的，但有時會有所抖動，而這些抖動會造成錯誤計數！<br>
# What are the possible solutions?<br>
# 有哪些解決方案呢？<br>

# 1. Output prone to shaking in **half-squat position**?<br>**半蹲狀態**的輸出容易抖動？<br><br>
# Separate the detection threshold for standing and squatting by a certain distance. Specifically, we set two detection thresholds separated by a certain distance. <br>
# Only when the angle is lower than the lower threshold, the status is set as "squatting"; only when the angle is higher than the higher threshold, the status is set as "standing". <br>
# This design can effectively filter out the result fluctuations caused by small-range fluctuations in the measurement value during the binarization process.<br>
# 將站立和蹲下的判斷閾值分開一定的距離。具體而言，我們設置兩個判斷閾值，分開一定的距離。<br>
# 當角度低於較低的判斷閾值，才把狀態設置爲“蹲下”；角度高於較高的判斷閾值，才把狀態設置爲“起立”。<br>
# 這樣的設計可以有效過濾二值化過程中，測量值在小範圍的波動引起的結果波動。<br>
# ![img6-3.png](img6-3.png) <br>
# <br>
# ```python
# if hip_angle<125 and knee_angle<105:
#     squat = 1
# elif hip_angle>145 and knee_angle>125:
#     if squat == 1:
#         repetition += 1
#     squat = 0
# ```
# 2. **Meaningless title** also determined the squatting status?<br>**毫無意義的片頭**也被用來判斷深蹲狀態？<br>
# Do not let the squat counter count when MoveNet is not confident with the result.<br>
# 在MoveNet對結果不確定的時候，不要讓squat counter計數<br>
# <br>
# ```python
# def .......(....):
#     ......
#     if (a1s>keypoint_threshold)*(b1s>keypoint_threshold)*(c1s>keypoint_threshold):
#         return (angle1 + angle2) / 2
#     else:
#         return None
# ```
# ```python
# if (hip_angle is not None) and (knee_angle is not None):
#     .........
#     text = f"Reps: {repetition}  Knee: {int(knee_angle)}  Hip: {int(hip_angle)}"
# else:
#     text = f"Reps: {repetition}  Knee: ?  Hip: ?"
# ```
# 3. **Other** possible sources that trigger jittering?<br>**其他**可能觸發抖動的來源？<br>
# Collect multiple image samples and only update the squatting status when several consecutive results indicate squatting (standing).<br>
# Avoid random interference (such as background, casual movements) through multiple "measurements".<br>
# 採集多個圖像樣本，只有連續幾張照片的結果都爲蹲下或站立才更新squat的狀態。<br>
# 藉由多次“測量”避免偶然干擾（如背景、偶然動作）的影響。<br>
# <br>
# ```python
# squatting = 0
# CONSECUTIVE_THRESHOLD = 3    # << You can change this!
# # squatting == 0: last consecutive 3 confident results from images are all standing
# # squatting == CONSECUTIVE_THRESHOLD: last consecutive 3 confident results from images are all squatting
# def frame_process(frame):
#     global repetition, squat, squatting
#     ..........
#     if hip_angle<125 and knee_angle<105:
#         # if MoveNet result is squatting
#         #   if current state is standing, squatting += 1
#         #   otherwise, reset squatting to CONSECUTIVE_TH
#         if squat == 0:
#             squatting += 1
#         else:
#             squatting = CONSECUTIVE_TH
#     elif hip_angle>145 and knee_angle>125:
#         # if MoveNet result is standing
#         #   if current state is squatting, squatting -= 1
#         #   otherwise, reset squatting to 0
#         if squat == 1:
#             squatting -= 1
#         else:
#             squatting = 0
#     if squatting == 0 and squat == 1:
#         repetition += 1
#         squat = 0
#     elif squatting == CONSECUTIVE_TH and squat == 0:
#         squat = 1
#     ...........
# ```
# 

# ### Final Version 最終版本

# In[2]:


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def calculate_hip_joint_angle(keypoints_with_scores, keypoint_threshold = 0.2): 
    keypoints_with_scores = keypoints_with_scores[0][0]
    
    # get the position of keypoints from MoveNet output
    a1y, a1x, a1s = keypoints_with_scores[KEYPOINT_DICT["left_shoulder"]]
    a2y, a2x, a2s = keypoints_with_scores[KEYPOINT_DICT["right_shoulder"]]
    b1y, b1x, b1s = keypoints_with_scores[KEYPOINT_DICT["left_hip"]]
    b2y, b2x, b2s = keypoints_with_scores[KEYPOINT_DICT["right_hip"]]
    c1y, c1x, c1s = keypoints_with_scores[KEYPOINT_DICT["left_knee"]]
    c2y, c2x, c2s = keypoints_with_scores[KEYPOINT_DICT["right_knee"]]

    # calculate angle of left and right body respectively
    angle1 = calculate_angle( (a1y, a1x), (b1y, b1x), (c1y, c1x) )
    angle2 = calculate_angle( (a2y, a2x), (b2y, b2x), (c2y, c2x) )

    # if confident score of keypoints are all above threshold, return the midpoint of two angle
    # otherwise, return None
    if (a1s>keypoint_threshold)*(b1s>keypoint_threshold)*(c1s>keypoint_threshold):
        return (angle1 + angle2) / 2
    else:
        return None

def calculate_knee_joint_angle(keypoints_with_scores, keypoint_threshold = 0.2): 
    keypoints_with_scores = keypoints_with_scores[0][0]

    # get the position of keypoints from MoveNet output
    a1y, a1x, a1s = keypoints_with_scores[KEYPOINT_DICT["left_hip"]]
    a2y, a2x, a2s = keypoints_with_scores[KEYPOINT_DICT["right_hip"]]
    b1y, b1x, b1s = keypoints_with_scores[KEYPOINT_DICT["left_knee"]]
    b2y, b2x, b2s = keypoints_with_scores[KEYPOINT_DICT["right_knee"]]
    c1y, c1x, c1s = keypoints_with_scores[KEYPOINT_DICT["left_ankle"]]
    c2y, c2x, c2s = keypoints_with_scores[KEYPOINT_DICT["right_ankle"]]

    # calculate angle of left and right body respectively
    angle1 = calculate_angle( (a1y, a1x), (b1y, b1x), (c1y, c1x) )
    angle2 = calculate_angle( (a2y, a2x), (b2y, b2x), (c2y, c2x) )

    # if confident score of keypoints are all above threshold, return the midpoint of two angle
    # otherwise, return None
    if (a1s>keypoint_threshold)*(b1s>keypoint_threshold)*(c1s>keypoint_threshold):
        return (angle1 + angle2) / 2
    else:
        return None


# In[5]:


def putText(frame, text, color = (0, 255, 0)):
    # Define the text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (50, 50)
    text_scale = 0.65
    text_color = color
    text_thickness = 2

    # Add text annotation on the frame
    cv2.putText(frame, text, text_position, font, text_scale, text_color, text_thickness)

repetition = 0
squat = 0
squatting = 0
CONSECUTIVE_TH = 3    # << You can change this!
def frame_process(frame):
    global repetition, squat, squatting
    # calculate and display angle
    keypoints_with_scores = get_keypoints_with_scores_from_image_with_movenet(frame,input_size,input_size)
    hip_angle = calculate_hip_joint_angle(keypoints_with_scores)
    knee_angle = calculate_knee_joint_angle(keypoints_with_scores)


    # check if confidence is qualified, if yes, proceed to judgement
    if hip_angle is not None and knee_angle is not None:
        # if capture continuous CONSECUTIVE_TH angles in squatting, change squat to 1
        # if capture continuous CONSECUTIVE_TH angles in standing, change squat to 0 and repetition += 1
        if hip_angle<125 and knee_angle<105:
            if squat == 0:
                squatting += 1
            else:
                squatting = CONSECUTIVE_TH
        elif hip_angle>145 and knee_angle>125:
            if squat == 1:
                squatting -= 1
            else:
                squatting = 0
        if squatting == 0 and squat == 1:
            repetition += 1
            squat = 0
        elif squatting == CONSECUTIVE_TH and squat == 0:
            squat = 1
        text = f"Reps: {repetition}  Knee: {int(knee_angle)}  Hip: {int(hip_angle)}"
    else:
        text = f"Reps: {repetition}  Knee: ?  Hip: ?"
    if squat:
        text_color = (0,255,0)
    else:
        text_color = (128,192,64)

    # draw visualized prediction from MoveNet on the frame. Attention: Not recommended, VERY LAGGY!
    #frame = draw_prediction_on_image( tf_image_to_model(frame,600,600)[0]/255, keypoints_with_scores)
    
    
    putText(frame, text, text_color)

    return frame


# Run
# ================
if __name__ == "__main__":
    display(stopButton)
    thread = threading.Thread(target=view, args=(stopButton, 0, 0, frame_process))
    thread.start()


