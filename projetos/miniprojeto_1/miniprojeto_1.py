import numpy as np
import cv2
import time

cap_jon = cv2.VideoCapture("jon_snow.mp4")
video_cap = cv2.VideoCapture("video2.mp4")

video_width = int(video_cap.get(3))  # Width
video_height = int(video_cap.get(4))  # Height

input_width = cap_jon.get(3)  # Width
input_height = cap_jon.get(4)  # Height
video_fps = cap_jon.get(5)  # FPS
pts2 = np.float32([[0, 0], [input_width, 0], [0, input_height], [input_width, input_height]])

# Colors Threshold
# Red color
low_red = np.array([0, 0, 115])
high_red = np.array([90, 90, 255])

# Blue color
low_blue = np.array([94, 0, 0])
high_blue = np.array([255, 90, 55])

# Purple color
low_purple = np.array([47, 15, 48])
high_purple = np.array([80, 45, 70])

# Green color
low_green = np.array([0, 100, 0])
high_green = np.array([40, 255, 50])

while video_cap.isOpened() and cap_jon.isOpened():

    ret, frame = video_cap.read()
    frame_cp = frame
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret1, jon_frame = cap_jon.read()
    blur = cv2.GaussianBlur(frame, (15, 15), 0)
    img_bin = cv2.threshold(frame_grey, 85, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


    # Mask Blue
    mask_blue = cv2.inRange(blur, low_blue, high_blue)
    mask_blue = cv2.GaussianBlur(mask_blue, (15, 15), 0)
    contours_blue = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cv2.drawContours(frame, contours_blue, -1, (255, 0, 0), 3)
    if contours_blue:
        cmax_blue = max(contours_blue, key=cv2.contourArea)
        cv2.drawContours(frame, cmax_blue, -1, (255, 0, 0), 1)
        Mb = cv2.moments(cmax_blue)
        cb_X, cb_Y = int(Mb["m10"] / Mb["m00"]), int(Mb["m01"] / Mb["m00"])

    # Mask Red
    mask_red = cv2.inRange(blur, low_red, high_red)
    mask_red = cv2.GaussianBlur(mask_red, (11, 11), 0)
    contours_red = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours_red:
        cmax_red = max(contours_red, key=cv2.contourArea)
        cv2.drawContours(frame, cmax_red, -1, (0, 0, 255), 1)
        Mr = cv2.moments(cmax_red)
        cr_X, cr_Y = int(Mr["m10"] / Mr["m00"]), int(Mr["m01"] / Mr["m00"])

    # Mask Green
    mask_green = cv2.inRange(blur, low_green, high_green)
    mask_green = cv2.GaussianBlur(mask_green, (11, 11), 0)
    contours_green = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours_green:
        cmax_green = max(contours_green, key=cv2.contourArea)
        cv2.drawContours(frame, cmax_green, -1, (0, 255, 0), 1)
        Mg = cv2.moments(cmax_green)
        cg_X, cg_Y = int(Mg["m10"] / Mg["m00"]), int(Mg["m01"] / Mg["m00"])

    # Mask Purple
    mask_purple = cv2.inRange(blur, low_purple, high_purple)
    # Subtracting the primary colors to avoid target confusion
    mask_purple = cv2.subtract(mask_purple, mask_red)
    mask_purple = cv2.subtract(mask_purple, mask_blue)
    mask_purple = cv2.subtract(mask_purple, mask_green)
    # Subtract outside the frame
    mask_purple = cv2.subtract(mask_purple, img_bin)
    mask_purple = cv2.GaussianBlur(mask_purple, (17, 17), 0)
    contours_purple = cv2.findContours(mask_purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if contours_purple:
        cmax_purple = max(contours_purple, key=cv2.contourArea)
        cv2.drawContours(frame, cmax_purple, -1, (128, 0, 128), 1)
        Mp = cv2.moments(cmax_purple)
        cp_X, cp_Y = int(Mp["m10"] / Mp["m00"]), int(Mp["m01"] / Mp["m00"])
    else:
        cp_X, cp_Y = np.nan, np.nan

    # Centers for image localization
    pts1 = np.float32([[cr_X, cr_Y], [cg_X, cg_Y], [cb_X, cb_Y], [cp_X, cp_Y]])

    # Inserting Image
    if pts1.any() is not np.nan:
        M = cv2.getPerspectiveTransform(pts2, pts1)
        cv2.warpPerspective(jon_frame, M, (frame.shape[1], frame.shape[0]), frame,
                            borderMode=cv2.BORDER_TRANSPARENT)

    """
    # Visualization of Masks
    mask_red = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
    mask_blue = cv2.cvtColor(mask_blue, cv2.COLOR_GRAY2BGR)
    mask_green = cv2.cvtColor(mask_green, cv2.COLOR_GRAY2BGR)
    mask_purple = cv2.cvtColor(mask_purple, cv2.COLOR_GRAY2BGR)
    frame_grey = cv2.cvtColor(frame_grey, cv2.COLOR_GRAY2BGR)
    img_binary = cv2.cvtColor(img_binary, cv2.COLOR_GRAY2BGR)
    vis1 = np.hstack((frame, cv2.resize(jon_frame, (video_width, video_height))))
    """
    vis1 = np.hstack((frame, video_cap.read()[1]))
    img_bin = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)


    cv2.imshow("output", vis1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(1 / video_fps)
video_cap.release()
