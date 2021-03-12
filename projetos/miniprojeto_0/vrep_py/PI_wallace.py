import time

import cv2
import numpy as np

import vrep

print('program started')
vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
print('Connected to remote API server')
r, colorCam = vrep.simxGetObjectHandle(clientID, "kinect_rgb", vrep.simx_opmode_oneshot_wait);
r, leftmotor = vrep.simxGetObjectHandle(clientID, "Pioneer_p3dx_leftMotor", vrep.simx_opmode_oneshot_wait);
r, rightmotor = vrep.simxGetObjectHandle(clientID, "Pioneer_p3dx_rightMotor", vrep.simx_opmode_oneshot_wait);

vrep.simxSetJointTargetVelocity(clientID, leftmotor, 0, vrep.simx_opmode_streaming);
vrep.simxSetJointTargetVelocity(clientID, rightmotor, 0, vrep.simx_opmode_streaming);

r, resolution, image = vrep.simxGetVisionSensorImage(clientID, colorCam, 1, vrep.simx_opmode_streaming);
time.sleep(0.5)

l_speed = 1
r_speed = -1
cent_point = 240
dim_y, dim_x = 480, 640


# Encontra centroide de faixa
def centroide_posicao(mask, tira):
    r = np.mean(np.where(mask[tira, :] <= 50))
    if np.isnan(r):
        out_line = True
    else:
        out_line = False
    return r, out_line


while True:
    r, resolution, image = vrep.simxGetVisionSensorImage(clientID, colorCam, 1, vrep.simx_opmode_buffer);
    mat = np.asarray(image, dtype=np.uint8)
    mat2 = mat.reshape(resolution[1], resolution[0], 1)
    # Binarização da imagem e eliminação de ruidos
    cam_img = cv2.flip(mat2, 0)
    r, cam_bin = cv2.threshold(cam_img, 100, 255, cv2.THRESH_BINARY)
    cam_bin = cv2.medianBlur(cam_bin, 5)
    # centroide da imagem e posição do bot na linha
    cent_point, out_line = centroide_posicao(cam_bin, dim_y // 2)
    bot_point, bot_out = centroide_posicao(cam_bin, dim_y - 1)

    # Fora da linha central
    if out_line and last_cent <= 320:
        l_speed = -2
        r_speed = 2
    elif out_line and last_cent > 320:
        l_speed = 2
        r_speed = -2

    # Na linha central
    else:
        last_cent = cent_point
        if cent_point == dim_x / 2:
            l_speed = 10
            r_speed = 10
        else:

            l_speed = cent_point * 8 / dim_x
            r_speed = (640 - cent_point) * 8 / dim_x

    print("cent_point = {}".format(cent_point))
    print("vl = {}, vr = {}".format(l_speed, r_speed))
    vrep.simxSetJointTargetVelocity(clientID, leftmotor, l_speed, vrep.simx_opmode_streaming)
    vrep.simxSetJointTargetVelocity(clientID, rightmotor, r_speed, vrep.simx_opmode_streaming)

    # Representação na imagem
    if not out_line and not bot_out:
        cv2.ellipse(cam_img, center=(int(cent_point), dim_y // 2), axes=(20, 20), angle=0, startAngle=0, endAngle=360,
                    color=(255, 0, 255), thickness=-1)
        cv2.line(cam_img, (int(bot_point), dim_y - 1), (int(cent_point), dim_y // 2), (0, 0, 255), thickness=2,
                 lineType=-1)
        cv2.ellipse(cam_img, center=(int(bot_point), dim_y - 1), axes=(20, 20), angle=0, startAngle=0, endAngle=360,
                    color=111, thickness=-1)

    cv2.imshow('robot camera', cam_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.waitKey(1)
