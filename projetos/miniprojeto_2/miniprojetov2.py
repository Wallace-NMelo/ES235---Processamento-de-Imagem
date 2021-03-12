import cv2
import numpy as np

video_cap = cv2.VideoCapture("video.mp4")

freezeFrame = True
erase = False
mouse_event = False
point = (0, 0)

video_width = int(video_cap.get(3))  # Width
video_height = int(video_cap.get(4))  # Height
video_fps = video_cap.get(5)  # FPS
mask = np.zeros((video_height, video_width), dtype="uint8")

exhibition_mode = "Original/Inpainted"  # Flag to choose exhibition mode
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_inpait.avi', fourcc, 30.0, (video_width, video_height))


def mouse_click_event(event, x, y, flags, params):
    global erase, mouse_x, mouse_y, mouse_event

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        mouse_x, mouse_y = x, y
        mouse_event = True

    elif event == cv2.EVENT_MBUTTONDOWN:
        erase = not erase
        mouse_event = False

    else:
        mouse_event = False


cv2.namedWindow(winname='Freeze Frame')
cv2.setMouseCallback("Freeze Frame", mouse_click_event)
ret, frame = video_cap.read()
freeze_frame = frame.copy()

while video_cap.isOpened():

    while freezeFrame:
        cv2.destroyWindow(exhibition_mode)
        # Check for existing mask in files
        if cv2.imread("mask.jpg"):
            mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)
            freeze_frame = False
            cv2.destroyAllWindows()
            break
        else:
            # Create mask manually
            if mouse_event and not erase:
                cv2.circle(freeze_frame, (mouse_x, mouse_y), 6, (255, 255, 255), -1)
                cv2.circle(mask, (mouse_x, mouse_y), 6, 255, -1)

            elif mouse_event and erase:
                x_, y_ = mouse_x - 10, mouse_y - 10
                freeze_frame[y_:(y_ + 2 * 10), x_:(x_ + 2 * 10)] = frame.copy()[y_:(y_ + 2 * 10), x_:(x_ + 2 * 10)]
                cv2.circle(mask, (mouse_x, mouse_y), 12, 0, -1)

            if cv2.waitKey(1) & 0xFF == ord('m'):
                freezeFrame = False
                cv2.destroyAllWindows()
                print("Exhibition Modes Keys:\n")
                print("----- Original/Inpainted press b ----\n")
                print("----- Original press o -----\n")
                print("----- Inpainted press i -----\n")
                break

            mask_show = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

            frame_show = np.hstack((freeze_frame, mask_show))

            cv2.imshow("Freeze Frame", freeze_frame)

    # Key inputs
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        freezeFrame = True
        freeze_frame = frame.copy()

    if key == ord('o'):
        exhibition_mode = "Original"
        cv2.destroyAllWindows()
    if key == ord('i'):
        exhibition_mode = "Inpainted"
        cv2.destroyAllWindows()
    if key == ord('b'):
        exhibition_mode = "Original/Inpainted"
        cv2.destroyAllWindows()
    ret, frame = video_cap.read()
    frame_painted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

    if exhibition_mode == "Original/Inpainted":
        frame_show = np.hstack((frame, frame_painted))
    elif exhibition_mode == "Original":
        frame_show = frame
    elif exhibition_mode == "Inpainted":
        frame_show = frame_painted

    frame_show = cv2.resize(frame_show, (960, 500))
    cv2.imshow(exhibition_mode, frame_show)
    out.write(frame_painted)

video_cap.release()
out.release()
cv2.destroyAllWindows()
