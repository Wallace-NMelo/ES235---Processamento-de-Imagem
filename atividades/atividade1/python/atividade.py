import cv2  # OpenCV
import matplotlib.pyplot as plt  # Matplotlib

# Getting Images
R_GREY = cv2.imread('R.png', cv2.IMREAD_GRAYSCALE)
G_GREY = cv2.imread('G.png', cv2.IMREAD_GRAYSCALE)
B_GREY = cv2.imread('B.png', cv2.IMREAD_GRAYSCALE)
RGB_original = cv2.imread('RGB.png', cv2.IMREAD_COLOR)

RGB_merged = cv2.merge((R_GREY, G_GREY, B_GREY))

assert (RGB_original == RGB_merged).all(), 'The Images are not the same'  # Cheking if images are the same
print("The Images are the same")
cv2.imwrite("RGB_answer.png", RGB_merged)  # Saving Image

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(RGB_original, cv2.COLOR_BGR2RGB))
plt.title("RGB Original")
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(RGB_merged, cv2.COLOR_BGR2RGB))
plt.title("RGB Merged")
plt.show()
