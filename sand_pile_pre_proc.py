import numpy as np
from skimage import color
from skimage import io
import matplotlib.pyplot as plt
from scipy import signal
import cv2


NOMINATOR = 80
MEAN_BUFFER = int(NOMINATOR/13)

def image_derivative(filename: str):
    # Todo: remove luminus
    image = io.imread(filename)
    im = color.rgb2lab(image)
    # im[:, :, 2] = 0
    im = color.lab2rgb(im)
    im_shape = im.shape
    denominator = int(im_shape[1] / NOMINATOR)
    im = cv2.resize(im, (
    int(im_shape[1] / denominator), int(im_shape[0] / denominator)),
                    interpolation=cv2.INTER_LINEAR)
    grad_im_R = cv2.Sobel(im[:, :, 0], cv2.CV_64F, 0, 1, ksize=5) / 255
    grad_im_G = cv2.Sobel(im[:, :, 1], cv2.CV_64F, 0, 1, ksize=5) / 255
    grad_im_B = cv2.Sobel(im[:, :, 2], cv2.CV_64F, 0, 1, ksize=5) / 255
    grad_im = np.sqrt(np.square(grad_im_B) +
                      np.square(grad_im_G) +
                      np.square(grad_im_R))
    grad_im = np.where(grad_im > 0.05, 1, 0)
    grad_im = cv2.blur(grad_im, (3, 3))
    grad_im = np.where(grad_im > (1 / 3), 1, 0)
    grad_im = cv2.blur(grad_im, (3, 3))
    grad_im = np.where(grad_im > (1 / 3), 1, 0)

    plt.imshow(im)
    plt.show()
    plt.imshow(grad_im, cmap='gray')
    plt.show()
    im_shape = im.shape
    mid = int(im_shape[1] / 2)

    lines = cv2.HoughLines((grad_im * 255).astype(np.uint8), 1, np.pi / 180,
                           25)
    # print(lines.shape)
    lines2 = lines.reshape((len(lines), -1))
    thetamax = np.argmax(lines2, 0)[1]
    thetamin = np.argmin(lines2, 0)[1]
    binary_im = np.ones((im_shape[0],im_shape[1]))
    for rho, theta in lines[thetamax]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(binary_im, (x1, y1), (x2, y2), 0, 1)

    for rho, theta in lines[thetamin]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(binary_im, (x1, y1), (x2, y2), 0, 1)

    plt.imshow(binary_im, cmap='gray')
    plt.show()
    height_array = np.zeros(2 * MEAN_BUFFER + 1)
    for i in range(2 * MEAN_BUFFER + 1):
        for j in range(im_shape[0]):
            if binary_im[im_shape[0] - j - 1, i + mid - MEAN_BUFFER] == 0:
                height_array[i] = j
                break
    print(np.percentile(height_array,0.3))

if __name__ == '__main__':
    for k in range(1, 5):
        image_derivative('sand_pile' + str(k) + '.jpg')
