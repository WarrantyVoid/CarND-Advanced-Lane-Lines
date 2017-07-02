
import cv2
import numpy as np
import skimage.transform as imgtf
import matplotlib.pyplot as plt
import skimage.exposure as imgexp
# Change to appropriate color map for gray scale images
plt.rcParams['image.cmap'] = 'gray'


############ Functions ############


# Normalizes image data to the range 0..255
def normalize(image):
    img_min = np.min(image)
    img_max = np.max(image)
    return ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)


#  Symmetric edge threshold
def symmetric_sobel_thresh(img, thresh_min=-25, thresh_max=25, translate=7, orient='x', sobel_kernel=7):
    if orient == 'x':
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    min_sobel = np.min(sobel)
    max_sobel = np.max(sobel)
    scaled_sobel = sobel / max(abs(min_sobel), abs(max_sobel)) * 255

    binary_output_neg = np.zeros_like(img)
    binary_output_neg[(scaled_sobel < thresh_min)] = 1

    trans = imgtf.AffineTransform(translation=(translate, 0))
    binary_output_neg = imgtf.warp(binary_output_neg, trans)

    binary_output_pos = np.zeros_like(img)
    binary_output_pos[(scaled_sobel > thresh_max)] = 1

    trans = imgtf.AffineTransform(translation=(-translate, 0))
    binary_output_pos = imgtf.warp(binary_output_pos, trans)

    binary_output = np.zeros_like(img)
    binary_output[(binary_output_neg > 0) & (binary_output_pos > 0)] = 1
    return binary_output


# Adaptive threshold based on mean value difference inside grid cells
def adaptive_threshold(img, grid=(1.0, 0.1), p_threshold=0.25):
    binary_output = np.zeros_like(img)
    step_y = int(np.floor(img.shape[0] * grid[1]))
    step_x = int(np.floor(img.shape[1] * grid[0]))
    for y in range(0, img.shape[0], step_y):
        for x in range(0, img.shape[1], step_x):
            img_grid = img[y:y + step_y, x:x + step_x]
            img_mean = np.mean(img_grid)
            img_max = np.max(img_grid)
            threshold_grid = np.round(img_mean + (img_max - img_mean) * p_threshold)
            binary_output[y:y + step_y, x:x + step_x] = (img_grid > threshold_grid)
    return binary_output


# Displays channel matrix with different color spaces
def color_space_map(img):
    rgb = img
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Plotting thresholded images
    f, ax = plt.subplots(5, 4, figsize=(15, 10), frameon=False)
    f.subplots_adjust(hspace=0.15, wspace=0.00, left=0, bottom=0, right=1, top=0.97)

    ax[0,0].set_title('RGB')
    ax[0,0].axis('off')
    ax[0,0].imshow(img)

    for i in range (1, 4):
        ax[0, i].axis('off')
        ax[0, i].imshow(img[:, :, i-1])

    ax[1, 0].set_title('HLS')
    ax[1, 0].axis('off')
    ax[1, 0].imshow(hls)

    for i in range (1, 4):
        ax[1, i].axis('off')
        ax[1, i].imshow(hls[:, :, i-1])

    ax[2, 0].set_title('HSV')
    ax[2, 0].axis('off')
    ax[2, 0].imshow(hsv)

    for i in range (1, 4):
        ax[2, i].axis('off')
        ax[2, i].imshow(hsv[:, :, i-1])

    ax[3, 0].set_title('YUV')
    ax[3, 0].axis('off')
    ax[3, 0].imshow(yuv)

    for i in range (1, 4):
       ax[3, i].axis('off')
       ax[3, i].imshow(yuv[:, :, i-1])

    ax[4, 0].set_title('LAB')
    ax[4, 0].axis('off')
    ax[4, 0].imshow(lab)

    for i in range(1, 4):
        ax[4, i].axis('off')
        ax[4, i].imshow(lab[:, :, i - 1])

    plt.show()


# Combines three thresholds for white, yellow and edges
def multi_threshold(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = (imgexp.equalize_adapthist(lab[:, :, 0]) * 255).astype(np.uint8)
    lab[:, :, 2] = (imgexp.equalize_adapthist(lab[:, :, 2]) * 255).astype(np.uint8)

    white = adaptive_threshold(lab[:, :, 0], (1.0, 0.2), 0.8)
    yellow = adaptive_threshold(lab[:, :, 2], (1.0, 0.2), 0.6)
    edges = symmetric_sobel_thresh(lab[:, :, 0], -1.0, 1.0, sobel_kernel=7)

    r = np.zeros_like(white)
    r[((white > 0) & (edges > 0)) | ((yellow > 0) & (edges > 0))] = 255

    '''
    img[:, :, 0] = r
    img[:, :, 1] = img[:, :, 1] / 2
    img[:, :, 2] = img[:, :, 2] / 2

    f, ax = plt.subplots(2, 2, figsize=(15, 10), frameon=False)
    f.subplots_adjust(hspace=0.0, wspace=0.0, left=0, bottom=0, right=1, top=1)
    ax[0, 0].imshow(img)
    ax[0, 0].axis('off')
    ax[0, 1].imshow(white)
    ax[0, 1].axis('off')
    ax[1, 0].imshow(yellow)
    ax[1, 0].axis('off')
    ax[1, 1].imshow(edges)
    ax[1, 1].axis('off')
    plt.show()
    '''

    return r
