import numpy as np
import cv2

def accept_check(img):
    if np.sum(img[:,0] > 0) > 0:
        return 0
    if np.sum(img[:,img.shape[0]-1] > 0) > 0:
        return 0
    if np.sum(img[0,:] > 0) > 0:
        return 0
    if np.sum(img[img.shape[1]-1,:] > 0) > 0:
        return 0
    return 1

def rot_shift_image(image, angle, shift):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[:,2] += shift
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def load_item(name='fork'):
    return np.load(f'assets/{name}.npy')

# def rotate_image(image, angle):
#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#     return result

from matplotlib.patches import ArrowStyle
arrow_style = ArrowStyle("Simple", head_length=.6, head_width=1.0, tail_width=.1)

def show_grasp(ax, x, angle, label=None):
    x2_coef = np.sin(angle)
    x1_coef = np.cos(angle)
    x1 = x + x1_coef*50#+50#*50
    x2 = 224/2 + x2_coef*50# +50#*50
        
    color = 'r'
    if label is not None:
        if label == 1:
            color = 'b'
        
    ax.annotate("", xy=(x+x1_coef*20, 224/2+x2_coef*20), xytext=(x1, x2),arrowprops=dict(arrowstyle=arrow_style, color=color, linewidth=2))
    
    x1 = x - x1_coef*50
    x2 = 224/2 - x2_coef*50
    ax.annotate("", xy=(x-x1_coef*20, 224/2-x2_coef*20), xytext=(x1, x2),arrowprops=dict(arrowstyle=arrow_style, color=color, linewidth=2))
    
    ax.scatter(x, 224/2, s=40, color='g')

def count_real_positives(refined_points):
    # refined_points[:,0] *= 224
    # refined_points[:,1] *= (2*3.14)

    xs = np.asarray(refined_points[:,0])
    # alphas = np.asarray(refined_points[:,1])*(180/np.pi)
    alphas = np.asarray(refined_points[:,1])
    x_mask = np.logical_or(np.logical_and(xs >= 80, xs <= 120), np.logical_and(xs >= 160, xs <= 170))
    a_mask = np.logical_or(np.logical_and(alphas >= 80, alphas <= 100), np.logical_and(alphas >= 260, alphas <= 280))
    mask = np.logical_and(x_mask, a_mask)
    return np.sum(mask), mask