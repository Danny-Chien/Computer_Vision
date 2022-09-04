from locale import normalize
from PIL import Image
import numpy as np

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    reshape_size = 16
    # tiny_images = np.zeros(shape= (len(image_paths), reshape_size ** 2))
    tiny_images = []
    # print(tiny_images.shape)
    # tiny_images = []
    i = 0
    for path in image_paths:
        img = Image.open(path)        
        # width, height = img.size
        # left = width / 4
        # top = height / 4  
        # right = 3 * width / 4
        # bottom = 3 * height / 4
        
        # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # img = img.crop((left, top, right, bottom))
        # img = img.rotate(90, Image.NEAREST)
        # img = img.resize([int(width / 10), int(height / 10)])
        img = img.resize([reshape_size, reshape_size])

        tmp = np.array(img)
        # mean = np.mean(tmp)
        # std = np.std(tmp)
        # tmp = (tmp - mean) / std
        # tmp = tmp / 255
        tmp = tmp.reshape(-1)
        mean = np.mean(tmp)
        tmp = tmp - mean
        norm = np.linalg.norm(tmp, ord= None)
        tmp = tmp / norm
        tiny_images.append(tmp)
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
