from operator import index
from turtle import shape
from matplotlib.pyplot import axis
import numpy as np
import cv2
# from cv2 import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def findlocalmax(self,  dog_images, octave_idx):
        tmp = []
        w = dog_images[4*octave_idx].shape[0]
        h = dog_images[4*octave_idx].shape[1]
        for i in range(1, w-1):
            for j in range(1, h-1):
                img1 = dog_images[4*octave_idx][(i-1):(i+2), (j-1):(j+2)]
                img2 = dog_images[4*octave_idx + 1][(i-1):(i+2), (j-1):(j+2)]
                img3 = dog_images[4*octave_idx + 2][(i-1):(i+2), (j-1):(j+2)]
                img4 = dog_images[4*octave_idx + 3][(i-1):(i+2), (j-1):(j+2)]
                mat1 = np.array(img1)
                mat2 = np.array(img2)
                mat3 = np.array(img3)
                mat4 = np.array(img4)
                tmp1 = (np.array([mat1, mat2, mat3]))
                tmp2 = (np.array([mat2, mat3, mat4]))

                if(np.abs(tmp1[1, 1, 1]) > self.threshold):                               
                    if((np.max(tmp1) == tmp1[1, 1, 1]) | (np.min(tmp1) == tmp1[1, 1, 1])) & (octave_idx == 0): 
                        tmp.append([i, j])
                    elif((np.max(tmp1) == tmp1[1, 1, 1]) | (np.min(tmp1) == tmp1[1, 1, 1])) & (octave_idx == 1):
                        tmp.append([2*i, 2*j])
                if(np.abs(tmp2[1, 1, 1]) > self.threshold) :                  
                    if((np.max(tmp2) == tmp2[1, 1, 1]) | (np.min(tmp2) == tmp2[1, 1, 1])) & (octave_idx == 0): 
                        tmp.append([i, j])
                    elif((np.max(tmp2) == tmp2[1, 1, 1]) | (np.min(tmp2) == tmp2[1, 1, 1])) & (octave_idx == 1):
                        tmp.append([2*i, 2*j])  
        return tmp
    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = [] 

        for i in range(self.num_octaves):
            for j in range(self.num_guassian_images_per_octave):
                gaussian_images.append(image)
                image = cv2.GaussianBlur(gaussian_images[5*i] , ksize = (0, 0), sigmaX = self.sigma ** (j + 1))
            if i == 0:
                image = cv2.resize(gaussian_images[-1], dsize = (int(gaussian_images[-1].shape[1] / 2), int(gaussian_images[-1].shape[0] / 2)), interpolation = cv2.INTER_NEAREST )     
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        
        for i in range(self.num_octaves):
            for j in range(self.num_DoG_images_per_octave):
                if i == 0:
                    img = cv2.subtract(gaussian_images[j + 1], gaussian_images[j])
                    
                else:
                    img = cv2.subtract(gaussian_images[j + 6], gaussian_images[j + 5])
                
                # path = './image/' + str(i) + '-' + str(j) + '.png'
                # tmp = img / np.max(img) * 255
                # cv2.imwrite(path, tmp)
                dog_images.append(img) 
        
        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        keypoints += self.findlocalmax(dog_images = dog_images, octave_idx = 0)
        keypoints += self.findlocalmax(dog_images = dog_images, octave_idx = 1)

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis = 0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
