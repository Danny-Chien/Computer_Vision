import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # If crossCheck==true, then the knnMatch() method with k=1 
    w = 0
    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]
        w += im1.shape[1]
        # TODO: 1.feature detection & matching
        keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)        
        
        goodu = []
        goodv = []

        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                goodu.append(keypoints1[m.queryIdx].pt)
                goodv.append(keypoints2[m.trainIdx].pt)
        goodu = np.array(goodu)
        goodv = np.array(goodv)
        # print(f'the shape of goodu is {goodu.shape} and the shape of goodv is {goodv.shape}')
        # TODO: 2. apply RANSAC to choose best H
        # ref : https://www.csdn.net/tags/MtjaMg3sNDAwNS1ibG9n.html
        threshold = 4
        num_iteration = 5
        randu = np.zeros(shape=(4,2))
        randv = np.zeros(shape=(4,2))
        H_best = np.eye(3)
        inline_max = 0
        i = 0
        # for i in range(5001):
        while(i < num_iteration):
            for j in range(4):
                id = random.randint(0, len(goodu)-1)
                randu[j] = goodu[id]
                randv[j] = goodv[id]
            
            H = solve_homography(randv, randu)

            M = np.concatenate((goodv.T, np.ones(shape=(1, len(goodv)))), axis= 0)
            W = np.concatenate((goodu.T, np.ones(shape=(1, len(goodu)))), axis= 0)
            matrix = np.dot(H, M)
            matrix = np.divide(matrix, matrix[-1,:])

            err = np.linalg.norm(x=(matrix - W)[:-1, :], ord=1, axis=0)
            inlineN = sum(err<threshold)
            inline_u = goodu[err<threshold]
            inline_v = goodv[err<threshold]

            if (inlineN > inline_max):
                inline_max = inlineN
                H_best = H
                i += 1
        # TODO: 3. chain the homographies
        # TODO: 4. apply warping
        last_best_H = np.dot(last_best_H, H_best)
        out = warping(im2, dst, last_best_H, ymin=0, ymax=im1.shape[0], xmin=w, xmax= (w + im2.shape[1]), direction= 'b')
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)