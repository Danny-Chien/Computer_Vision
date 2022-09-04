import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def do_filter(self, img, padded_guidance, padded_img, Guassian_spatial, output, num_channel):
        for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    # do range kernel
                    if(num_channel == 3):
                        center_r = padded_guidance[0][(i + self.pad_w), (j + self.pad_w)]
                        center_g = padded_guidance[1][(i + self.pad_w), (j + self.pad_w)]
                        cenetr_b = padded_guidance[2][(i + self.pad_w), (j + self.pad_w)]
                        Tp_r = padded_guidance[0][i:(i + self.wndw_size), j:(j + self.wndw_size)]
                        Tp_r = Tp_r.reshape(1, -1)
                        Tp_g = padded_guidance[1][i:(i + self.wndw_size), j:(j + self.wndw_size)]
                        Tp_g = Tp_g.reshape(1, -1)
                        Tp_b = padded_guidance[2][i:(i + self.wndw_size), j:(j + self.wndw_size)]
                        Tp_b = Tp_b.reshape(1, -1)
                        cen_r = (center_r/255 - (Tp_r/255)) ** 2
                        cen_g = (center_g/255 - (Tp_g/255)) ** 2
                        cen_b = (cenetr_b/255 - (Tp_b/255)) ** 2
                        tmp = cen_r + cen_g + cen_b
                    elif(num_channel == 1):
                        center = padded_guidance[(i + self.pad_w), (j + self.pad_w)]                   
                        Tp = padded_guidance[i:(i + self.wndw_size), j:(j + self.wndw_size)]
                        Tp = Tp.reshape(1, -1)
                        tmp = (center/255 - Tp/255) ** 2
                    Gr = np.exp(-(tmp) / (2 * (self.sigma_r ** 2)))
                    # do convolution
                    Iq_r = padded_img[0][i:(i + self.wndw_size), j:(j + self.wndw_size)]
                    Iq_r = Iq_r.reshape(1, -1)
                    Iq_g = padded_img[1][i:(i + self.wndw_size), j:(j + self.wndw_size)]
                    Iq_g = Iq_g.reshape(1, -1)
                    Iq_b = padded_img[2][i:(i + self.wndw_size), j:(j + self.wndw_size)]
                    Iq_b = Iq_b.reshape(1, -1)
                    
                    tmp_r = Gr * Iq_r
                    tmp_g = Gr * Iq_g
                    tmp_b = Gr * Iq_b
                    
                    output[0][i, j] = np.dot(Guassian_spatial, tmp_r.T) / np.dot(Guassian_spatial, Gr.T)
                    output[1][i, j] = np.dot(Guassian_spatial, tmp_g.T) / np.dot(Guassian_spatial, Gr.T)
                    output[2][i, j] = np.dot(Guassian_spatial, tmp_b.T) / np.dot(Guassian_spatial, Gr.T)
        return output


    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        output = np.zeros(shape=(3, img.shape[0], img.shape[1]))
        x, y = np.mgrid[-(self.pad_w):(self.pad_w + 1), -(self.pad_w):(self.pad_w + 1)]
        Guassian_spatial = np.exp(-(x ** 2 + y ** 2) / (2 * (self.sigma_s ** 2) ))
        Guassian_spatial = Guassian_spatial.reshape(1, -1)
        padded_img = padded_img.transpose(2, 0, 1)
        
        # check the channel of image
        num_channel = len(padded_guidance.shape)
        if(num_channel == 3):
            padded_guidance = padded_guidance.transpose(2, 0, 1)
            output = self.do_filter(img, padded_guidance, padded_img, Guassian_spatial, output, num_channel= 3)

        elif(num_channel == 2):
            output = self.do_filter(guidance, padded_guidance, padded_img, Guassian_spatial, output, num_channel = 1)
    
        output = output.transpose(1, 2, 0)
        
        return np.clip(output, 0, 255).astype(np.uint8)