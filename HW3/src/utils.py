from turtle import shape
from matplotlib.pyplot import axis
import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    ux = u[:, 0].reshape((N, 1))
    uy = u[:, 1].reshape((N, 1))
    vx = v[:, 0].reshape((N, 1))
    vy = v[:, 1].reshape((N, 1))
    # with the formula of solution2(p8、9)
    upA =  np.concatenate((ux, uy, np.ones(shape=(N,1)), np.zeros(shape=(N,3)), -1*(ux*vx), -1*(uy*vx), -vx), axis= 1)
    downA = np.concatenate((np.zeros(shape=(N,3)), ux, uy, np.ones(shape=(N,1)), -1*(ux*vy), -1*(uy*vy), -vy), axis= 1)
    A = np.concatenate((upA, downA), axis= 0)
    # TODO: 2.solve H with A
    U, sigma, V_transpose = np.linalg.svd(A)
    h = V_transpose[-1,:]/V_transpose[-1,-1]
    H = h.reshape(3, 3)
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    # with th pdf (p40 tips for acclerating)
    x, y = np.meshgrid(np.arange(xmin, xmax, 1), np.arange(ymin, ymax, 1), sparse= False)
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    xrow = x.reshape(1, -1)
    yrow = y.reshape(1, -1)
    matrix = np.concatenate((xrow, yrow, np.ones(shape=xrow.shape)), axis= 0)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst_matrix = np.dot(H_inv, matrix)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        dst_matrix = np.divide(dst_matrix, dst_matrix[-1, :]) # with dividing the last row can get the 座標點
        srcy = np.round( dst_matrix[1,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)
        srcx = np.round( dst_matrix[0,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)  
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        # make sure in the region
        h_mask = (0<srcy)*(srcy<h_src)
        w_mask = (0<srcx)*(srcx<w_src)
        mask = h_mask * w_mask # if in region return true, else return false
        # TODO: 6. assign to destination image with proper masking
        dst[y[mask], x[mask]] = src[srcy[mask], srcx[mask]]
        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst_matrix = np.dot(H, matrix)
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        dst_matrix = np.divide(dst_matrix, dst_matrix[-1, :])
        dsty = np.round( dst_matrix[1,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)
        dstx = np.round( dst_matrix[0,:].reshape((ymax-ymin, xmax-xmin)) ).astype(int)  
        # TODO: 5.filter the valid coordinates using previous obtained mask
        # TODO: 6. assign to destination image using advanced array indicing
        dst[np.clip(dsty, 0, dst.shape[0]-1), np.clip(dstx, 0, dst.shape[1]-1)] = src
        pass

    return dst
