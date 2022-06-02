import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = 'Homework/HW1-1-data/'
nums = ['0000', '1131', '1262', '1755']

# For each image read infra1 (left), infra2(right) and color
for num in nums:
    right_image  = cv2.imread(f"{DATA_DIR}/infra2_{num}.jpg", cv2.IMREAD_GRAYSCALE)
    left_image = cv2.imread(f"{DATA_DIR}/infra1_{num}.jpg", cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(f"{DATA_DIR}/color{num}.jpg", cv2.IMREAD_COLOR)


    # More details could be found here https://github.com/IntelRealSense/librealsense/blob/master/doc/depth-from-stereo.md
    focal_length = 970  # lense focal length
    baseline = 50       # distance in mm between the two cameras
    disparities = 128   # num of disparities to consider
    block = 31          # block size to match
    units = 0.001       # depth units (mm to meter)

    # Create Stereo Block Matcher to calculate disparity.
    sbm = cv2.StereoBM_create(numDisparities=disparities,
                            blockSize=block)

    disparity = sbm.compute(left_image, right_image)
    

    # calculate the depth using triangle similarity
    # - units converts mm to meter
    # - we ignore pixels where disparity is 0 or less because we do not have a valid triangle.
    depth = np.zeros(shape=left_image.shape).astype(float)

    depth[disparity > 0] = (focal_length * baseline) / (units * disparity[disparity > 0])

    cv2.imshow(f"{num}_infra_left", left_image)
    cv2.imshow(f"{num}_reconstructed_depth", depth)

    cv2.waitKey()

cv2.destroyAllWindows()
'''
- Laser Pattern is Off:
  - If the image has texture , like the surface of the chessboard block matching performance is good.
  - If the image has no texture, the block matching performance is worse.
- Laser Pattern is On:
  - Block Matching in generale is better, especially for surfaces without texture.
'''