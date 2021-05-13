import traceback
import cv2
import numpy as np


def createDetector():
    detector = cv2.ORB_create(nfeatures=2000)
    return detector


def getFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = createDetector()
    kps, descs = detector.detectAndCompute(gray, None) #tim keypoint, descriptor cua tung keypoint
    return kps, descs, img.shape[:2][::-1]


def detectFeatures(img, train_features):
    train_kps, train_descs, shape = train_features
    # get features from input image
    kps, descs, _ = getFeatures(img)
    # check if keypoints are extracted
    if not kps:
        return None
    # now we need to find matching keypoints in two sets of descriptors (from sample image, and from current image)
    # knnMatch uses k-nearest neighbors algorithm for that
    #use brute force matcher. 
    # It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation. 
    # And the closest one is returned.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING) #calculate in hamming distance == number of different bits between two strings
    matches = bf.knnMatch(train_descs, descs, k=2) #get 2 best matches
    print(matches[0])
    good = []
    # apply ratio test to matches of each keypoint
    # idea is if train KP have a matching KP on image, it will be much closer than next closest non-matching KP,
    # otherwise, all KPs will be almost equally far
    #LOWE RATIO TEST:: 
    # each keypoint of the first image is matched with a number of keypoints from the second image. 
    # We keep the 2 best matches for each keypoint (best matches = the ones with the smallest distance measurement). 
    # Lowe's test checks that the two distances are sufficiently different. 
    # If they are not, then the keypoint is eliminated and will not be used for further calculations.
    try:
        for m, n in matches:
            # if m.distance < 1 * n.distance:
            good.append([m])

        # stop if we didn't find enough matching keypoints
        if len(good) < 0.1 * len(train_kps):
            return None

        # estimate a transformation matrix which maps keypoints from train image coordinates to sample image
        src_pts = np.float32([train_kps[m[0].queryIdx].pt for m in good
                              ]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m[0].trainIdx].pt for m in good
                              ]).reshape(-1, 1, 2)

        m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if m is not None:
            # apply perspective transform to train image corners to get a bounding box coordinates on a sample image
            scene_points = cv2.perspectiveTransform(np.float32([(0, 0), (0, shape[0] - 1),
                                                                (shape[1] - 1, shape[0] - 1),
                                                                (shape[1] - 1, 0)]).reshape(-1, 1, 2), m)
            rect = cv2.minAreaRect(scene_points)
            # check resulting rect ratio knowing we have almost square train image

            # return rect
            if rect[1][1] > 0 and 0.8 < (rect[1][0] / rect[1][1]) < 1.2:
                return rect
    except:
        pass
    return None
