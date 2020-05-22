import cv2
import numpy as np
from constants import *

class Digitizer:
    def __init__(self):
        self.source_img = None
        self.source_img_g = None
        self.source_mask = None

        self.startPos = None
        self.endPos = None

        self.lastBallPos = None

        self.s_kp = None
        self.s_des = None
        self.orb = cv2.ORB_create(edgeThreshold=131)
        self.fast = cv2.FastFeatureDetector_create(12, True, 2)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = {}
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 20  # the graylevel of images
        params.maxThreshold = 250

        params.filterByColor = True
        params.blobColor = 255

        params.filterByInertia = True
        params.minInertiaRatio = 0.1
        params.maxInertiaRatio = 1

        params.filterByCircularity = True
        params.minCircularity = 0.1
        params.maxCircularity = 1

        # Filter by Area
        params.filterByArea = True
        #params.minArea = 1100
        #params.maxArea = 1500
        params.minArea = 900
        params.maxArea = 3000
        self.b_detector = cv2.SimpleBlobDetector_create(params)

    def set_src_img(self, img):
        #img = np.clip(img.astype('float') * 2 + 10, 0, 255).astype('uint8')
        self.source_img = img
        self.source_img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #self.s_kp = self.ffd.detect(img, None)

        # find the keypoints with ORB
        #self.s_kp, self.s_des = self.orb.detectAndCompute(self.source_img_g, None)
        self.s_kp = self.fast.detect(self.source_img_g, None)
        self.s_kp, self.s_des = self.orb.compute(self.source_img_g, self.s_kp)
        img2 = cv2.drawKeypoints(self.source_img_g, self.s_kp, None, color=(255, 0, 0))
        cv2.imshow("Eye view image fast", img2)

    def digitize_source(self):
        #b = cv2.threshold(self.source_img_g, WG_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        b = cv2.adaptiveThreshold(self.source_img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 51, 5)
        #cv2.imshow("bin,", b)

        endPos = np.argmax(self.source_img[:, :, 1].astype('int16') -
                             (self.source_img[:,:,0].astype('int16') + self.source_img[:,:,2].astype('int16'))*0.4)
        endPosY = int(endPos / PI_CAMERA_RES)
        endPosX = endPos % PI_CAMERA_RES
        self.endPos = (endPosX, endPosY)

        startPos = np.argmax(self.source_img[:, :, 2].astype('int16') -
                             (self.source_img[:,:,0].astype('int16') + self.source_img[:,:,1].astype('int16'))*0.4)
        startPosY = int(startPos / PI_CAMERA_RES)
        startPosX = startPos % PI_CAMERA_RES
        self.startPos = (startPosX, startPosY)

        i = cv2.circle(cv2.circle(self.source_img, (startPosX, startPosY), 8, (0,0,255)), (endPosX, endPosY), 8, (0, 255, 0))
        #cv2.imshow("positions", i)

        if b[startPosY + POS_DIST_CHECK, startPosX + POS_DIST_CHECK] == 0:
            b = 255-b
            print("INVERTED BINARY")

        b[startPosY-POS_DIST_CHECK:startPosY+POS_DIST_CHECK, startPosX-POS_DIST_CHECK:startPosX+POS_DIST_CHECK] = 255
        b[endPosY - POS_DIST_CHECK:endPosY + POS_DIST_CHECK,endPosX - POS_DIST_CHECK:endPosX + POS_DIST_CHECK] = 255
        cv2.imshow("corrected b,", b)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        er = cv2.erode(255-b, kernel)
        di = cv2.dilate(er, kernel)
        b = ((255-di) / 255).astype('uint8')

        # Holes ---
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 20  # the graylevel of images
        params.maxThreshold = 200

        params.filterByColor = True
        params.blobColor = 0

        # Filter by Area
        params.filterByArea = True
        params.minArea = 1000
        params.maxArea = 2000
        b_detector = cv2.SimpleBlobDetector_create(params)

        h_keypoints = b_detector.detect(self.source_img_g)

        im_with_keypoints = cv2.drawKeypoints(self.source_img_g, h_keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("holes", im_with_keypoints)
        # ---------

        # Holes insertion
        for kp in h_keypoints:
            b = cv2.circle(b, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2)+3, 1, thickness=-1, lineType=cv2.LINE_AA)
            b = cv2.circle(b, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size/2)+1, 2, thickness=-1, lineType=cv2.LINE_AA)

        wall = b == 0
        ground = b == 1

        b[wall] = 1
        b[ground] = 0

        self.source_mask = b

        cv2.imshow("final mask", self.source_mask*127)

        # TODO: Send b to A*

    def get_ball_pos(self, img):
        #img = cv2.blur(img, (3, 3))

        #kp, des = self.orb.detectAndCompute(img, None)

        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp = self.fast.detect(img_g, None)
        kp, des = self.orb.compute(img_g, kp)
        #img_kp = cv2.drawKeypoints(img_g, kp, None, color=(255, 0, 0))
        #cv2.imshow("frame image", img_kp)

        matches = self.flann.knnMatch(des, self.s_des, k=2)
        ratio_thresh = 0.5
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return False
        
        #img3 = cv2.drawMatches(img_g, kp, self.source_img_g, self.s_kp, good_matches, None,
        #                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        #cv2.imshow("matches", img3)
        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.s_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        """

        matches = self.bf.match(des, self.s_des)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = []
        i=0
        ml = len(matches)
        for m in matches:
            print(kp[m.queryIdx].pt, "-", kp[m.trainIdx].pt)
            print("Dist1: ", abs(kp[m.queryIdx].pt[0] - kp[m.trainIdx].pt[0]))
            print("Dist2: ", abs(kp[m.queryIdx].pt[1] - kp[m.trainIdx].pt[1]))
            if (abs(kp[m.queryIdx].pt[0] - kp[m.trainIdx].pt[0]) < 120 and
            abs(kp[m.queryIdx].pt[1] - kp[m.trainIdx].pt[1]) < 120):
                print(m.distance)
                good_matches.append(m)
                i += 1
            print("-------------------")

        matches = good_matches

        if len(matches) < 4:
            return False
        #matches = matches[:8]
        # Draw matches.
        img3 = cv2.drawMatches(img, kp, self.source_img_g, self.s_kp, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("matches", img3)

        src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.s_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        """
        #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        M, mask = cv2.findHomography(src_pts, dst_pts)
        n_img = cv2.warpPerspective(img, M, (PI_CAMERA_RES, PI_CAMERA_RES))
        #cv2.imshow("wrapped", n_img)
        #check_result = cv2.addWeighted(self.source_img, 0.5, n_img, 0.5, 1)
        #cv2.imshow("show result", check_result)

        #ball_mask = np.zeros((512, 512), dtype='uint8')
        th = n_img[:,:,0].astype('int16') - 0.3*(n_img[:,:,1].astype('int16') + n_img[:,:,2]) > 200
        #ball_mask[th] = 255
        #cv2.imshow("ball mask", ball_mask)
        #cv2.imshow("n_img", n_img)
        ball_indexs = np.argwhere(th)

        if ball_indexs.any():
            # ballPos [y,x]
            self.lastBallPos = ball_indexs.mean(axis=0)

        return self.lastBallPos

        #h_keypoints = self.b_detector.detect(n_img)
        #print("N Keypoints: ", len(h_keypoints))

        #im_with_keypoints = cv2.drawKeypoints(n_img, h_keypoints, np.array([]), (0, 0, 255),
        #                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("ball pos", im_with_keypoints)
        """
        if len(h_keypoints) == 1:
            self.lastBallPos = h_keypoints[0].pt
            return h_keypoints[0].pt
        else:
            return self.lastBallPos
        """
# Testing
if __name__ == "__main__":
    d = Digitizer()
    d.set_src_img(cv2.imread('frame_tests/fs.jpg', 1))
    d.digitize_source()
    d.get_ball_pos(cv2.imread('frame_tests/ft6.jpg', 1))
    cv2.waitKey()
