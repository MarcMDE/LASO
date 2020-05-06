import cv2
import numpy as np
from constants import *

class Digitizer:
    def __init__(self):
        self.source_img = None
        self.source_img_g = None
        self.mask = None


        self.s_kp = None
        self.s_des = None
        self.orb = cv2.ORB_create(edgeThreshold=51)
        self.fast = cv2.FastFeatureDetector_create(11, True, 2)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = {}
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

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

    def get_next_move(self, img):
        img = cv2.imread('test_dw_2.jpg', 0)
        #img = np.clip(img.astype('float')*2+10, 0, 255).astype('uint8')
        self._digitalize(img)
        #self.codi_narcis.get_rotacio_i_distancia o velictat(pos_pilota, angle_actual)

    def _digitalize_source(self):
        #b = cv2.threshold(self.source_img_g, WG_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        b = cv2.adaptiveThreshold(self.source_img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 51, 5)
        cv2.imshow("bin,", b)

        endPos = np.argmax(self.source_img_g)
        endPosY = int(endPos / PI_CAMERA_RES)
        endPosX = endPos % PI_CAMERA_RES
        startPos = np.argmax(self.source_img[:, :, 2].astype('int16') -
                             (self.source_img[:,:,0].astype('int16') + self.source_img[:,:,1].astype('int16'))*0.4)
        startPosY = int(startPos / PI_CAMERA_RES)
        startPosX = startPos % PI_CAMERA_RES
        i = cv2.circle(cv2.circle(self.source_img, (startPosX, startPosY), 8, (255,0,0)), (endPosX, endPosY), 8, (0, 255, 0))
        cv2.imshow("positions", i)

        if b[startPosY + POS_DIST_CHECK, startPosX + POS_DIST_CHECK] == 0:
            b = 255-b
            print("INVERTED BINARY")

        b[startPosY-POS_DIST_CHECK:startPosY+POS_DIST_CHECK, startPosX-POS_DIST_CHECK:startPosX+POS_DIST_CHECK] = 255
        b[endPosY - POS_DIST_CHECK:endPosY + POS_DIST_CHECK,endPosX - POS_DIST_CHECK:endPosX + POS_DIST_CHECK] = 255
        cv2.imshow("corrected b,", b)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        er = cv2.erode(255-b, kernel)
        di = cv2.dilate(er, kernel)
        b = ((255-di) / 255).astype('uint8')
        final = cv2.imshow("final", b*255)

        # TODO: Send b to A*

    def get_ball_pos(self, img):
        #img = cv2.blur(img, (3, 3))

        #kp, des = self.orb.detectAndCompute(img, None)

        kp = self.fast.detect(img, None)
        kp, des = self.orb.compute(img, kp)

        img_kp = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        cv2.imshow("frame image", img_kp)

        matches = self.flann.knnMatch(des, self.s_des, k=2)
        ratio_thresh = 0.3
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        print(len(good_matches))
        if len(good_matches) < 4:
            return False
        
        img3 = cv2.drawMatches(img, kp, self.source_img_g, self.s_kp, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("matches", img3)
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
        cv2.imshow("wrapped", n_img)
        check_result = cv2.addWeighted(self.source_img_g, 0.5, n_img, 0.5, 1)
        cv2.imshow("show result", check_result)

# Testing
if __name__ == "__main__":
    d = Digitizer()
    d.set_src_img(cv2.imread('s.jpg', 1))
    d._digitalize_source()
    #d.get_ball_pos(cv2.imread('t5.jpg', 0))
    cv2.waitKey()
