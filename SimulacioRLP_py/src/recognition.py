import cv2
import numpy as np

class Digitizer:
    def __init__(self):
        self.eye_view = None
        self.s_kp = None
        self.s_des = None
        #self.ffd = cv2.FastFeatureDetector_create()
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    def set_eye_view_img(self, img):
        self.eye_view = img
        #self.s_kp = self.ffd.detect(img, None)

        # find the keypoints with ORB
        self.s_kp = self.orb.detect(img, None)
        # compute the descriptors with ORB
        self.s_kp , self.s_des = self.orb.compute(img, self.s_kp)

        img_kp = cv2.drawKeypoints(img, self.s_kp, None, color=(255, 0, 0))
        cv2.imshow("Eye view image", img_kp)



    def get_next_move(self, img):
        img = cv2.imread('test1.jpg')
        self._digitalize(img)
        #self.codi_narcis.get_rotacio_i_distancia o velictat(pos_pilota, angle_actual)
    def _digitalize(self, img):
        h, w, c = img.shape
        #img = cv2.blur(img, (3, 3))

        kp = self.orb.detect(img, None)
        # compute the descriptors with ORB
        kp, des = self.orb.compute(img, kp)

        img_kp = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        cv2.imshow("Softed image", img_kp)

        matches = self.bf.match(self.s_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(self.eye_view, self.s_kp, img, kp, matches[:4], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("matches", img3)

# Testing
if __name__ == "__main__":
    d = Digitizer()
    d.set_eye_view_img(cv2.imread('eye_view.jpg', 0))
    d.get_next_move(None)
    cv2.waitKey()