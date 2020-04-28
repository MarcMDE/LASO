import cv2
import numpy as np

class Digitizer:
    def __init__(self):
        self.eye_view = None
        self.s_kp = None
        self.s_des = None
        #self.ffd = cv2.FastFeatureDetector_create()
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        """
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)
        search_params = {}
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        """
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

    def _digitalize_source(self):
        pass


    def _digitalize(self, img):
        h, w, c = img.shape
        #img = cv2.blur(img, (3, 3))

        kp = self.orb.detect(img, None)
        # compute the descriptors with ORB
        kp, des = self.orb.compute(img, kp)

        img_kp = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))
        cv2.imshow("Softed image", img_kp)
        """
        matches = self.flann.knnMatch(des, self.s_des, k=2)
        ratio_thresh = 0.7
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        print(len(good_matches))
        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.s_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        """


        matches = self.bf.match(des, self.s_des)
        matches = sorted(matches, key=lambda x: x.distance)
        matches = matches[:8]
        # Draw first 8 matches.
        img3 = cv2.drawMatches(img, kp, self.eye_view, self.s_kp, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("matches", img3)

        src_pts = np.float32([kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.s_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        M, mask = cv2.findHomography(src_pts, dst_pts)
        n_img = cv2.warpPerspective(img, M, (512, 512))
        cv2.imshow("wrapped", n_img)
        check_result = cv2.addWeighted(self.eye_view, 0.5, n_img, 0.5, 1)
        cv2.imshow("show result", check_result)

# Testing
if __name__ == "__main__":
    d = Digitizer()
    d.set_eye_view_img(cv2.imread('eye_view.jpg', 1))
    d.get_next_move(None)
    cv2.waitKey()