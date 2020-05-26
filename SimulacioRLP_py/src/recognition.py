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

        # Utlitzem l'algorisme ORB per a obtenir els descriptors ja que es un dels més rapids.
        self.orb = cv2.ORB_create(edgeThreshold=81) #131
        # Utilitzem el FAST per a la feature detection.

        self.thresh_inicial = 12
        self.fast = cv2.FastFeatureDetector_create(self.thresh_inicial, True, 2)
        """
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        """

        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=2)

        search_params = {}
        #search_params = dict(checks=100)
        # Utilitzem un "flann based matcher" per posar en correspondéncia els punts.
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        """
        # Hem intentat reconeixer la pilota utilitzant el blob detector, però finalment hem optat
        # pre un threshold de color. 
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
        """

    def set_src_img(self, img):

        # En primer lloc definim la imatge de referénca (Laberint totalment plà).
        self.source_img = img
        self.source_img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detectem les features i descrptors de la imatge de referéncia.
        self.s_kp = self.fast.detect(self.source_img_g, None)
        self.s_kp, self.s_des = self.orb.compute(self.source_img_g, self.s_kp)

    def digitize_source(self):
        # Mitjançant l'adaptative thresholding podem separar les parets del terra evitant la intromissió de les
        # sombres.
        b = cv2.adaptiveThreshold(self.source_img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,61, 15)
        #cv2.imshow("Binary", b)

        # Les posicións inicial i final estàn señalitzades amb un punt verd i un vermell respectivament.
        # Apliquem un thresholding un cop feta la substracció de la resta de colors, es a dir, si volem trobar el punt
        # final (verd), restem els canals blau i vermell (ponderats) al canal verd i apliquem un threshold.
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
        print("START POS: ", self.startPos)

        i = cv2.circle(cv2.circle(self.source_img, (startPosX, startPosY), 8, (0,0,255)), (endPosX, endPosY), 8, (0, 255, 0))
        cv2.imshow("Positions", i)

        # Definim que es terra i que es paret segons on es troben els punts inicals i finals
        if b[startPosY + POS_DIST_CHECK, startPosX + POS_DIST_CHECK] == 0:
            b = 255-b

        # Ens assegurem que les posicións pròximes als punts inicials i finals no es troben obstruides
        b[startPosY-POS_DIST_CHECK:startPosY+POS_DIST_CHECK, startPosX-POS_DIST_CHECK:startPosX+POS_DIST_CHECK] = 255
        b[endPosY - POS_DIST_CHECK:endPosY + POS_DIST_CHECK,endPosX - POS_DIST_CHECK:endPosX + POS_DIST_CHECK] = 255
        #cv2.imshow("corrected b,", b)

        # Apliquem els filtres morfológic erode i dilate per a eliminar el soroll present en el terra sense reduir el
        # tamany de les parets
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        er = cv2.erode(255-b, kernel)
        di = cv2.dilate(er, kernel)
        b = ((255-di) / 255).astype('uint8')

        # Detectem els forats del laberint utilitzant el SimpleBlobDetector de opencv.
        params = cv2.SimpleBlobDetector_Params()
        # Change thresholds
        params.minThreshold = 0  # the graylevel of images
        params.maxThreshold = 250

        params.filterByColor = True
        params.blobColor = 0

        # Filter by Area
        params.filterByArea = True
        params.minArea = 1500
        params.maxArea = 3000
        b_detector = cv2.SimpleBlobDetector_create(params)

        h_keypoints = b_detector.detect(self.source_img)
        im_with_keypoints = cv2.drawKeypoints(self.source_img_g, h_keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #cv2.imshow("Holes", im_with_keypoints)

        # Pintem els forats a la mascara final
        for kp in h_keypoints:
            b = cv2.circle(b, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size / 2)+3, 1, thickness=-1, lineType=cv2.LINE_AA)
            b = cv2.circle(b, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size/2)+1, 2, thickness=-1, lineType=cv2.LINE_AA)

        # Establim els valors finals de la màscara (0 terra, 1 paret, 2 forat)
        wall = b == 0
        ground = b == 1
        b[wall] = 1
        b[ground] = 0

        self.source_mask = b
        cv2.imshow("Final mask", self.source_mask*127)
        #cv2.imwrite("out1.jpg", self.source_mask*127)

    def get_ball_pos(self, img):
        trobat = False
        th = self.thresh_inicial
        while not trobat:
            img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Obtenim punts i descriptors del frame
            kp = self.fast.detect(img_g, None)
            kp, des = self.orb.compute(img_g, kp)
            img_kp = cv2.drawKeypoints(img_g, kp, None, color=(255, 0, 0))
            #cv2.imshow("Frame KP", img_kp)

            # Apliquem el flann matching per trobar la correspondéncia entre els punts del frame actual i els de la
            # imatge de referénca.
            matches = self.flann.knnMatch(des, self.s_des, k=2)
            ratio_thresh = 0.45
            good_matches = []

            try:
                for m, n in matches:
                    if m.distance < ratio_thresh * n.distance:
                        good_matches.append(m)
                trobat = True

            except:
                th -= 1
                self.fast = cv2.FastFeatureDetector_create(th, True, 2)

        # Ens assegurem de tenir 4 punts com a mínim.
        if len(good_matches) < 4:
            return False
        
        img3 = cv2.drawMatches(img_g, kp, self.source_img_g, self.s_kp, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("Matches", img3)
        src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.s_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calculem l'hmografía i apliquem la correcció de perspectiva al frame.
        M, mask = cv2.findHomography(src_pts, dst_pts)
        n_img = cv2.warpPerspective(img, M, (PI_CAMERA_RES, PI_CAMERA_RES))
        cv2.imshow("Wrapped frame", n_img)
        #cv2.imwrite("out2_1.jpg", n_img)
        #check_result = cv2.addWeighted(self.source_img, 0.5, n_img, 0., 1)

        # Mitjançant un threshold bàsic amb substracció com en el cas de les posicións, trobem els pixels
        # corresponents a la pilota.
        th = n_img[:,:,0].astype('int16') - 0.3*(n_img[:,:,1].astype('int16') + n_img[:,:,2]) > 120

        #ball_mask = np.zeros((PI_CAMERA_RES, PI_CAMERA_RES), dtype='uint8')
        #ball_mask[th] = 255
        #cv2.imshow("Ball mask", ball_mask)
        ball_indexs = np.argwhere(th)

        #cv2.imwrite("out2_2.jpg", ball_mask)

        # Calculem la posició de la pilota amb la mitjana dels indexs corresponents als pixels de la pilota.
        if ball_indexs.any():
            # ballPos [y,x]
            self.lastBallPos = ball_indexs.mean(axis=0)

        return self.lastBallPos


# Testing
if __name__ == "__main__":
    d = Digitizer()
    d.set_src_img(cv2.imread('inp1.jpg', 1))
    d.digitize_source()
    d.get_ball_pos(cv2.imread('inp2.jpg', 1))
    cv2.waitKey()

