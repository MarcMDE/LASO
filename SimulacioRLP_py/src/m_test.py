import cv2
import numpy as np
from constants import *
from recognition import Digitizer
from aStar import aStar

if __name__ == "__main__":
    d = Digitizer()
    d.set_src_img(cv2.imread('frame_tests/sg.jpg', 1))
    d.digitalize_source()

    a = aStar()
    pm, p = a.a_star(d.source_mask, 24, 40, d.startPos[1], d.startPos[0], d.endPos[1], d.endPos[0])
    cv2.imshow("laberint resolt", np.clip(pm*255, 0, 255).astype('uint8'))

    check_result = cv2.addWeighted(d.source_img_g.astype('uint8'), 0.5, np.clip(pm*255, 0, 255).astype('uint8'), 0.5, 1)
    cv2.imshow("laberint resolt sobre original", check_result)
    #d.get_ball_pos(cv2.imread('tg4.jpg', 0))
    cv2.waitKey()
