import cv2
import numpy as np
#from matplotlib import pyplot as plt
from math import sqrt
import time


class aStar:

	def distance(self, punt0, puntf):
		x0, y0 = punt0
		xf, yf = puntf
		return sqrt((xf - x0)*(xf - x0) + (yf - y0)*(yf - y0))

	def get_neighbours(self, punt, end_p, lab, exp, width, height):
		x, y = punt[0]
		dist = punt[1]
		ret = []
		
		if(lab[x][y-1] == 0 and exp[x][y-1] == -1):
			ret.append([(x, y-1), dist + 1, self.distance((x, y-1), end_p)])	
		if(x+1 >= 0 and x < width and lab[x][y+1] == 0 and exp[x][y+1] == -1):
			ret.append([(x, y+1), dist + 1, self.distance((x, y+1), end_p)])
		if(lab[x-1][y] == 0 and exp[x-1][y] == -1):
			ret.append([(x-1, y), dist + 1, self.distance((x-1, y), end_p)])
		if(lab[x+1][y] == 0 and exp[x+1][y] == -1):
			ret.append([(x+1, y), dist + 1, self.distance((x+1, y), end_p)])

		return ret

	def insert_ordered(self, point_l, insert_l):
		for p in insert_l:
			dist = p[1] + p[2]
			#dist = p[2]

			for index, i in enumerate(point_l):
				d = i[1] + i[2]
				#d = i[2]
				if(dist < d):
					point_l.insert(index, p)
					break
			else:
				point_l.append(p)
		return point_l

	def a_star(self, lab, radi, mseg, x0, y0, xf, yf): #Assumeix lliure = 0, paret = 1, forat = 2
		
		# ------- Inflar parets i forats per tenir en compte radi i marge de seguretat

		#print(lab)

		lab[0, :] = 1
		lab[:, 0] = 1
		lab[-1, :] = 1
		lab[:, -1] = 1

		width = lab.shape[0]
		height = lab.shape[1]

		parets = np.zeros(lab.shape).astype(np.uint8)
		parets[lab == 1] = 1

		forats = np.zeros(lab.shape).astype(np.uint8)
		forats[lab == 2] = 1

		SEd = np.zeros((radi*2,radi*2), np.uint8)
		cv2.circle(SEd, (radi, radi), radi, 1, -1)

		parets = cv2.dilate(parets,SEd,iterations = 1)

		#plt.imshow(parets*255, 'gray')
		#plt.show()

		lab[parets == 1] = 1

		

		SEd = np.zeros((mseg*2,mseg*2), np.uint8)
		cv2.circle(SEd, (mseg, mseg), mseg, 1, -1)

		forats = cv2.dilate(forats,SEd,iterations = 1)

		#plt.imshow(forats*255, 'gray')
		#plt.show()

		lab[forats == 1] = 2

		start_p = (x0, y0)
		end_p = (xf, yf)

		if(lab[x0][y0] != 0 or lab[xf][yf] != 0):
			print("Posició inici o final no valida")
			return False, None, None

		expand_p = -np.ones(lab.shape).astype(np.int)
		
		# -------------------- Start expanding --------------------

		expand_p[x0][y0] = 0

		point_list = [[(x0, y0), 0, self.distance((x0, y0), (xf, yf))]]

		# ------------------ Main bucle --------------------------

		start_time = time.time()

		while(len(point_list) > 0):
			punt_actual = point_list.pop(0)
			#print(punt_actual)
			if(punt_actual[0] == end_p):
				break
			
			neighs = self.get_neighbours(punt_actual, (xf, yf), lab, expand_p, width, height)
			for n in neighs:
				expand_p[n[0][0]][n[0][1]] = n[1]
			#print(expand_p[punt_actual[0][0] - 3 : punt_actual[0][0] + 4, punt_actual[0][1] - 3 : punt_actual[0][1] + 4])
			#cv2.imshow("", (expand_p != -1).astype(np.uint8)*255)
			#cv2.waitKey(1)
			self.insert_ordered(point_list, neighs)
			#print(point_list)
			#input()
			if(len(point_list) == 0):
				return False, None, None

		# -------- Recuperar cami ------------------

		expand_p[expand_p == -1] = 100000

		path = [end_p]
		path_matrix = np.zeros(lab.shape).astype(int)

		punt_actual_1 = end_p

		punt_actual_2 = end_p

		punt_actual = end_p

		dir_anterior = tuple(map(lambda i, j: i - j, punt_actual, punt_actual_2))

		path_matrix[punt_actual[0] - 1 : punt_actual[0] + 2, punt_actual[1] - 1 : punt_actual[1] + 2] = 1

		while(expand_p[punt_actual] != 0):

			a = expand_p[punt_actual[0] - 1 : punt_actual[0] + 2, punt_actual[1] - 1 : punt_actual[1] + 2]

			#print(a)

			#input()

			ind = np.unravel_index(np.argmin(a, axis=None), a.shape)

			punt_actual_2 = punt_actual_1

			punt_actual_1 = punt_actual

			punt_actual = tuple(map(sum, zip(punt_actual, ind, (-1, -1))))

			direccio = tuple(map(lambda i, j: i - j, punt_actual, punt_actual_2))

			path_matrix[punt_actual[0] - 1 : punt_actual[0] + 2, punt_actual[1] - 1 : punt_actual[1] + 2] = 1

			if(direccio != dir_anterior):
				path.insert(0, punt_actual)

			dir_anterior = direccio

		#print("Temps total: ", time.time() - start_time)

		return True, path_matrix, path

