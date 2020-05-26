class pid:
	def __init__(self, xInit, yInit, p = 0.1, d = 0.14):
		self.p = p
		self.d = d
		self.pAngle = 0.5
		self.prevX = xInit
		self.prevY = yInit

	def getPR(self, x0, y0, xf, yf, p_act, r_act, dt):
		difX = xf - x0
		difY = yf - y0

		velX = (x0 - self.prevX) / dt
		velY = (y0 - self.prevY) / dt

		self.prevX = x0
		self.prevY = y0

		angleR = difX*self.p - velX*self.d
		angleP = difY*self.p - velY*self.d

		retR = max(min((angleR - r_act)*self.pAngle, 1), -1)
		retP = max(min((angleP - p_act)*self.pAngle, 1), -1)

		return retP, retR