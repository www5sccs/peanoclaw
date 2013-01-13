'''
Random Terrain Generation Using The Diamond-Square Algorithm
By Ryan Lowerr 
Email Ryan dot Lowerr at gmail dot com
'''

import random
 
class FractalTerrain:

	def __init__(self, iterations=10, seed=10.0, deviation=5.0, roughness=1.5):
		self.Iterations = iterations
		self.Seed = seed
		self.Deviation = deviation
		self.Roughness = roughness
		self.Size = 2**self.Iterations + 1
		self.Vertices = [[0 for x in range(0, self.Size)] for y in range(0, self.Size)]
		
		# seed the four corner vertices
		self.SeedVerts(self.Seed)
		
		# generate the remaining vertices
		self.GenVerts()
		
	def SeedVerts(self, seed):
		self.Vertices[0][0] = seed
		self.Vertices[0][-1] = seed
		self.Vertices[-1][0] = seed
		self.Vertices[-1][-1] = seed
		
	def SeedVerts2(self, seed):
		self.Vertices[0][0] = random.gauss(self.Seed, self.Deviation)
		self.Vertices[0][-1] = random.gauss(self.Seed, self.Deviation)
		self.Vertices[-1][0] = random.gauss(self.Seed, self.Deviation)
		self.Vertices[-1][-1] = random.gauss(self.Seed, self.Deviation)
		
	def GenVerts(self):
	
		# how many units (width/height) the array is
		size = self.Size - 1
		deviation = self.Deviation
		roughness = self.Roughness
		
		for i in range(self.Iterations):
		
			span = size / 2**(i+1)
			span2 = span*2
		
			for x in range(2**i):
				for y in range(2**i):
					dx = x * span2
					dy = y * span2
				
					# diamond step
					A = self.Vertices[dx][dy]
					B = self.Vertices[dx + span2][dy]
					C = self.Vertices[dx + span2][dy + span2]
					D = self.Vertices[dx][dy + span2]
					E = random.gauss(((A + B + C + D) / 4.0), deviation)
					
					if self.Vertices[dx + span][dy + span] == 0.0:
						self.Vertices[dx + span][dy + span] = E
						
					# squared step
					if self.Vertices[dx][dy + span] == 0.0:
						self.Vertices[dx][dy + span] = random.gauss(((A + C + E) / 3.0), deviation) # F
						
					if self.Vertices[dx + span][dy] == 0.0:
						self.Vertices[dx + span][dy] = random.gauss(((A + B + E) / 3.0), deviation) # G
						
					if self.Vertices[dx + span2][dy + span] == 0.0:
						self.Vertices[dx + span2][dy + span] = random.gauss(((B + D + E) / 3.0), deviation) # H
						
					if self.Vertices[dx + span][dy + span2] == 0.0:
						self.Vertices[dx + span][dy + span2] = random.gauss(((C + D + E) / 3.0), deviation) # I
					
			deviation = deviation * (2**-roughness)
			
	def PrintVerts(self):
		for x in range(self.Size):
			for y in range(self.Size):
				print self.Vertices[x][y],
			print ''
