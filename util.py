import sys
import numpy as np

#http://www.scipy.org/
try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))


def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors :
		cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	
	#norm1 = norm(vector1)
	#norm2 = norm(vector2)

	#if norm1 == 0 or norm2 == 0:
		#return 0.0
	#return float(dot(vector1, vector2) / (norm1 * norm2))
	return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))

def euclideandistance(vector1, vector2):
	if len(vector1) == 0 or len(vector2) == 0:
		return float('inf') 
	return np.sqrt(np.sum(np.power(np.array(vector1) - np.array(vector2) , 2)))

