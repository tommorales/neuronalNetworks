
import numpy as np

def normalizacion_0(array):
	return (array - array.min())/(array.max()-array.min())

def normalizacion_1(array, ra, rb):
	""" ??? """
	a = array.min()
	b = array.max()
	return rb + ((ra - rb)*(array - a))/(b-a)

def normalizacion_2(array, AP0=None, AP1=None):
	""" meteo:
	( AP - AP0) / (AP1 - AP0)
	AP : current air preassure
	AP0: air preassure you want to sent to 0
	AP1: air preassure you want to sent to 1
   	"""
	if AP0 is None: AP0=array.min()
	if AP1 is None: AP1=array.max()
	return (array - AP0) / (AP1 - AP0)

def normalization_3(array):
	"""Y. Le Cun (Efficient Backprop)
	C_{i} = \frac{1}{p} \sum_{i=1}^{9} (Z_{i}^{p})^{2}
	"""
	# Here ************************
	return np.sum()/len(array)






def main():
	wind = np.random.randint(10, 80, size=50)
	print wind; print "*"*43
	print normalizacion_1(wind, 5, 6)
	print ""; print "meteo ...."
	print normalizacion_2(wind)
if __name__ == "__main__":
	main()

