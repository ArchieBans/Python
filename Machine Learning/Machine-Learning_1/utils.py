# import libraries here
import cv2
import numpy
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import skimage.color

def read_image(image_path,*args,**kwargs):
	image = cv2.imread('input.jpg',1)
	rbg_correct = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	return rbg_correct

def invert_image(image,*args,**kwargs):
	white = numpy.ones([640,480,3],dtype=numpy.uint8)*255	
	new_image = numpy.subtract(white,image)	
	return new_image


def save_image_to_h5(image, h5_path, *args, **kwargs):
	h_variable = h5py.File(h5_path,'w')
	h_variable.create_dataset('dataset_1',data=image)
	h_variable.close()


def read_image_from_h5(h5_path, *args, **kwargs):
	h_variable = h5py.File(h5_path,'r')
	image = h_variable['dataset_1'][:]
	h_variable.close()
	return image
def gray_scale_image(image, *args, **kwargs):
	# x,y,z = image.shape
	# image[:] = image.mean(axis=-1,keepdims=1)
	image = skimage.color.rgb2gray(image)
	return image

def find_difference_of_gaussian_blur(image, k1, k2, *args, **kwargs):
	gray_scale = gray_scale_image(image)

	# plt.figure()
	# plt.imshow(gray_scale)
	# plt.show()

	blur_1 = gaussian_filter(gray_scale,sigma=k1,mode='reflect')
	blur_2 = gaussian_filter(gray_scale,sigma=k2,mode='reflect')
	difference = blur_2 - blur_1
	difference = difference.astype('float32')
	difference /= 255.0
	return difference

def keep_top_percentile(image, percentile, *args, **kwargs):
	"""Find the difference of two Gaussian blurs from an image.

	Parameters
	----------
	image : ndarray
	Original image.

	percentile : scalar
	Top percentile pixels will be kept.

	Returns
	-------
	thresholded : ndarray
	Image with the high value pixles.
	"""
	# TODO: Implement the method
	sorted_array = -numpy.sort(-image)
	value = sorted_array.shape
	value_colum = int((percentile/100)*value[0])
	value_row = int((percentile/100)*value[1])
	copy_array = numpy.zeros((value_colum,value_row))	
	for x in range(0,value_colum):
		for y in range (0,value_row):
			copy_array[x][y] = sorted_array[x][y]
	threshold = numpy.min(copy_array)	
	image[image<(threshold)] = 0
	thresholded = image	
	return thresholded
