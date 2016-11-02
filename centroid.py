import sys
import math
import cv2
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc

'''
USAGE: 

python centroider.py 'filename.dat' 'threshold' 'minimum area' 'maskname.png'

threshold should be between 1 and 255
minimum area will set the size of regions of noisy pixels that don't represent image components
if no mask is present, this argument will be ignored

'''

''' Declare standard settings in the matplotlib rc '''
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# Use latex for rendering math symbols
rc('text', usetex=True)

def normalize2D(array, normTo=1., makeInt=False):
	''' 
	Normalize all values in a 2D array by subtracting the minimum and dividing by the maximum
	'''
	array = np.reshape(array, (len(array)**2)) # Convert to 1D
	
	minimum = min(array)
	for i in enumerate(array):
		array[i[0]] -= minimum # baseline correction
	
	maximum = max(array)
	for i in enumerate(array):
		array[i[0]] /= maximum # normalization
		array[i[0]] *= normTo
			
	array = np.reshape(array, (int(math.sqrt(len(array))), int(math.sqrt(len(array)))))
	return array


basename = sys.argv[1][0:-4]
userThreshold = int(sys.argv[2])
areaMinimum = int(sys.argv[3])

try:
	maskName = sys.argv[4]
except:
	pass

# Extract last column from data and store in "zCount" 
zCount = np.loadtxt(basename+'.dat', delimiter=',', usecols=(2,))

# Get the dimension of the square matrix of "zCount"
zCountsDim = int(math.sqrt(len(zCount)))

# reshape zCount to 2D array of size "zDim" x "zDim"
zCount = np.reshape(zCount, (zCountsDim, zCountsDim))

# Subset data for relevant parts. This reduces unneccessary computation and takes the clean data part.
zCount = zCount[80:390, 90:400]

# Normalize data from 0 to 255
zCount = normalize2D(zCount, 255, True)

# Save monochrome image
zCount=zCount.astype(np.uint8)
cv2.imwrite(basename+'.png', zCount)

# Set mask is mask present
try: 
	mask = cv2.imread(maskName,0)
	zCount = cv2.bitwise_and(zCount,zCount, mask = mask)
	maskedFlag = True
	print('')
	print('Masking.')
	print('')
except:
	maskedFlag = False
	print('')
	print("No masking.")
	print('')



# Threshold, Contours and Centroiding
ret, thresh = cv2.threshold(zCount,userThreshold,255,0)
thresholded=np.copy(thresh)

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

drawnContours = np.copy(zCount)
cv2.drawContours(drawnContours, contours, -1, (255,0,0), 2)

centroids = []
for i in contours:
	M = cv2.moments(i)
	try:
		centroid_x = int(M['m10']/M['m00'])
		centroid_y = int(M['m01']/M['m00'])
		if M['m00'] > areaMinimum:
			centroids.append([centroid_x,centroid_y])
		
	except:
		pass

# mask zCount for a black background where pixels are 0. 
# If no mask is present no pixels will be modified.
zCount = np.ma.array (zCount, mask=0)
cmap = mpl.cm.jet
cmap.set_bad('k',1.)

# Plotting and saving figures
fig = plt.figure(facecolor='1.0')

origPlot = fig.add_subplot(221)
plt.imshow(zCount, cmap=cmap)
sign = -1

# Annotate figure with centroid positions and labels.
for i in centroids:
	cx = i[0]
	cy = i[1]
	plt.plot(cx, cy, 'k+', markersize=8)
	plt.annotate('('+str(cx)+','+str(cy)+')', (cx+20*sign, cy+20*sign), color='w')
	sign *= -1

plt.xlim(0, len(zCount))
plt.ylim(len(zCount),0)
plt.title('Linear Image', size=12)

# Mark centroid areas on figure
logPlot = fig.add_subplot(222)
plt.imshow(zCount, norm=mpl.colors.LogNorm(), interpolation='nearest',  cmap=cmap)
for i in centroids:
	cx = i[0]
	cy = i[1]
	plt.plot(cx, cy, 'k+', markersize=8)

plt.xlim(0, len(zCount))
plt.ylim(len(zCount),0)
plt.title('Logarithmic Image', size=12)

threshPlot = fig.add_subplot(223)
plt.imshow(thresholded,  cmap=cmap)
plt.title('Threshold at '+str(userThreshold), size=12)

contPlot = fig.add_subplot(224)
plt.imshow(drawnContours,  cmap=cmap)
plt.title('Contour', size=12)

if maskedFlag == True:
	basename = basename + '_MASK'

plt.tight_layout()
plt.savefig(basename+'_CENTROIDS.png')
plt.savefig(basename+'_CENTROIDS.pdf')

# Find distances between points
try:
	hypotx = abs(centroids[1][0]-centroids[0][0])
	hypoty = abs(centroids[1][1]-centroids[0][1])

	hypdist= str(int(np.round(math.hypot(hypotx, hypoty)))) ###pixel distance between centers
except:
	hypdist = 'undefined'

centroid_summary = 'centroids = '
for i in centroids:
	cx = str(i[0])
	cy = str(i[1])
	centroid_summary += cx+' '+cy+' '
centroid_summary += '\n'
centroid_summary += 'distance = '+hypdist+'\n'
centroid_summary += 'userThreshold = '+str(userThreshold)+'\n'
centroid_summary += 'area minimum = '+str(areaMinimum)

print(centroid_summary)

output = open(basename+'_CENTROIDS.txt','w')
output.write(centroid_summary)

quit()
