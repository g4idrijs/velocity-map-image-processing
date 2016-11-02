import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from scipy.special import erf 
import scipy, pylab
# from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy import optimize
# from thermocouples_reference import thermocouples
from matplotlib import rc

'''
USAGE: 

python slice_fit.py 'filename.dat' centroid_1x centroid_1y centroid_2x centroid_2y

Centroids are found from the output of centroid.py
'''

''' Declare standard settings in the matplotlib rc '''
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

'''
----------------------------------
			FUNCTIONS
----------------------------------
'''


def gaussian(x, pars):
    '''
    Generate gaussian from x values and parameters
    '''   
    a1 = pars[0]  # Peak height
    a2 = pars[1]  # Central value 
    a3 = pars[2]  # Standard deviation

    f = a1*np.exp(-1*(x-a2)**2/(2*a3**2))
    return f

def two_peaks(x, *pars):    
    '''
    Generate sum of two gaussians from x values and parameters
    '''
    a11 = pars[0]  # Peak height
    a12 = pars[1]  # Central value 
    a13 = pars[2]  # Standard deviation
    a21 = pars[3]  # Peak height
    a22 = pars[4]  # Central value 
    a23 = pars[5]  # Standard deviation 
    p1 = gaussian(x, [a11, a12, a13])
    p2 = gaussian(x, [a21, a22, a23])
    return p1 + p2


def listArithmatic(list=[[],[]], xDelta=1, yDelta=1, axis=0):
	'''
	Increase every list element by "Delta"
	'''
	for i in enumerate(list[0]):
		for j in enumerate(list[1]):
			list[i[0]][j[0]] += delta

	return list

def movePoints(points=[[],[]], xDelta=1, yDelta=1, axis='x'):
	for i, [element0, element1] in enumerate(points):
		if axis == 'x' or axis == 'xy':
			points[i][1] += xDelta
		if axis == 'y' or axis == 'xy':
			points[i][0] += yDelta
		else:
			pass

	return points


def floatLols(ListofLists=[[],[]]):
	'''
	Convert the integer values in a list of lists into floating point values
	'''
	for i in enumerate(ListofLists):
		(i[1][0], i[1][1]) = (float(i[1][0]), float(i[1][1]))
	return ListofLists


def lineEquation(x, pointPair):
	'''
	Determines the equation of the line between the two points listed in pointPair"
	'''
	floatLols(pointPair)
	m = (((pointPair[1][1]-pointPair[0][1])/(pointPair[1][0]-pointPair[0][0])))
	y = (m * (x - pointPair[0][0])) + pointPair[0][1]

	return y

def lineEquation2(inCoord, ((x1,y1),(x2,y2))):
	'''
	Calculate the equation of the line between two points
	'''
	dx = float(x2 - x1)
	dy = float(y2 - y1)
	# check for divide by zero/vertical line
	if dx == 0:
		outCoord = x1
				
		return (int(round(outCoord)), int(round(inCoord)))
	else:
		m = dy/dx
		b = y1-m*x1
				
		if abs(dx) > abs(dy):
		 	outCoord = m*inCoord + b
		 	return (int(round(inCoord)), int(round(outCoord)))
		else:
		 	outCoord = (inCoord-b)/m
		 	print inCoord, outCoord
		 	return (int(round(outCoord)), int(round(inCoord)))

def sliceMatrix2(matrix, pixelCoords):
	'''
	Extract a value of the matrix at pixelCoord
	'''
	matrixSlice=[]
	for i, j in pixelCoords:
		# if index has negative component return a zero
		if i >= 0:
			matrixSlice.append(matrix[j][i])
		else:
			matrixSlice.append(None)

	return matrixSlice

def sliceMatrix(array, pointPair):
	'''
	Extracts the data along the line derived from "lineEquation"
	'''
	sliceArray = []
	for x in range(0, array.shape[1], 1):
		y = int(round(lineEquation(x, pointPair)))
		sliceArray.append(array[abs(y)][abs(x)])

	return sliceArray

def normalize2D(array):
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

	array = np.reshape(array, (math.sqrt(len(array)), math.sqrt(len(array)))) # Convert to 2D

	return array

def get_filenameConstants(string):
	'''
	Convert the elements in the filename to useful numbers.
	'''
	basename=string[0:-4]
	stringList=basename.split('_')
	stringDict={}
	typeC = thermocouples['C']
	stringDict['Gas'] = stringList[0].replace('SCCM','SCCM ')
	stringDict['Noble Gas'] = stringList[1].replace('SCCM','SCCM ')
	stringDict['Kelvin'] = str(int(round(typeC.inverse_KmV(float(stringList[2].replace('p','.').replace('mV','').replace('T','')), Tref=296))))+'\,K'
	stringDict['eV'] = stringList[3].replace('p','.').replace('eV',' eV')
	stringDict['Source'] = stringList[4].replace('Pres','').replace('p','.').replace('-','e-')
	stringDict['Main'] = stringList[5].replace('PresM','').replace('p','.').replace('-','e-')

	# for i, element in enumerate(stringList):
	# 	stringList[i] = stringList[i].replace('Pres','').replace('SCCM','').replace('p','.').replace('T','').replace('mV','').replace('eV','').replace('M','').replace('-','e-')

	return stringDict


'''
----------------------------------
			MAIN
----------------------------------
'''

# This section was used to programatically label the plots according the the conditions in the filename. It is not useful for the sample data
# constants = get_filenameConstants(sys.argv[1])

# Convert mV from filename to Kelvin
# typeC = thermocouples['C']
# try:
#     # T_mV = float(0.0)
#     T_mV = float(sys.argv[1][sys.argv[1].find('mV')-5:sys.argv[1].find('mV')].replace('p','.').replace('T','').replace('_',''))
# except: 
#     T_mV = float(0.0)

# T_K=str(int(round(typeC.inverse_KmV(T_mV, Tref=296))))+' K'


# Set command line arguments to variables
try:
	basename = sys.argv[1][0:-4]
except:
	basename = '15SCCM_O2_22p23eV_4BDAs' # default to that file

# Specify centroids N.B. write from sys.argv later

userCentroids = ((float(sys.argv[2]),float(sys.argv[3])), (float(sys.argv[4]),float(sys.argv[5])))
x1 = float(sys.argv[2])
y1 = float(sys.argv[3])
x2 = float(sys.argv[4])
y2 = float(sys.argv[5])


# Extract last column from data and store in "zCount" 
zCount = np.loadtxt(basename+'.dat', delimiter=',', usecols=(2,))

# Get the dimension of the square matrix of "zCount"
zCountsDim = int(math.sqrt(len(zCount)))

# reshape zCount to 2D array of size "zDim" x "zDim"
zCount = np.reshape(zCount, (zCountsDim, zCountsDim))

zCount = zCount[80:390, 90:400]

# Subsection of whole data set reflecting centroided data positions '''

centroids=userCentroids
# manually move the centroids. Used for debugging.
# moveTroids = [[centroids[0][0],centroids[0][1]],[centroids[1][0],centroids[1][1]]]
# centroids = movePoints(moveTroids, 100, 80, 'xy')

# Normalize zCount values
zCount = normalize2D(zCount)

# Extract line along centroids
xPixel = np.arange(0, len(zCount), 1.)
pixelCoords=[]
yPixel = []
for i in xPixel:
	Coords = lineEquation2(i, centroids)
	pixelCoords.append(Coords)
	yPixel.append(Coords[1])

zCountSlice = sliceMatrix2(zCount, pixelCoords)


# Mask array to make 0 values black

masked_array = np.ma.array (zCount, mask=0)
cmap = mpl.cm.jet
cmap.set_bad('k',1.)



''' FITTING '''

guessParameters = np.asarray((zCount[y1][x1], x1, 10., zCount[y2][x2], x2, 10.))
popt, pcov = curve_fit(two_peaks, xPixel, zCountSlice, guessParameters, maxfev=1000000000)

gFitSum=two_peaks(xPixel, *popt)

zCount_residual = zCountSlice - gFitSum


''' Take parameters from fit for each component gaussian '''
pars1 = popt[0:3]
pars2 = popt[3:6]
a1 = str(popt[0])
b1 = str(popt[1])
c1 = str(popt[2])
a1 = str(popt[3])
b1 = str(popt[4])
c1 = str(popt[5])
''' Generate gaussian fits from component gaussians '''
gFit1 = gaussian(xPixel, pars1)
gFit2 = gaussian(xPixel, pars2)
''' Calculate area under gaussians '''
area1 = str(np.trapz(gFit1, xPixel))
area2 = str(np.trapz(gFit2, xPixel))


''' Calculate chi squared and update covariance matrix  '''
dof = len(xPixel) - len(popt)
chisq = sum((zCountSlice-gFitSum)**2)
cov = pcov#*dof/chisq



## Output results
print "******** RESULTS FROM FIT ******** (slice_fit.py)"

print "\nNumber of Data Points = %7g, Number of Parameters = %1g"\
      %(len(xPixel), len(popt) )

print "\nCovariance Matrix : \n", cov, "\n"
try:
    # Calculate Chi-squared
    chisq = sum((zCountSlice-gFitSum)**2)
    # WARNING : Scipy seems to use non-standard poorly documented notation for cov,
    #   which misleads many people. See "cov_x" on
    #   http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq
    #   (which underlies curve_fit) and also discussion at
    #   http://stackoverflow.com/questions/14854339/in-scipy-how-and-why-does-curve-fit-calculate-the-covariance-of-the-parameter-es.
    #   I think this agrees with @cdeil at http://nbviewer.ipython.org/5014170/.
    #   THANKS to Wes Watters <wwatters@wellesley.edu> for pointing this out to me (16 September 2013)
    #
    # Convert Scipy cov matrix to standard covariance matrix.
    # cov = pcov*dof/chisq
    print "Correlation Matrix :"
    for i,row in enumerate(cov):
        for j in range(len(popt)) :
            print "%10f"%(cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])),
                # Note: comma at end of print statement suppresses new line
        print
    print "\nEstimated parameters and uncertainties (with initial guesses)"
#  Note: If the fit is poor, i.e. chisq/dof is large, the uncertainties
#   are scaled up. If the fit is too good, i.e. chisq/dof << 1, it suggests
#   that the uncertainties have been overestimated, but the uncertainties
#   are not scaled down.
    for i in range(len(popt)) :
        print ("   p[%d] = %10.5f +/- %10.5f      (%10.5f)"
                   %(i,popt[i],cov[i,i]**0.5*max(1,np.sqrt(chisq/dof)),
                       guessParameters[i]))

    cdf = scipy.special.chdtrc(dof,chisq)
    print( "\nChi-Squared/dof = %10.5f, CDF = %10.5f%%")\
        %(chisq/dof, 100.*cdf)
    if cdf < 0.05 :
        print( "\nNOTE: This does not appear to be a great fit, so the")
        print( "      parameter uncertainties may underestimated.")
    elif cdf > 0.95 :
        print( "\nNOTE: This fit seems better than expected, so the")
        print( "      data uncertainties may have been overestimated.")


# If cov has not been calculated because of a bad fit, the above block
#   will cause a python TypeError which is caught by this try-except structure.
except TypeError:
    print( "**** BAD FIT ****")
    print( "Parameters were: ",popt)
    # Calculate Chi-squared for current parameters
    chisq = sum((zCountSlice-gFitSum)**2)
    print( "Chi-Squared/dof for these parameter values = %10.5f, CDF = %10.5f%%")\
        %(chisq/dof, 100.*float(scipy.special.chdtrc(dof,chisq)))
    print( "Uncertainties not calculated.")
    print()
    print( "Try a different initial guess for the fit parameters.")
    print ("Or if these parameters appear close to a good fit, try giving")
    print( "    the fitting program more time by increasing the value of maxfev.")
    chisq = None


''' Write Data to txt file '''
np.savetxt(basename+'_FITS.txt', (np.column_stack((xPixel, zCountSlice, gFit1, gFit2, gFitSum, zCount_residual))), delimiter=',',  header='Pixel, zCount, Fit1, Fit2, FitSum, Residual', comments='', newline='\n')

''' title for figures '''
figTitle= basename.replace('_','\_')


fig = plt.figure(facecolor='1.0')
fig.suptitle(figTitle)
imagePlot = fig.add_subplot(111)
imagePlot.imshow(zCount, norm=mpl.colors.LogNorm(), interpolation='nearest', cmap=cmap)
plt.plot(xPixel, yPixel, 'w--')
plt.plot([centroids[0][0],centroids[1][0]], [centroids[0][1],centroids[1][1]], 'wo', markersize=5)
plt.axis('image')
# plt.colorbar()
plt.ylabel('pixel y')
plt.xlabel('pixel x')
# plt.title('Line through centroids - O'r'$_2$'+' at '+T_K)

# plt.xlim([80,400])
# plt.ylim([400,80])


plt.savefig(basename+'_LINE.png')
plt.savefig(basename+'_LINE.pdf')


# plt.show()

''' Plot data, fits, and residuals '''
# fig = plt.figure(facecolor="1.0")
fig2 = plt.figure(facecolor='1.0')
fit = fig2.add_subplot(311)
# fit = plt.subplot2grid((3,3), (0,0), colspan=3, rowspan=2)

# remove tick labels from upper plot (for clean look)
# fit.set_xticklabels( () )

plt.ylabel("Normalized Count")
fig2.suptitle(figTitle, size=16)
''' plt.subplots_adjust(top=0.8) '''

# plt.xlabel("Pixel")
# Plot data as red circles, and fitted function as (default) line
#   (The sort is in case the x data are not in sequential order.)
fit.plot(xPixel,zCountSlice, 'k', linewidth=0, markersize=6, marker='o', mfc='w')
fit.plot(xPixel, gFitSum, 'c-', linewidth=1)#, alpha=0.7)
fit.plot(xPixel, gFit1, 'r--', linewidth=1)
fit.plot(xPixel, gFit2, 'b--', linewidth=1)#, alpha = 0.7)
fit.set_xticklabels( () )

fit.set_xlim([0,300])

fit.legend(['Data', 'Fit Sum', 'Fit 1', 'Fit 2'], fontsize=12)





residuals = fig2.add_subplot(312) # 3 rows, 1 column, subplot 2
residuals.errorbar(xPixel, zCount_residual, fmt='k+', label="Residuals")
# make sure residual plot has same x axis as fit plot
residuals.set_xlim(fit.get_xlim())
residuals.axhline(y=0, color='r') # draw horizontal line at 0 on vertical axis
# Label axes
plt.xlabel("Pixel")

# residuals.set_xlim([140,300])
# residuals.set_xticks(scipy.arange(150,300, 20), minor=True)
# residuals.set_yticks(np.arange(-0.06, 0.12, 0.08))

try:
    plt.ticklabel_format(style='plain', useOffset=False, axis='x')
except: pass

plt.tight_layout

plt.figtext(0.1, 0.2, r'$f(x) = a exp \left(- \frac{(x - b)}{2c^2} \right) $', size=20)
plt.figtext(0.1, 0.15, r'$\chi^2 =$ '+str(chisq), size=14)
plt.figtext(0.1, 0.1, r'$DoF =$ '+str(dof), size=14)

# for i, element in enumerate(popt):
# 	popt[i] = round(element, 5)

# for i, element in enumerate(cov):
# 	print i, element
	# cov[i] = round(element, 5)
# plt.figtext(0.05,0.25,"Converged with ChiSq = " + str(chisq) + ", DOF = " +
#         str(dof) + ", CDF = " + str(100*scipy.special.chdtrc(dof,chisq))+"%")
labels = ('a$_1$', 'b$_1$', 'c$_1$', 'a$_2$', 'b$_2$', 'c$_2$')
for i, value in enumerate(popt):
	step = i*0.03
	plt.figtext(0.55,0.22-step, labels[i] + " = "+str(round(popt[i],8)).ljust(18))
	plt.figtext(0.75, 0.225-step, r'$\pm$ '+ str(round(np.sqrt(cov[i,i]),8)))
plt.savefig(basename+'_FITS.png')
plt.savefig(basename+'_FITS.pdf')

# plt.show()



quit()

