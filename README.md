# velocity-map-image-processing
A velocity map imaging processing suite for automated centroiding, masking, multipeak fitting, and visualization with numerical and computer vision libraries.

## Overview

**velocity-map-image-processing** is designed to process 2D imaging data recorded from a velocity-map imaging apparatus where two distinct velocity components are present. However, this could be adpated/applied to any image data where two components are present. For a description of its implimentation for this research, see the motivation section at the end.
These scripts were created for use with python 2.7.6 on the Windows command line. While not tested on a Posix system, they should theoretically work. Image data is processed according to the following steps.

* Part 1: Centroid (`centroid.py`)
 * read, reshape, and subset 1D data column into 2D numpy data
 * mask data (if mask supplied)
 * threshold
 * find contours and moments (centroids)
 * plot/save centroid/mask visualizations
 
* Part 2: Slicing & Fitting (`slice_fit.py`)
 * extract data slice along line defined by two centroids
 * fit two gaussians to data slice using centroids as guess
 * plot/save slice, fits, and parameters

## Dependencies

* Python 2+ (tested on 2.7.6 MSC v.1500 32 bit (Intel))
 * opencv (2.4.13)
 * numpy (1.11. 2)
 * matplotlib (2.0.0b4)
 * scipy (0.18.1)
 
## Usage

velocity-map-image-processing is executed with two scritps ([centroid.py](centroid.py), [slice_fit.py](slice_fit.py)). These are called from the command line with user-specified arguments.

```
python centroid.py \path\to\data threshold minimum_area mask_name 

python slice_fit.py \path\to\data centroid_1_x centroid_1_y centroid_2_x centroid_2_y
```

### Centroiding Without Masking

Using this script with the provided [sample.dat](sample.dat).

```
C:\path\to\script> python centroid.py sample.dat 20 30 

No Masking.

centroids = 78 156 148 162
distance = 70
userThreshold = 20
area minimum = 30
```

This will generate 4 files:
* .png and .pdf indicating the centroids, thresholded image, and contour regions

![centroid threshold contour](./images/sample_CENTROIDS.png)

* grey scale .png of the subset data

![grey scale subset](./images/sample.png)

* text file containg the centroids and parameters, [sample_CENTROIDS.txt](sample_CENTROIDS.txt).

### Slice Through Centroid and Multipeak Fitting

Now using those centroid points from the output of `centroid.py` we can fit the data along the slice between the two centroid points.

```
C:\path\to\script> python slice_fit.py sample.dat 78 156 148 162
```
This generates 5 files:
* an .png visualizing the line of data sliced through the centroids

![data slice](./images/sample_LINE.png)

* .png and .pdf demonstrating the multi-peak gaussian fitting along the data slice

![fits](./images/sample_FITS.png)

* text file containg the data slice, fits, fits sum, residuals

[sample_CENTROIDS.txt](sample_FITS.txt)

### Centroiding With Masking

When thresholding cannot separate the components (such as when the image components are overlapped) the centroiding fails to find the centers. To solve this issue we can introduce image masking in order to isolate the highest intensity centers of each component. Masking is achieved by providing a binary image mask that will isolate the deisred components.

Using [sample_mask.dat](sample_mask.dat). Without masking we find the [centroid.py](centroid.py) routine fails where a low threshold (20) convolves both components and a higher threshold segments the lower intensity component and fails to centroid as demonstrated below.

![centroid fail](./images/centroid_fail.png)

By providing a unique mask for each component we can find their centers separately. Binary image masks can be made with any basic image software such as paint whereby black parts of the image will be masked (ignored). Masks should be the same size as the image data. We can process these data with the two masks pictured below.

![binary masks](./images/masks.png)

Running [centroid.py](centroid.py) independently with each mask enables us to determine each centroid for fitting. First with [outer mask.png](outer_mask.png).

```
C:\path\to\script> python centroid.py sample.dat 20 30 mask_outer.png

Masking.

centroids = 109 157
distance = undefined
userThreshold = 50
area minimum = 30
```
![outer_mask](./images/outer_mask.png)

The same treatment with [inner_mask.png](inner_mask.png) determines the inner-component centroid.

![inner_mask](./images/inner_mask.png)

We can now run the [slice_fit.py](slice_fit.py) with these centroids to achieve the fit below.

![slice fit mask](./images/sample_mask_FITS.png)
### Introduction

In my collaborative research at Berkeley Labs, we use a silicon carbide (SiC) thermal reactor nozzle to probe chemistry at high temperatures (25 - 1500C). One of the problems we faced in these studies was determing how close the temperature of the reactor (measured with a thermocouple) is to the temperature of the gas inside the reactor. By using a technique known as velocity map imaging we can 'ionize' the hot gas and image the ions as a function of their velocity. Since temperature can be considered to be a measure of average kinetic energy (velocity) we can use this technique to determine the average temperature of the gas. The experimental schematic is depicted below in figure 1.

