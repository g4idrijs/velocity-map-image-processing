# velocity-map-image-processing
A velocity map imaging processing suite for automated centroiding, masking, multipeak fitting, and visualization with numerical and computer vision libraries.

## Overview

**velocity-map-image-processing** is designed to process 2D imaging data recorded from a velocity-map imaging apparatus where two distinct velocity components are present. However, this could be adpated/applied to any images where two components are present.

### Requisites

* Python 2+ (test on 2.7.6 MSC v.1500 32 bit (Intel))
 * opencv (2.4.13)
 * numpy (1.11.2)
 * matplotlib (2.0.0b4)
 * scipy (

It was created for use with python 2.7.6 on the Windows command line. While not tested on a Posix system, it should theoretically work. Image data is processed according to the following steps.

* Part 1: Centroid (`centroid.py`)
 * read, reshape, and subset 1D data column into 2D numpy data
 * mask data (if mask supplied)
 * threshold image to user value
 * find contours
 * find moments (centroids)
 * plot centroid/mask visualizations
 * output centroids
 
* Part 2: Slicing & Fitting (`slice_fit.py`)
 * extract data slice along line defined by two centroids
 * fit two gaussians to data slice using centroids as guess
 * plot slice onto data
 * plot fits and parameters
 * output fit parameters

### Quick Start Usage

velocity-map-image-processing is executed with two scritps ([centroid.py](centroid.py), [slice_fit.py](slice_fit.py)). These are called from the command line with user-specified arguments.

```
python centroid.py \path\to\data threshold minimum_area mask_name 

python slice_fit.py \path\to\data centroid_1_x centroid_1_y centroid_2_x centroid_2_y

```

Using the provided [sample data](2SCCMAr_98SCCMHe_T21p4mV_16p0eV_Pres1p4-4_PresM2p6-6.dat), `2SCCMAr_98SCCMHe_T21p4mV_16p0eV_Pres1p4-4_PresM2p6-6.dat`. Here we don't supply a mask. See a more detailed discussion below for masking.

```
C:\path\to\script> python centroid.py sample.dat 20 30 

No Masking.

centroids = 78 156 148 162
distance = 70
userThreshold = 20
area minimum = 30
```

This will generate 4 files:
* grey scale image of the subset data
* .png indicating the centroids, thresholded image, and contour regions
* .pdf of the above
* text file containg the output

Now using those centroid points we can fit the data along the slice between the two centroid points.

[img!]
```
python slice_fit.py sample.dat 78 156 148 162

```


### Introduction

In my collaborative research at Berkeley Labs, we use a silicon carbide (SiC) thermal reactor nozzle to probe chemistry at high temperatures (25 - 1500C). One of the problems we faced in these studies was determing how close the temperature of the reactor (measured with a thermocouple) is to the temperature of the gas inside the reactor. By using a technique known as velocity map imaging we can 'ionize' the hot gas and image the ions as a function of their velocity. Since temperature can be considered to be a measure of average kinetic energy (velocity) we can use this technique to determine the average temperature of the gas. The experimental schematic is depicted below in figure 1.

