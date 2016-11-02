# velocity-map-image-processing
A velocity map imaging processing suite for automated centroiding, masking, and multipeak fitting with numerical recipies and computer vision libraries implimented with python using numpy and opencv.

## Overview

**velocity-map-image-processing** is designed to process 2D imaging data recorded from a velocity-map imaging apparatus where two distinct velocity components are present. However, this could be adpated/applied to any images where two components are present.

### Requisites

* Python 2+ (test on 2.7.6 MSC v.1500 32 bit (Intel))
 * opencv (2.4.13)
 * numpy (1.11.2)
 * matplotlib (2.0.0b4)

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

velocity-map-image-processing is executed with two scritps ([centroid.py](centroid.py), [slice_fit.py](slice_fit.py)) from the command line with arguments specified by the user.

```
# centroider.py data_path threshold minimum_area mask_name 

```

This generates three output files

Using the provided sample [data](2SCCMAr_98SCCMHe_T21p4mV_16p0eV_Pres1p4-4_PresM2p6-6.dat), `2SCCMAr_98SCCMHe_T21p4mV_16p0eV_Pres1p4-4_PresM2p6-6.dat`, we begin with Step 1. From the command line, navigate to `centroid.py` and execute the following.


### Introduction

In my collaborative research at Berkeley Labs, we use a silicon carbide (SiC) thermal reactor nozzle to probe chemistry at high temperatures (25 - 1500C). One of the problems we faced in these studies was determing how close the temperature of the reactor (measured with a thermocouple) is to the temperature of the gas inside the reactor. By using a technique known as velocity map imaging we can 'ionize' the hot gas and image the ions as a function of their velocity. Since temperature can be considered to be a measure of average kinetic energy (velocity) we can use this technique to determine the average temperature of the gas. The experimental schematic is depicted below in figure 1.

