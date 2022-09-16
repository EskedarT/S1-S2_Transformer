# S1-S2-Transformer

##Introduction

This repository prototypes the fusion of Sentinel-1 SAR and Sentinel-2 optical data time-series for mapping tropical dry-forest. Since seasonality is the main challenge to map tropical dry-forests, we learn the seasonality in the data implicitly using transformers. The implementation uses Google Earth Engine (GEE) to prepare Sentinel-1 and Sentinel-2 time series and Tensorflow is used for deep learning. 

##Architecture

The architecture used in this implementation is a siamese arechitecture with transformers used for each branch separately. It requires the data to be prepared as tables of time series values for different spatial locations. The reference data should be prepared as sparse point feature collection as a GEE asset.

![mhsa_lps2022_1](https://user-images.githubusercontent.com/48068921/190612508-0843559a-3107-4c19-a006-e1d4206f6413.png)

##Usage

To train a model, the user needs to provide an area of interest in GEE geometry format and run the prepare_data.py first to prepare the training datasets. It is assumed the user has access to a Google Cloud Storage buckets. Time series smoothing using a moving median filter is implemented to smoothen both Sentinel-1 and Sentinel-2 time series. The user can select the interval in the params dictionary. 

The script is provided in a jupyter notebook format to avoid errors when running the script. This should make it easier for users to run the code in Google colab without worrying about software dependencies.

#Dependencies

The scripts are written in Tensorflow 2.8 so there may be issues with earlier versions of Tensorflow. There maybe issues when using later versions of tensorflow.

## Acknowledgment
Some functions were adopted from Google Earth Engine example workflow [page](https://developers.google.com/earth-engine/guides/tf_examples).
