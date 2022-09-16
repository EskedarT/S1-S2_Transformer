# S1-S2_Transformer

##Introduction

This repository prototypes the fusion of Sentinel-1 SAR and Sentinel-2 optical data time-series for mapping tropical dry-forest. Since seasonality is the main challenge to map tropical dry-forests, we learn the seasonality in the data implicitly using transformers. The implementation uses Google Earth Engine (GEE) to prepare Sentinel-1 and Sentinel-2 time series and Tensorflow is used for deep learning. 

##Architecture

The architecture used in this implementation is a siamese arechitecture with transformers used for each branch separately. It requires the data to be prepared as tables of time series values for different spatial locations. The reference data should be prepared as sparse point feature collection as a GEE asset.
