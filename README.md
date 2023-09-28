# Cylone_Detection_Intensity_Estimation

Developing a Convolutional Neural Network (CNN) method for estimating cyclone intensity using visible band satellite images sourced from the INSAT-3D satellite.

## Dataset

The dataset consists of the images of various cyclones (of 30-minute intervals) of the Indian peninsula and its cyclone intensities estimated at every 6hr interval. The collection is of IR images that are suitable for collecting images of cyclones even at night. The missing cyclone intensity values are imputed by time-series data analysis.  The dataset is collected from the Meteorological and oceanographic Satellite data archive center (MOSDAC) of ISRO. The dataset consists of 1032 images. 

The sample images of dataset used can be found at: https://shrtco.de/Lxm8WD

Visit MOSDAC at: https://www.mosdac.gov.in/


## Methodology

The approach is divided into 3 parts:

i.   Detect the cyclone from the satellite image using the YOLOv5 object detection model.

ii.  Extract the Region of interest (ROI) from the detected cyclone.

iii.  Calculate the intensity of the ROI image using the VGG16 and InceptionV3 model. 

<img src="https://github.com/manchalaharikesh/Cylone_Detection_Intensity_Estimation/blob/main/static/Methodoly_Cyclone_Estimation.jpg?raw=true">


## Results

After training the model using TensorFlow we have achieved the results:

**Results of VGG-16**
1. Mean Absolute Error(MAE): 0.103
2. Root Mean Absolute Error(RMSE): 0.32
3. R2 Score: 0.74

**Results of Inception-V3**
1. Mean Absolute Error(MAE): 0.25
2. Root Mean Absolute Error(RMSE): 0.503
3. R2 Score: 0.76


## Further Scope
Leveraging Brightness Temperature (BT) to extract crucial information about water temperature from satellite imagery, thereby enhancing the precision of cyclone intensity estimation models.
