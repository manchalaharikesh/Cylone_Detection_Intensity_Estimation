# Cylone_Detection_Intensity_Estimation

Developing a Convolutional Neural Network (CNN) method for estimating cyclone intensity using visible band satellite images sourced from the INSAT-3D satellite.

## Dataset

A dataset comprising 1032 images at an interval of 30 minutes  was obtained from the MOSDAC website, an archive of cyclone imagery captured in the visible band, showcasing varying intensities. These images were organized and matched with corresponding cyclone intensity values for the purpose of estimation.

## Methodology

The approach is divided into 3 parts:

i.   Detect the cyclone from the satellite image using the YOLOv5 object detection model.

ii.  Extract the Region of interest (ROI) from the detected cyclone.

iii.  Calculate the intensity of the ROI image using the VGG16 and InceptionV3 model. 


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
