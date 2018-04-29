# Characterizing Driving Styles with Deep Learning

Here, a tensorflow implementation for <a href="https://arxiv.org/abs/1607.03611">Characterizing Driving Styles with Deep Learning</a> is provided. 

## Requirements

1. Python 2.7
2. Tensorflow 1.3.0
3. Cuda 8.0.61

## Data

A sample data file for 5 drivers, with 5 trajectories for each, is shared in 'data' folder. The data file has following columns: Driver, ID, Time, Lat, and Lon. 

## Experiments

1. Statistical Feature Matrix: In order to create the statistical feature matrix as described in the paper, you need to run 'IBM16_FeatureMatrix.py' which creates two files in data folder. 
2. CNN: In oredr to train and test the CNN-based architecture in the paper, you need to run 'IBM16_CNN.py'. This script trains and saves the best model in 'models' folder, and uses the best model for testing. 
3. RNN: In oredr to train and test the RNN-based architecture in the paper, you need to run 'IBM16_RNN.py'. This script trains and saves the best model in 'models' folder, and uses the best model for testing.

## Results

Our best results for driver classification task based on a real-world, private, and non-anonymized (gps coordinates) dataset of 50 drivers with 200 trajectories for each is as follows (prediction is on segment-level, and not the trip-level):

| Model | #Drivers | #Trajectories | Test Accuracy | Note |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| CNN  | 50  | 200  | 16.4%  | Using Momentum Nestrov Optimizer; Momentum=0.9 |
| RNN  | 50  | 200  | 25.0%  | LSTM cells, 2 Layers; Dropout on Second layer (0.6), Batch 256 |

Note that here we used LSTM cells, instead of using RNN cells with identity matrix for recurrent weight initialization, as such thing is not available in Tensorflow currently. However, as noted by <a href="https://arxiv.org/abs/1504.00941">Le et al.</a>, the identity-matrix initialization of recurrent weights provides comaprable results to LSTM cells. 

## References 

1. [Characterizing Driving Styles with Deep Learning, 2016](https://arxiv.org/pdf/1607.03611.pdf)
