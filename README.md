# Characterizing Driving Styles with Deep Learning

Here, a tensorflow implementation for <a href="https://arxiv.org/abs/1607.03611">Characterizing Driving Styles with Deep Learning</a> is provided. 

## Requirements

1. Python 2.7
2. Tensorflow 1.3.0
3. Cuda 8.0.61

## Experiments
1. Data: A sample data file for 5 drivers, each with 5 trajectories, is shared in 'data' folder. The data files has these columns: Driver, ID, Time, Lat, and Lon. 
2. Statistical Feature Matrix: In order to create the statistical feature matrix as described in the paper, you need to run 'IBM16_FeatureMatrix.py' which creates two files in data folder. 
3. CNN: In oredr to train and test the CNN-based architecture in the paper, you need to run 'IBM16_CNN.py'. This script trains and saves the best model in 'models' folder, and uses the best model for testing. 
4. RNN: In oredr to train and test the RNN-based architecture in the paper, you need to run 'IBM16_RNN.py'. This script trains and saves the best model in 'models' folder, and uses the best model for testing.

## Results

## References 
1. [Characterizing Driving Styles with Deep Learning, 2016](https://arxiv.org/pdf/1607.03611.pdf)
