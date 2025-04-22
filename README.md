# Real-Time-Edge-FoG-recognition
End-to-end one-dimensional convolutional neural network optimized for real-time freezing of gait detection running on resource-limited devices 

## Train and Evaluate Script

The script "train_and_evaluate.py" trains and tests a one dimensional convolutional neural network (1D-CNN) developed for FoG detection. 
The model is schematically represented in the figure below.

<img src="1d_cnn_architecture.png" alt="Figure 1: Schematic of the multi-head convolutional neural network" width="50%">

It consists of a 1D-CNN with two consecutive convolutional layers followed by a max-pooling layer. A third convolutional block is followed by a max-pooling layer, while the last convolutional layer is connected to a global average pooling layer. The latter is fully-connected to a dense layer, followed by the single output neuron. Rectified linear unit activation function was used in all layers except for the output, where a sigmoid activation function determines the class probability. Dropout of 0.4 and l2 regularization of 0.001 were used in all layers to prevent over-fitting.
The network has 16 filters in the first two convolutional layers, and 24 filters in the last two. The kernel size is 9 and 7 in the first and second layers, and 5 in the third and forth layers.
The pool size is 4 and 3 in the first and second pooling layers, respectively.
Causal padding and glorot uniform weight initialization were used in all convolutional layers.

The input has size 120 timesteps × 3 channels, where 120 correponds to the windows size (2-second window, sampling frequency = 60 Hz) and 3 is the number of components of the 3-axis gyroscope placed on the left ankle. The output of the CNN blocks is flattened and connected to a single dense layer (48 units and a dropout rate of 0.4) and a final output layer with a sinle output corresponding to the probability of FoG. 

### Input

The script takes two CSV files as input:

- `train_data.csv` for training data
- `test_data.csv` for testing data

These files should contain a table with N samples and 4 columns. The 4 columns contain angular velocity data (`gyrX`, `gyrY`, `gyrZ`, measured in degree/s) and the FoG label (`fogLabel`). 
The number of samples depends on the amount of data, that should be sampled or resampled at 60 Hz.

#### Example Data Format

Here's an example of the expected format for the CSV files:

| gyrX   | gyrY   | gyrZ   | fogLabel |
|--------|--------|--------|----------|
| -19.34 |  -6.40 |  1.89  |        0 |
| -1.11  |  -5.38 |  -0.88 |        1 |
| -31.41 | -12.21 |  -7.82 |        0 |
|   ...  |   ...  |   ...  |      ... |

### Output

The script outputs the test label and prediction, writing to two CSV files:

- `test_label.csv`
- `test_prediction.csv`

The test label is extracted from the `test_data.csv` file. The prediction is obtained from the trained model and is in the form of probability (from 0 to 1). 
Test label and prediction can then be used for computing classification metrics. 
When evaluating test performance, remember that test data and label are here segmented with a 75% overlap (0.5s slide).

### Important Information

1. Make sure your data is sampled or resampled to 32 Hz.
2. Adjust the learning rate, number of epochs, and batch size based on your dataset size. Specifically, as your dataset size increases, reduce the learning rate and increase the number of epochs and batch size. This model configuration was trained, validated and tested on 15 hours of data from 62 subjects.
3. While the window size should be fixed at 2 seconds, you can adjust the overlap as you prefer. As the overlap increases, more windows are generated, producing more training data, which is beneficial. However, too large overlap may lead to overfitting. Thus, find the best compromise.
4. This model has shown good performance on accelerometer data recorded from the lower back. Evaluations on data recorded from other body locations produce different results.

## Citation

Please cite the following paper in your publications if this repository helps your research.

Borzì L, Sigcha L, Firouzi F, Olmo G, Demrozi F, Bacchin RA, et al. Edge-based freezing of gait recognition in Parkinson's disease. Computers and Electrical Engineering. 2025. doi:article-submitted
