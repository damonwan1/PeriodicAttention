
# Periodic Attention Mechanism for Multivariate Time Series Forecasting

Welcome to the official implementation of the **Transformer with Learnable Period Detection and Periodic Attention** for multivariate time series (MTS) forecasting.

### Key Features

- **Channel-specific Period Detection**: We introduce a wavelet-based method that decomposes each channel’s time series into its high and low-frequency components, allowing the model to capture the dominant period of each variable.
  
- **Periodic Attention Decay**: This mechanism aligns attention distribution with periodic structures in the data, enhancing the model's ability to focus on key time intervals where periodic features are prominent.
  
- **State-of-the-Art Performance**: Our model achieves superior results on several benchmark datasets, including electricity, weather, and traffic data, outperforming leading models like TimesNet and SDformer.

## Installation

To use this repository, first clone it to your local machine:

```bash
git clone https://github.com/damonwan1/PeriodicAttention.git
cd PeriodicAttention
```

Then, install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

To run the model, ensure you have the correct format of multivariate time series data. The data should be structured such that each column represents a different time series variable, and each row corresponds to a time point.

You can use the following command to preprocess your data:

```bash
python preprocess.py --data_path [your_data_path]
```

### Training

To train the model on your dataset, use the following command:

```bash
python train.py --config configs/config.yaml
```

You can adjust the parameters such as learning rate, batch size, and number of epochs in the `config.yaml` file.

## Datasets

We validate our model using the following publicly available datasets:

1. **Electricity**: Consists of power consumption data.
2. **Weather**: Meteorological data with temperature, humidity, and other variables.
3. **PEMS**: Traffic flow data.
4. **ETT**: Electric transformer temperature data.

You can download the datasets from their respective sources or use the preprocessed versions provided in the `data` folder.

## Acknowledgements

We would like to thank all the contributors and collaborators for their valuable input and support during the development of this project.
