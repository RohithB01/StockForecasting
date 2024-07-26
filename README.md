# Stock Price Forecasting using ARIMA in Python

This repository demonstrates how to forecast stock prices using the ARIMA (AutoRegressive Integrated Moving Average) model in Python. The stock market data of HCL from April 2017 to April 2022 is used as a case study to predict the future stock prices.

## Table of Contents
- [Introduction](#introduction)
- [Data Information](#data-information)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Further Reading](#further-reading)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The stock market is highly volatile and unpredictable, making it challenging to forecast stock prices accurately. In this project, we use the ARIMA model, a class of statistical models for analyzing and forecasting time series data, to predict future stock prices.

## Data Information
The dataset used in this project consists of stock market data for HCL from 2017-04-21 to 2022-04-21. The dataset includes various attributes such as Open, High, Low, Close, Adj Close, and Volume. The 'Adj Close' column is used for predicting the future stock prices.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- statsmodels
- pmdarima

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StockPriceForecasting_ARIMA.git
Certainly! Below is a suggested README file for your GitHub repository on forecasting stock prices using ARIMA in Python. I've also included some potential names for your repository.

### Repository Name Suggestions:
1. `StockPriceForecasting_ARIMA`
2. `ARIMA_StockPrediction`
3. `StockPricePredictor`
4. `StockForecasting`
5. `ARIMA_StockAnalysis`

### README.md

```markdown
# Stock Price Forecasting using ARIMA in Python

This repository demonstrates how to forecast stock prices using the ARIMA (AutoRegressive Integrated Moving Average) model in Python. The stock market data of HCL from April 2017 to April 2022 is used as a case study to predict the future stock prices.

## Table of Contents
- [Introduction](#introduction)
- [Data Information](#data-information)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Further Reading](#further-reading)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The stock market is highly volatile and unpredictable, making it challenging to forecast stock prices accurately. In this project, we use the ARIMA model, a class of statistical models for analyzing and forecasting time series data, to predict future stock prices.

## Data Information
The dataset used in this project consists of stock market data for HCL from 2017-04-21 to 2022-04-21. The dataset includes various attributes such as Open, High, Low, Close, Adj Close, and Volume. The 'Adj Close' column is used for predicting the future stock prices.

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- statsmodels
- pmdarima

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/StockPriceForecasting_ARIMA.git
   ```
2. Navigate to the project directory:
   ```bash
   cd StockPriceForecasting_ARIMA
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Load the dataset:
   ```python
   import pandas as pd

   data = pd.read_csv('path/to/your/data.csv')
   ```
2. Check for missing values:
   ```python
   data.isnull().sum()
   ```
3. Visualize the data:
   ```python
   import matplotlib.pyplot as plt

   plt.plot(data['Adj Close'])
   plt.title('Stock Prices of HCL')
   plt.show()
   ```
4. Check for stationarity using Augmented Dicky Fuller (ADF) test and make the data stationary if needed:
   ```python
   from statsmodels.tsa.stattools import adfuller

   result = adfuller(data['Adj Close'])
   print('ADF Statistic:', result[0])
   print('p-value:', result[1])
   ```
5. Apply the ARIMA model:
   ```python
   from pmdarima import auto_arima

   model = auto_arima(data['Adj Close'], seasonal=False)
   model.summary()
   ```
6. Make predictions and evaluate the model:
   ```python
   train = data[:int(0.75*len(data))]
   test = data[int(0.75*len(data)):]

   model.fit(train['Adj Close'])
   predictions = model.predict(n_periods=len(test))

   plt.plot(test['Adj Close'].values, label='Actual')
   plt.plot(predictions, label='Predicted')
   plt.legend()
   plt.show()
   ```

## Results
The ARIMA model with optimal parameters (p=0, d=1, q=1) is used to predict the stock prices. The model performance is evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).

## Further Reading
- [Autoregressive Integrated Moving Average - Wikipedia](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average)
- [What is ARIMA and SARIMA Model?](https://becominghuman.ai)
- [Autocorrelation and Partial Autocorrelation in Time Series Data](https://statisticsbyjim.com)

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
