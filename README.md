# Analysis and prediction of traffic accidents
### Final project of Master degree studies

The subject of this project is the analysis of data on traffic accidents in Belgrade that occurred during 2017, as well
as their prediction, which is of great importance for the improvement of traffic and risky roads in the city.

## Project goal

Using different machine learning algorithms, the aim is to compare the results of their work and select the most
suitable one that gives the best results. After that, using the chosen algorithm, it is necessary to predict the
locations in the city where a traffic accident would occur at a certain time and under certain weather conditions.

## Processing description

The inputs for machine learning are located inside the CSV file, with columns such as the coordinates of the location of
the accident, type, exact time and description of the accident, as well as climate change at the time of the accident.
Based on this data, machine learning of various algorithms is started using the Python programming language (using
machine learning libraries), and it is determined which of them is the most suitable for further prediction. Using input
data such as climate change over a period of time, it is necessary to predict the location and type of accident where it
could occur at a given time.

### Technologies used

- Python
- Scikit-learn library

### Prerequisites

- Python - Version from 3.3.x and above

## Starting machine learning algorithms 

The idea is simple. In [algorithms](algorithms) directory, there are Python scripts for every ML algorithm, which are:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression

By running each one of them, results would be printed in console output.

### Dataset

[Dataset](dataset/generated_accidents_dataset.csv) which contains data for analysing and further processing is already
prepared with all necessary values for algorithms. Another application was used for generating this dataset in this
specific format.

## How to launch algorithms

1. Check that you have Python version 3.3+
2. Clone the code: `git clone https://github.com/marko-domic/etf.traffic-accidents.git`
3. Setup Python interpreter for this project
4. Run ML algorithm script from [algorithms](algorithms) directory

## Possible improvements

Beside these 3, it is possible to find more suitable regression algorithms for this kind of problem, which are more
efficient, faster and give better results.
