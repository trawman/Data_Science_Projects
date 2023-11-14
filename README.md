# California Housing dataset
--------------------------

**Data Set Characteristics:**

    :Number of Instances: 20640

    :Number of Attributes: 8 numeric, predictive attributes and the target

    :Attribute Information:
        - MedInc        median income in block group
        - HouseAge      median house age in block group
        - AveRooms      average number of rooms per household
        - AveBedrms     average number of bedrooms per household
        - Population    block group population
        - AveOccup      average number of household members
        - Latitude      block group latitude
        - Longitude     block group longitude

    :Missing Attribute Values: None

This dataset was obtained from the StatLib repository.
https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

The target variable is the median house value for California districts,
expressed in hundreds of thousands of dollars ($100,000).

This dataset was derived from the 1990 U.S. census, using one row per census
block group. A block group is the smallest geographical unit for which the U.S.
Census Bureau publishes sample data (a block group typically has a population
of 600 to 3,000 people).

A household is a group of people residing within a home. Since the average
number of rooms and bedrooms in this dataset are provided per household, these
columns may take surprisingly large values for block groups with few households
and many empty houses, such as vacation resorts.

It can be downloaded/loaded using the
:func:`sklearn.datasets.fetch_california_housing` function.

.. topic:: References

    - Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
      Statistics and Probability Letters, 33 (1997) 291-297
# Imports
<p>from sklearn import datasets</p>
<p>import pandas as pd</p>
<p>from sklearn.model_selection import train_test_split, GridSearchCV</p>
<p>import seaborn as sns</p>
<p>import matplotlib.pyplot as plt</p>
<p>from sklearn.linear_model import LinearRegression</p>
<p>from sklearn.metrics import r2_score</p>
<p>from sklearn.metrics import mean_squared_error</p>
<p>import numpy as np</p>
<p>from sklearn.svm import SVR</p>

# Pre-processing
<p>housing = datasets.fetch_california_housing(as_frame = True)</p><br>

<p>df.drop('Latitude', axis=1, inplace=True)</p>
<p>df.drop('Longitude', axis=1, inplace=True)</p>
<p>df.drop('AveBedrms', axis=1, inplace=True)</p><br>

<p>df = pd.DataFrame(housing.data)</p><br>

<p>df.describe()</p>

![image](https://github.com/trawman/housing_california_project/assets/100029716/a61c53ce-8391-4f72-ab78-598aca206bb5)

<p>df.head()</p>

![image](https://github.com/trawman/housing_california_project/assets/100029716/0c44b2a9-3b99-40c2-bd67-2a4972eebb0e)

<p>sns.pairplot(df)</p>

![image](https://github.com/trawman/housing_california_project/assets/100029716/0f6d1295-d870-4964-b0ab-afc47fb23f61)

<p>df.corr()</p>

