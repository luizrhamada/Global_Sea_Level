import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('sealevel.csv')
df = pd.DataFrame(data)

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1048 entries, 0 to 1047
Data columns (total 9 columns):
 #   Column                       Non-Null Count  Dtype  
---  ------                       --------------  -----  
 0   Year                         1048 non-null   int64  
 1   TotalWeightedObservations    1048 non-null   float64
 2   GMSL_noGIA                   1048 non-null   float64
 3   StdDevGMSL_noGIA             1048 non-null   float64
 4   SmoothedGSML_noGIA           1048 non-null   float64
 5   GMSL_GIA                     1048 non-null   float64
 6   StdDevGMSL_GIA               1048 non-null   float64
 7   SmoothedGSML_GIA             1048 non-null   float64
 8   SmoothedGSML_GIA_sigremoved  1048 non-null   float64
dtypes: float64(8), int64(1)
memory usage: 73.8 KB
"""

df = df.dropna()
df.info() # there is no non-null values on dataset
pd.set_option('display.max_columns', None)
# print(df.head())

"""
Year     TotalWeightedObservations  GMSL_noGIA  StdDevGMSL_noGIA  
0  1993                  327401.31      -38.59             89.86   
1  1993                  324498.41      -41.97             90.86   
2  1993                  333018.19      -41.93             87.27   
3  1993                  297483.19      -42.67             90.75   
4  1993                  321635.81      -37.86             90.26   

   SmoothedGSML_noGIA  GMSL_GIA  StdDevGMSL_GIA  SmoothedGSML_GIA
0              -38.76    -38.59           89.86            -38.75   
1              -39.78    -41.97           90.86            -39.77   
2              -39.62    -41.91           87.27            -39.61   
3              -39.67    -42.65           90.74            -39.64   
4              -38.75    -37.83           90.25            -38.72   

   SmoothedGSML_GIA_sigremoved  
0                       -38.57  
1                       -39.11  
2                       -38.58  
3                       -38.34  
4                       -37.21
"""

# print(df.isnull().values.any()) # False
# print(df.isnull().sum())
"""
Year                           0
TotalWeightedObservations      0
GMSL_noGIA                     0
StdDevGMSL_noGIA               0
SmoothedGSML_noGIA             0
GMSL_GIA                       0
StdDevGMSL_GIA                 0
SmoothedGSML_GIA               0
SmoothedGSML_GIA_sigremoved    0
dtype: int64
"""

# Identifying and removing duplicates
# print(df.drop_duplicates(inplace=True)) # None

pd.set_option('display.max_columns', 9)
# print(df.describe())
"""
              Year  TotalWeightedObservations   GMSL_noGIA  StdDevGMSL_noGIA  \
count  1048.000000                1048.000000  1048.000000       1048.000000   
mean   2006.742366              326568.269981     4.645515         87.007700   
std       8.231978               28044.226934    26.351001          5.525201   
min    1993.000000                 906.100000   -44.390000         77.410000   
25%    2000.000000              327418.897500   -18.250000         83.510000   
50%    2007.000000              331979.205000     1.930000         85.925000   
75%    2014.000000              335243.865000    25.857500         88.602500   
max    2021.000000              341335.090000    57.920000        118.720000   

       SmoothedGSML_noGIA     GMSL_GIA  StdDevGMSL_GIA  SmoothedGSML_GIA  \
count         1048.000000  1048.000000     1048.000000       1048.000000   
mean             4.702004     8.112557       87.062805          8.168273   
std             26.171990    28.310139        5.557840         28.138879   
min            -39.780000   -43.140000       77.420000        -39.770000   
25%            -17.882500   -16.615000       83.617500        -16.087500   
50%              1.505000     5.465000       85.935000          5.065000   
75%             26.115000    30.942500       88.732500         31.095000   
max             56.310000    64.390000      118.760000         63.070000   

       SmoothedGSML_GIA_sigremoved  
count                  1048.000000  
mean                      8.213044  
std                      27.976127  
min                     -39.110000  
25%                     -17.440000  
50%                       5.170000  
75%                      29.697500  
max                      60.560000 

After analyses some brief concepts about Global Isostatic Adjustment, it's possible to infer that the columns Year and 
TotalWeightedObservations has no estatistic signification to estatistic modelling, only to identification and measuaring 
of counting.

Depois de analisar alguns conceitos breve sobre Ajuste Isostático GLobal, é possível inferir que as colunas Year e 
TotalWeightedObservations não possuem significado estatístico na modelagem estatística, apenas para fins de 
identificação da medição e contagem
"""

plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['GMSL_noGIA'])
# plt.scatter(df['Year'], df['GMSL_noGIA'])
plt.title('Year x GMSL_noGIA')
plt.xlabel('Year')
plt.ylabel('GMSL_noGIA (mm)')
plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['StdDevGMSL_noGIA'])
plt.title('Year x StdDevGMSL_noGIA')
plt.xlabel('Year')
plt.ylabel('StdDevGMSL_noGIA (mm)')
plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['SmoothedGSML_noGIA'])
plt.title('Year x SmoothedGSML_noGIA')
plt.xlabel('Year')
plt.ylabel('SmoothedGSML_noGIA (mm)')
plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['GMSL_GIA'])
plt.title('Year x GMSL_GIA')
plt.xlabel('Year')
plt.ylabel('GMSL_GIA (mm)')
plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['StdDevGMSL_GIA'])
plt.title('Year x StdDevGMSL_GIA')
plt.xlabel('Year')
plt.ylabel('StdDevGMSL_GIA (mm)')
plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['SmoothedGSML_GIA'])
plt.title('Year x SmoothedGSML_GIA')
plt.xlabel('Year')
plt.ylabel('SmoothedGSML_GIA (mm)')
plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 6))
plt.bar(df['Year'], df['SmoothedGSML_GIA_sigremoved'])
plt.title('Year x SmoothedGSML_GIA_sigremoved')
plt.xlabel('Year')
plt.ylabel('SmoothedGSML_GIA_sigremoved (mm)')
plt.grid(True)
# plt.show()

"""
- After the last Glacial Era, the global level sea began to rise. Analysing the graphics, it's possible to see that 
between 2004 and 2005 years, Smoothed GSML, GSML without Global Isostatic Adjustment (GIA) applied or not applied 
have the milimeter changing from negative to positive, in the same years. It could be occurinr due to relative reference 
year choosed by scientists, in other words, if in 2005 had been choosed as reference year (year zero), the before data 
of this year could be show negative values, indicating that the sea level had mean value bellow of that 2005.

- Plotting a scatter graphic, it's possible to see that there is a linear behavior of rising of level sea since 1993 
until 2021.
"""

# # Identifying and removing possible outliers - Z-score
# zscore = stats.zscore(df)
# anomaly = df[np.abs(zscore)>3]
# df = df[zscore < 3]
# print(f'Anomaly identifyed:\n{anomaly}')
#
# """
# Anomaly identifyed:
#       Year  TotalWeightedObservations  GMSL_noGIA  StdDevGMSL_noGIA  \
# 20    1993                  186585.20      -28.46             90.91
# 107   1995                     906.10      -44.39             77.41
# 112   1996                  223561.00      -32.27             81.13
# 163   1997                  161160.91      -30.76             96.63
# 172   1997                  318493.59      -15.03            106.37
# ...    ...                        ...         ...               ...
# 962   2019                   62573.70       44.49             89.80
# 966   2019                  151972.00       43.38             87.62
# 996   2020                  201531.91       46.87             86.69
# 997   2020                   36361.00       41.22             89.08
# 1010  2020                  219988.30       44.15             82.70
#
#       SmoothedGSML_noGIA  GMSL_GIA  StdDevGMSL_GIA  SmoothedGSML_GIA  \
# 20                -34.73    -28.32           90.92            -34.60
# 107               -25.34    -43.14           77.42            -24.63
# 112               -29.81    -31.53           81.12            -29.07
# 163               -26.62    -29.69           96.59            -25.54
# 172               -14.55    -13.90          106.46            -13.41
# ...                  ...       ...             ...               ...
# 962                45.45     50.80           89.65             51.81
# 966                44.46     49.86           87.53             50.88
# 996                45.81     53.47           86.67             52.42
# 997                45.65     48.01           89.13             52.27
# 1010               46.64     50.91           82.75             53.35
#
#       SmoothedGSML_GIA_sigremoved
# 20                         -33.72
# 107                        -26.87
# 112                        -28.19
# 163                        -21.66
# 172                        -17.40
# ...                           ...
# 962                         53.56
# 966                         53.62
# 996                         53.60
# 997                         53.69
# 1010                        56.96
#
# [70 rows x 9 columns]
# """
# df.info()

# Correlation matrix

corr = df.corr(method='spearman')
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.xticks(rotation=15)
plt.yticks(rotation=40)
# plt.show()

# Linear Regression

# Separando as variáveis independentes da dependente
X = np.array(df['Year']).reshape(-1,1)
y = np.array(df['SmoothedGSML_GIA_sigremoved'])

# Reparting in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 70% as training and 30% as test

# Creating a model
model = LinearRegression()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

# Linear Regression metrics
mse = mean_squared_error(y_test, y_predict)
print(f'MSE (Mean Squared Error) = {mse:.2f}')

mae = mean_absolute_error(y_test, y_predict)
print(f"MAE(Mean Absolut Error): {mae:.2f}")

r2 = r2_score(y_test, y_predict)
print(f'R² (Determination Coeficient): {r2:.2f}')

# Prediction
future_year = int(input('Type a year you want to predict: '))
year = np.array([[future_year]])
global_level_sea_2030 = model.predict(year)
print(f"The mean rise of global level sea in {future_year} could be {global_level_sea_2030[0]:.2f} mm")



