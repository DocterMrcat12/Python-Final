# Python-Final
 Final for BISC 450
## Useing Jupiter Notebooks (1 and 2) 

<!-- Its rude to look at a girls code. Nothing to see here just a bunch of notes. That was a joke by the way...-->
Importing Data Packages
``` python
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set(style = "darkgrid")
```
<!-- each code block set up sepeartly -->
Reading Imported File
``` python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/Notebooks/fortune500.csv')
```
<!-- Excuse the stupid notes that will follow they stop me from going insane from all the copy pasting -->
Reading Head of File
``` python
df.head()
```

<!-- making this table might actually kill me -->

|  | Year |Rank |Company |Revenue (in millions) |Profit (In millions) |
| ------------- | ------------- | ---------- | ----------| ---------- | ---------- |
| 0 | 1995 | 1 | General Motors| 9823.5 | 806 |
| 1 | 1995 | 2 | Exxon Mobil| 5661.4 | 584.8 |
| 2 | 1995 | 3 | U.S. Steel| 3250.4 | 195.4 |
| 3 | 1995 | 4 | General Electric | 2959.1 | 212.6 |
| 4 | 1995 | 5 | Esmark | 2510.8 | 19.1 |

<!-- if I have to go back and fix this I will die -->

Reading Tail of File
``` python
df.tail()
```

|  | Year |Rank |Company |Revenue (in millions) |Profit (In millions) |
| ------------- | ------------- | ---------- | ----------| ---------- | ---------- |
| 25495 | 2005 | 496 | Wm. Wrigley Jr.| 3648.6 | 493 |
| 25496 | 2005 | 497 | Peabody Energy| 3631.6 | 175.4|
| 25497| 2005 | 498 | Wendy's International| 3630.4 | 57.8 |
| 25498 | 2005 | 499 | Kindred Healthcare | 3616.6 | 70.6 |
| 25499 | 2005 | 500 | Cincinnati Financial | 3614.0 | 584 |

Renaming Columns of File
``` python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```

|  | year |rank |company |revenue |profit  |
| ------------- | ------------- | ---------- | ----------| ---------- | ---------- |
| 0 | 1995 | 1 | General Motors| 9823.5 | 806 |
| 1 | 1995 | 2 | Exxon Mobil| 5661.4 | 584.8 |
| 2 | 1995 | 3 | U.S. Steel| 3250.4 | 195.4 |
| 3 | 1995 | 4 | General Electric | 2959.1 | 212.6 |
| 4 | 1995 | 5 | Esmark | 2510.8 | 19.1 |

Figuring out Ammount of Rows
``` python
len(df)
```
```
25500
```
Listing Types of Various Columns 
``` python
df.dtypes
```
```
year         int64
rank         int64
company     object
revenue    float64
profit      object
dtype: object
```
Listing Non-Numeric Profit Items Head
```python
non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head()
```
|  | year |rank |company |revenue |profit  |
| ------------- | ------------- | ---------- | ----------| ---------- | ---------- |
| 228 | 1955 | 229 | Norton| 135.0 | N.A. |
| 290 | 1955 | 291 | Schlitz Brewing| 100.0 | N.A. |
| 294 | 1955 | 295 | Pacific Vegetable Oil| 97.9 | N.A. |
| 296 | 1955 | 297 | Liebmann Breweries | 96.0 | N.A. |
| 352 | 1955 | 353 | Minneapolis-Moline | 77.4| N.A. |

<!-- why wont python just let me copy the tables ;-; -->

Listing All Non-Numeric Profit Sets
``` python
set(df.profit[non_numeric_profits])
```
```
{'N.A.'}
```
Listing How Many N.A. Values Exist in Data Set
``` python
len(df.profit[non_numeric_profits])
```
```
369
```
Plotting Histiogram of N.A. Values by Year
``` python
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins= range(1955, 2006) )
```

![Histiogram](https://github.com/DocterMrcat12/Python-Final/assets/133600341/1302a34e-3d11-439f-82fd-488a06a5c1fc)

Saving Over Data Frame and Removing Non-Numeric Profits
``` python
df = df.loc[~non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```

Calculating New Length
``` python
len(df)
```

```
25131 
```
Checking If the N.A. was properly removed
``` python
df.dtypes
```
```
year         int64
rank         int64
company     object
revenue    float64
profit     float64
dtype: object
```

Making a Plot of 'Increase in mean Fortune 500 company profits from 1995 to 2005'
``` python
group_by_year = df.loc[:, ['year','revenue','profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    ax.margins(x = 0,y = 0 )
```
``` python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1995 to 2005', 'Profit (millions)')
```
![Yet Another Plot](https://github.com/DocterMrcat12/Python-Final/assets/133600341/17d3e29d-66bd-4015-a80b-fd1515aee8ca)

Ploting by Revenue Instead

``` python
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955 to 2005', 'Revenue (Millions)')
```
![Need to check if this is working soon](https://github.com/DocterMrcat12/Python-Final/assets/133600341/5191f16b-676f-4c60-a514-b2a9d734fb44)

Making an Overlay Plot with Standard Error of Both Revenue and Profit Graphs

``` python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    plot(x, y, ax, title, y_label)
fig, (ax1, ax2) = plt.subplots(ncols = 2)
title = 'Increase in mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year. std().profit.values
stds2 = group_by_year. std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'Profits', 'Profit(millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'Revenue', 'Revenue (Millions)')
fig. set_size_inches(14,4)
fig.tight_layout()
```

![Done](https://github.com/DocterMrcat12/Python-Final/assets/133600341/001cc338-6495-46a9-a98b-f5dfdc37d2be)

<!-- I regret everything, I suck at unpaced classes, and now it's crunch time baby, three days and 2/3 of the class... I thinpeoplepeople'sinitialsI'mk most poeple literally just copy pasted other peoples code for this just saying just in case let me sneak in my initals all over the place - AV though I doubt anyone will steal my code since im going to be turning it in last minute :P -->

## Python Fundamentals

Any Python Interpreter can be used as a calculator:

``` python 
3 + 4 * 4
``` 
```
23
```
Saving Variables in Python

``` python 
weight_kg = 60
```

``` python
print(weight_kg)
```

```
60
```

Setting Peramitors For Vairbales

```
Weight0 = Valud
0weight = invalid
weight and Weight are diffrent
```

The Diffrent Types of Data

```
Interger Numbers: All Whole Numbers (exp. 60)
Floating Point Numbers: Numbers with Decimals (exp. 60.3)
Strings: Any Combination of Valid Charicters Made into a Variable (exp. Jhon_Smith)
```

``` python
weight_kg = 60.3
```
``` python
patient_name = "Jon Smith"
```
``` python
patient_id = 001
```

Using Preset Variables to Perform Calculations

``` python
weight_lb = 2.2 * weight_kg

print(weight_lb)
```
```
132.66
```

Appenging Variables
``` python
patient_id = 'inflam_' + patient_id

print(patient_id)
```
```
inflam_001
```

<!-- apparently this is a poor way to do it but well i dont know the better way-->

Combining Print Statments

``` python

pring(patient_id, 'weight in kilograms:', weight_kg)
```
```
inflam_001 weight in kinograms 60.3
```

Nested Functions

``` python
print(type(60.3))

print(type(patient_id))
```
```
<class 'float'>
<class 'str'>
```

Calculations Nested in Print Functions

``` python
print('weight in lbs:', 2,2 * weight_kg)
```
```
weight in lbs: 132.66
```

Overwriting Variables

``` python
weight_kg = 65.0
print('weight in kilograms in now', weight_kg)
```
<!-- Sneaky Intials AV-->
```
weight in kilograms is now: 65.0
```

Importing Numpy (A Numcerical Coding System)

``` python 
import numpy
```

Loading Text from Numpy Package

``` python 
numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```
```
array([[0., 0., 1., ..., 3., 0., 0.],
       [0., 1., 2., ..., 1., 0., 1.],
       [0., 1., 1., ..., 2., 1., 1.],
       ...,
       [0., 1., 1., ..., 1., 1., 1.],
       [0., 0., 0., ..., 0., 2., 0.],
       [0., 0., 1., ..., 1., 1., 0.]])
       
```
Saving Full Array as 'Data'
``` python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')
```

``` python
print (data[29,19])
```
```
16.0
```
``` python
print(data[0:4, 0:10])
```
```
[[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
 [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
 [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
 [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]
 ```
 
 ``` python
 print(data[5:10, 0:10])
 ```
 ```
 [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
 [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
 [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
 [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
 [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]
 ```
 ``` python
 small = data[:3, 36:]
 ```
 ``` python
 print(small)
 ```
 ```
 [[2. 3. 0. 0.]
 [1. 1. 0. 1.]
 [2. 2. 1. 1.]]
 ```
 
 ```python
print(numpy.mean(data))
```
    6.14875
```

```python
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)
```

```python
print(maxval)
print(minval)
print(stdval)
```
```
    20.0
    0.0
    4.613833197118566
```

```python
maxval = numpy.amax(data)
minval = numpy.amin(data)
stdval = numpy.std(data)
```

```python
print(maxval)
print(minval)
print(stdval)
```
```
    20.0
    0.0
    4.613833197118566
```

```python
print(maxval)
print(minval)
print(stdval)
```
```
    20.0
    0.0
    4.613833197118566
```

```python
patient_0 = data[0, :] 
print(numpy.amax(patient_0))
```
```
   18.0
```


```python
print(numpy.amax(data[2, :]))
```
```
    19.0
```


```python
print(numpy.mean(data, axis = 0))
```
```
    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]
```


```python
print(numpy.mean(data, axis = 0).shape)
```
```
    (40,)
```


```python
print(numpy.mean(data, axis = 1))
```
```
    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]

```

```python
import numpy
data = numpy.loadtxt(fname= 'inflammation-01.csv', delimiter = ',')
```

```python
import matplotlib.pyplot
image = matplotlib.pyplot.imshow(data)
matploylib.pyplot.show()
```

![I forgot to save ;-;](https://github.com/DocterMrcat12/Python-Final/assets/133600341/63e358ba-dbf5-4e7c-ba13-2f73d8ae98cf)

``` python
ave_infalmmation = numpy.mean(data, axis =0)
ave_ploy = matplotlib.pyplot.plot(ave_infalmmation)
matplotlib.pyplot.show()
```
![Murder me](https://github.com/DocterMrcat12/Python-Final/assets/133600341/6486a2a2-e87a-42f3-b0eb-cd0b718bd76a)

``` python
max_plot = matplotlib.pyplot.plot(numpy.amax(data, axis =0))
matplotlib.pyplot.show()
```

![why did we do these all seperatly](https://github.com/DocterMrcat12/Python-Final/assets/133600341/cbc52bd1-0f15-4b97-b60d-942e682a0bd8)

  
  ``` python
  min_plot = matplotlib.pyplot.plot(numpy.amin(data, axis =0))
matplotlib.pyplot.show()
```

![image](https://github.com/DocterMrcat12/Python-Final/assets/133600341/1cccec6e-2e7e-4ff4-ad18-a7e939885504)

``` python
fig = matplotlib.pyplot.figure(figsize =(10.0, 3.0))

axes1 = fig.add_subplot(1, 3, 1)
axes2 = fig.add_subplot(1, 3, 2)
axes3 = fig.add_subplot(1, 3, 3)

axes1.set_ylabel('average')
axes1.plot(numpy.mean(data, axis = 0))

axes2.set_ylabel('max')
axes2.plot(numpy.amax(data, axis = 0))

axes3.set_ylabel('min')
axes3.plot(numpy.amin(data, axis = 0))

fig.tight_layout()

matplotlib.pyplot.savefig('inflammation.png')
matplotlib.pyplot.show()
```

![image](https://github.com/DocterMrcat12/Python-Final/assets/133600341/1502f8ee-aa64-4379-93cb-b5b03809822b)


``` python
odds = [1, 3, 5, 7]
print(odds)

```
```
[1, 3, 5, 7]
```

``` python
print (odds[0])
print (odds[3])
print (odds[-1])
```
```
1
7
7
```

``` python
names = ['Bab', 'Tim', 'Joe']

print(names)

names[0] = 'Bob'

print(names)
```

```
['Bab', 'Tim', 'Joe']
['Bob', 'Tim', 'Joe']
```

``` python
odds.append(11)
print(odds)
```

```
[1, 3, 5, 7, 11]
```

``` python 

removed_element = odds.pop(0)
print (odds)
print (removed_element)

```

```
[3, 5, 7, 11]
1
```

```python
odds.reverse()
print (odds)
```

``` 
[11, 7, 5, 3]
```

```python
odds = [3, 5, 7]
primes = odds
primes.append(2)
print(primes)
print(odds)
```
```
[3, 5, 7, 2]
[3, 5, 7, 2]
```

```python
odds = [3, 5, 7]
primes = list(odds)
primes.append(2)
print(primes)
print(odds)
```

```
[3, 5, 7, 2]
[3, 5, 7]
```

``` python
binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes[2:5]
print('autosomes:', autosomes)

last = chromosomes[-1]
print('last:', last)
```

```
group: Drosophila
species: melanogaster
autosomes: ['2', '3', '4']
last: 4
```

``` python 
date = 'Monday4 January 2023'
day = date[0:6]
print(day)
day = date[:6]
print(day)
```

```
Monday
Monday
```

```python
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print(sond)
```

```
['oct', 'nov', 'dec']
```




































