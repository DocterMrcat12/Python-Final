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

<!-- as much as I loved orginization and notes I now have a need for speed so ill tak the points off to just turn this in on time-->
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

``` python
odds = [1, 3, 5, 7]

for num in odds:
    print(num)
    ```
    
    ```
    1
3
5
7
```
``` python
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
```

```
1
3
5
7
```


```python
length = 0
names = ['curie', 'darwin', 'turing']

for value in names:
    length = length + 1
print(length)
```

    3 



```python
name = 'rosalind'
for name in ['curie', 'darwin', 'turing']:
    print(name)
print( name)
```

    curie
    darwin
    turing
    turing



```python
print(len([0, 1, 2, 3]))
```

    4



```python
name = ['curie', 'darwin', 'turing']
print(len(name))
```

    3
    
    ``` python
    
    import glob

```

``` python
print(glob.glob('inflammation*.csv'))
```

```
['inflammation-05.csv', 'inflammation-12.csv', 'inflammation-04.csv', 'inflammation-08.csv', 'inflammation-10.csv', 'inflammation-06.csv', 'inflammation-09.csv', 'inflammation-01.csv', 'inflammation-07.csv', 'inflammation-11.csv', 'inflammation-03.csv', 'inflammation-02.csv']
import glob
```

``` python
import glob
import numpy
import matplotlib.pyplot

filenames = sorted(glob.glob('inflammation*.csv'))
filenames = filenames[0:3]
for filename in filenames:
    print(filename)

    data = numpy.loadtxt(fname=filename, delimiter=',')

    fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))

    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)

    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis=0))

    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis=0))

    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis=0))

    fig.tight_layout()
    matplotlib.pyplot.show()
    ```
    
![why is it grey ;-;](https://github.com/DocterMrcat12/Python-Final/assets/133600341/19320034-52d9-4579-af66-4c0e22d5bba7)
![AHHHHHH](https://github.com/DocterMrcat12/Python-Final/assets/133600341/17ea6c8d-1e87-4e18-97b1-c05f4a87b70c)

```
Making Choices

``` python
num = 37
if num > 100:
    print('greater')
else:
    print('not greater')
print('done')
```
```
    not greater
    done
    ```
```


```python
num = 53
print('before conditional...')
if num > 100:
    print(num, 'is greater than 100')
print('...after conditional')
```
```
    before conditional...
    ...after conditional

```

``` python
num = 14
if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else: print(num, 'is negative')
```
```
    14 is positive
```


``` python
if (1 > 0) and (-1 >= 0):
    print('both parts are true')
else:
    print('at least one part is false')
```
```
    at least one part is false
```


``` python
if (-1 > 0) or (-1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```
```
    both of these are false
```


``` python
import numpy
```


``` python
data = numpy.loadtxt(fname='inflammation-01.csv', delimiter = ',')
```


``` python
max_inflammation_0 = numpy.amax(data, axis=0)[0]
```

``` python
max_inflammation_20 = numpy.amax(data, axis=0)[20]
```
```python
if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
```
```
    Suspicious looking maxima!
```


```python
max_inflammation_20 = numpy.amax(data, axis=0)[20]
if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
elif numpy.sum(numpy.amin(data, axis=0)) == 0:
    print('Minima add up to zero!')
else:
    print('Seems ok!')
```
```
    Suspicious looking maxima!
```


```python
data = numpy.loadtxt(fname = 'inflammation-03.csv', delimiter=',')

max_inflammation_0 = numpy.amax(data, axis=0)[0]

max_inflammation_20 = numpy.amax(data, axis=0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Suspicious looking maxima!')
elif numpy.sum(numpy.amin(data, axis=0)) == 0:
    print('Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!')
else:
    print('Seems OK!')
```
```
    Minima add up to zero! -> HEALTHY PARTICIPANT ALERT!
```



```python
fahrenheit_val = 99
celsius_val = ((fahrenheit_val - 32) * (5/9))

print(celsius_val)
```
```
    37.22222222222222
```


```python
def explicit_fahr_to_celsius(temp))
    converted = ((temp - 32) * (5/9))
    return converted
```


``` python
def fahr_to_celsius(temp):
    # More efficient function without creating a new variable
    return ((temp -32) * (5/9))
```


``` python
fahr_to_celsius(32)
```


```
    0.0
```


```python
explicit_fahr_to_celsius(32)
```


```
    0.0
    ```


```python
print(fahr_to_celsius(32), 'C')
print(fahr_to_celsius(212), 'C')
```
```
   0.0 C
   100.0 C
```


```python
def celsius_to_kelvin(temp_c):
    return temp_c + 273.15
print(celsius_to_kelvin(0.))
```
```
   273.15
```


``` python
def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print(fahr_to_kelvin(212.0))
```
```
   373.15
```


```python
temp_kelvin = fahr_to_kelvin(212)
print(temp_kelvin)
```
```
    Temperature in Kelvin was: 373.15
```


```python
def print_temperatures():
    print(temp_fahr)
    print(temp_kelvin)

temp_fahr = 212.0
temp_kelvin = fahr_to_kelvin(temp_fahr)

print_temperatures()
```
```
    Temperature in Fahrenheit was: 212.0
    Temperature in Kelvin was: 373.15
```

siydhvbfshdbfioavhbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbdf
``` python
import numpy
import matplotlib.pyplot
import glob
```

```python
def visualize(filename):
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize= (10.0, 3.0))
    
    axes1 = fig.add_subplot(1, 3, 1)
    axes2 = fig.add_subplot(1, 3, 2)
    axes3 = fig.add_subplot(1, 3, 3)
    
    axes1.set_ylabel('Average')
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_ylabel('Max')
    axes2.plot(numpy.mean(data, axis=0))
    
    axes3.set_ylabel('Min')
    axes3.plot(numpy.mean(data, axis=0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```

```python
def detect_problems(filename):
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis=0)[20] == 20:
        print("Suspicious looking maxima!")
    elif numpy.sum(numpy.amin(data, axis=0)) == 0:
        print('Minima add up to zero!')
    else:
        print('Seems ok!')
```


```python
filenames = sorted(glob.glob('inflammation*.csv'))

for filename in filenames:
    print(filename)
    visualize(filename)
    detect_problems(filename)
```
![1](https://github.com/DocterMrcat12/Python-Final/assets/133600341/69c09d06-004e-4664-9f92-cf47acd72879)
![2](https://github.com/DocterMrcat12/Python-Final/assets/133600341/6040b2b7-aecc-46e4-bcd3-d03f929f67db)
![3](https://github.com/DocterMrcat12/Python-Final/assets/133600341/8a761ae4-2e4c-428b-ad92-b2412711121c)
![4](https://github.com/DocterMrcat12/Python-Final/assets/133600341/0cfba26e-10f6-4b3f-a98d-2eeb62204b85)
![5](https://github.com/DocterMrcat12/Python-Final/assets/133600341/77cfd58a-67c6-4c96-8191-0d6622814d67)
![6](https://github.com/DocterMrcat12/Python-Final/assets/133600341/74f0d22b-8ce6-4f78-9f94-ff5f07575659)
![7](https://github.com/DocterMrcat12/Python-Final/assets/133600341/a2e1fd7b-2562-45e3-87e8-262aca3adc97)
![8](https://github.com/DocterMrcat12/Python-Final/assets/133600341/72d6311b-f35e-4f12-bb2c-e4e8e81318e8)
![9](https://github.com/DocterMrcat12/Python-Final/assets/133600341/8e1b90ba-3706-414b-b9c8-01964b649c0b)
![10](https://github.com/DocterMrcat12/Python-Final/assets/133600341/8485d377-f03a-41c9-94a8-17c0ce2e68a3)
![11](https://github.com/DocterMrcat12/Python-Final/assets/133600341/9bafdbdd-fc2f-4646-9b8b-62618ea2f331)
![12](https://github.com/DocterMrcat12/Python-Final/assets/133600341/83d7a8b2-4106-4a30-bad3-097d1e0cb7a1)



```python
def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```
```python
z = numpy.zeros((2,2))
print(offset_mean(z, 3))
```
```
    [[3. 3.]
     [3. 3.]]
```


```python
data = numpy.loadtxt(fname = 'inflammation-01.csv', delimiter = ',')

print(offset_mean(data, 0))
```
```
    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]

```

```python
print(numpy.amin(data), numpy.mean(data), numpy.amax(data))
offset_data = offset_mean(data, 0)
print(
    numpy.amin(offset_data),
    numpy.mean(offset_data),
    numpy.amax(offset_data))
```
```
   0.0 6.14875 20.0
   -6.14875 2.842170943040401e-16 13.85125

```

```python
print(numpy.std(data), numpy.std(offset_data))
```
```
    4.613833197118566 4.613833197118566
```

```python
print(
      numpy.std(data) - numpy.std(offset_data))
```
```
0.0
```


```python
def offset_mean(data, target_mean_value):
    return (data - numpy.mean(data)) + target_mean_value
```


```python
def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + target_mean_value
```

```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
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



```python
def offset_mean(data, target_mean_value = 0.0):
    return(data - numpy.mean(data)) + target_mean_value
```


```python
test_data = numpy.zeros((2,2))
print(offset_mean(test_data, 3))
```
```
    [[3. 3.]
     [3. 3.]]
```


```python
print(offset_mean(test_data))
```
```
    [[0. 0.]
     [0. 0.]]
```


```python
def display(a=1, b=2, c=3):
    print('a:', a, 'b:', b, 'c:', c)
print('no parameters:')
display()
print('one parameter:')
display(55)
print('two paramters:')
display(55,66)
```
sdflihhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh

    no parameters:
    a: 1 b: 2 c: 3
    one parameter:
    a: 55 b: 2 c: 3
    two paramters:
    a: 55 b: 66 c: 3



```python
print('only setting the value of c')
display(c = 77)
```

    only setting the value of c
    a: 1 b: 2 c: 77



```python
help(numpy.loadtxt)
```

    Help on function loadtxt in module numpy:
    
    loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
        Load data from a text file.
        
        Each row in the text file must have the same number of values.
        
        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings for Python 3k.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        
            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.
        
            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.
        
            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.
        
            .. versionadded:: 1.16.0
        
        Returns
        -------
        out : ndarray
            Data read from the text file.
        
        See Also
        --------
        load, fromstring, fromregex
        genfromtxt : Load data with missing values handled as specified.
        scipy.io.loadmat : reads MATLAB data files
        
        Notes
        -----
        This function aims to be a fast reader for simply formatted files.  The
        `genfromtxt` function provides more sophisticated handling of, e.g.,
        lines with missing values.
        
        .. versionadded:: 1.10.0
        
        The strings produced by the Python float.hex method can be used as
        input for floats.
        
        Examples
        --------
        >>> from io import StringIO   # StringIO behaves like a file object
        >>> c = StringIO(u"0 1\n2 3")
        >>> np.loadtxt(c)
        array([[0., 1.],
               [2., 3.]])
        
        >>> d = StringIO(u"M 21 72\nF 35 58")
        >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
        ...                      'formats': ('S1', 'i4', 'f4')})
        array([(b'M', 21, 72.), (b'F', 35, 58.)],
              dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])
        
        >>> c = StringIO(u"1,0,2\n3,0,4")
        >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
        >>> x
        array([1., 3.])
        >>> y
        array([2., 4.])
    



```python
numpy.loadtxt('inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
def s(p):
    s = 0
    for v in p:
        a += v
    m = a / len(p)
    d = 0
    for v in p:
        d += (v - m) * (v - m)
    return numpy.sqrt(d / len(p) - 1)

# same as above, but readable
def std_dev(sample):
    sample_sum = 0
    for value in sample:
        sample_sum += value

    sample_mean = sample_sum / len(sample)

    sum_squared_devs = 0
    for value in sample:
        sum_squared_devs += (value - sample_mean) * (value - sample_mean)

    return numpy.sqrt(sum_sqared_devs / len(sample) - 1)
```


## Defensive Programming
Using assertions as part of defensive programming strategies


```python
numbers = [1.5, 2.3, 0.7, 0.001, 4.4]
total = 0.0
for num in numbers:
    assert num > 0.0, 'Data should only contain positive values'
    total += num
print('total is:', total)
```

    total is: 8.901



```python
def normalize_rectangle(rect):
    """Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    Input should be of the format (x0, y0, x1, y1).
    (x0, y0) and (x1, y1) define the lower left and upper right corners of the recangle, respectively."""
    assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid X coordinates'
    assert y0 < y1, 'Invalid Y coordinates'

    dx = x1 - x0
    dy = y1 - y0
    if dx > dy:
        scaled = dx / dy
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dx / dy
        upper_x, upper_y = scaled, 1.0
    
    assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'

    return(0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle((0.0, 1.0, 2.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-6-a81b6ed7619a> in <module>
    ----> 1 print(normalize_rectangle((0.0, 1.0, 2.0)))
    

    <ipython-input-5-4a7982d53b1a> in normalize_rectangle(rect)
          3     Input should be of the format (x0, y0, x1, y1).
          4     (x0, y0) and (x1, y1) define the lower left and upper right corners of the recangle, respectively."""
    ----> 5     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
          7     assert x0 < x1, 'Invalid X coordinates'


    AssertionError: Rectangles must contain 4 coordinates



```python
print(normalize_rectangle((4.0, 2.0, 1.0, 5.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-7-5e28a32bada1> in <module>
    ----> 1 print(normalize_rectangle((4.0, 2.0, 1.0, 5.0)))
    

    <ipython-input-5-4a7982d53b1a> in normalize_rectangle(rect)
          5     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
          6     x0, y0, x1, y1 = rect
    ----> 7     assert x0 < x1, 'Invalid X coordinates'
          8     assert y0 < y1, 'Invalid Y coordinates'
          9 


    AssertionError: Invalid X coordinates



```python
print(normalize_rectangle((0.0, 0.0, 1.0, 5.0)))
```

    (0, 0, 0.2, 1.0)



```python
print(normalize_rectangle((0.0, 0.0, 5.0, 1.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-9-1337bef8f4bf> in <module>
    ----> 1 print(normalize_rectangle((0.0, 0.0, 5.0, 1.0)))
    

    <ipython-input-5-4a7982d53b1a> in normalize_rectangle(rect)
         18 
         19     assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
    ---> 20     assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
         21 
         22     return(0, 0, upper_x, upper_y)


    AssertionError: Calculated upper y coordinate invalid


## Transcribing DNA into RNA
A program that takes DNA sequences (formatted as a FASTA file) and produces the equivalent RNA sequence


```python
# Prompt user to enter the input FASTA file name

input_file_name = input("Input the name of the FASTA file to be transcribed: ")
```

    Input the name of the FASTA file to be transcribed:  UBC.txt



```python
# Open the input file and read the DNA sequence

with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip()
```


```python
# Transcribe the DNA to RNA

rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide
```


```python
# Prompt the user to enter the output file name

output_name = input("Enter the name of the ouput file: ")
```

    Enter the name of the ouput file:  Ubiquitin_C



```python
# Save the RNA sequence to a text file

with open(output_name, "w") as output_file:
    output_file.write(rna_sequence)
    print(f"The RNA sequence has been saved to {output_name}")
```

    The RNA sequence has been saved to Ubiquitin_C



```python
# Print the RNA sequence

print(rna_sequence)
```

    AUGCAGAUCUUCGUGAAGACUCUGACUGGUAAGACCAUCACCCUCGAGGUUGAGCCCAGUGACACCAUCGAGAAUGUCAAGGCAAAGAUCCAAGAUAAGGAAGGCAUCCCUCCUGACCAGCAGAGGCUGAUCUUUGCUGGAAAACAGCUGGAAGAUGGGCGCACCCUGUCUGACUACAACAUCCAGAAAGAGUCCACCCUGCACCUGGUGCUCCGUCUCAGAGGUGGGAUGCAAAUCUUCGUGAAGACACUCACUGGCAAGACCAUCACCCUUGAGGUCGAGCCCAGUGACACCAUCGAGAACGUCAAAGCAAAGAUCCAGGACAAGGAAGGCAUUCCUCCUGACCAGCAGAGGUUGAUCUUUGCCGGAAAGCAGCUGGAAGAUGGGCGCACCCUGUCUGACUACAACAUCCAGAAAGAGUCUACCCUGCACCUGGUGCUCCGUCUCAGAGGUGGGAUGCAGAUCUUCGUGAAGACCCUGACUGGUAAGACCAUCACCCUCGAGGUGGAGCCCAGUGACACCAUCGAGAAUGUCAAGGCAAAGAUCCAAGAUAAGGAAGGCAUUCCUCCUGAUCAGCAGAGGUUGAUCUUUGCCGGAAAACAGCUGGAAGAUGGUCGUACCCUGUCUGACUACAACAUCCAGAAAGAGUCCACCUUGCACCUGGUACUCCGUCUCAGAGGUGGGAUGCAAAUCUUCGUGAAGACACUCACUGGCAAGACCAUCACCCUUGAGGUCGAGCCCAGUGACACUAUCGAGAACGUCAAAGCAAAGAUCCAAGACAAGGAAGGCAUUCCUCCUGACCAGCAGAGGUUGAUCUUUGCCGGAAAGCAGCUGGAAGAUGGGCGCACCCUGUCUGACUACAACAUCCAGAAAGAGUCUACCCUGCACCUGGUGCUCCGUCUCAGAGGUGGGAUGCAGAUCUUCGUGAAGACCCUGACUGGUAAGACCAUCACUCUCGAAGUGGAGCCGAGUGACACCAUUGAGAAUGUCAAGGCAAAGAUCCAAGACAAGGAAGGCAUCCCUCCUGACCAGCAGAGGUUGAUCUUUGCCGGAAAACAGCUGGAAGAUGGUCGUACCCUGUCUGACUACAACAUCCAGAAAGAGUCCACCUUGCACCUGGUGCUCCGUCUCAGAGGUGGGAUGCAGAUCUUCGUGAAGACCCUGACUGGUAAGACCAUCACUCUCGAGGUGGAGCCGAGUGACACCAUUGAGAAUGUCAAGGCAAAGAUCCAAGACAAGGAAGGCAUCCCUCCUGACCAGCAGAGGUUGAUCUUUGCUGGGAAACAGCUGGAAGAUGGACGCACCCUGUCUGACUACAACAUCCAGAAAGAGUCCACCCUGCACCUGGUGCUCCGUCUUAGAGGUGGGAUGCAGAUCUUCGUGAAGACCCUGACUGGUAAGACCAUCACUCUCGAAGUGGAGCCGAGUGACACCAUUGAGAAUGUCAAGGCAAAGAUCCAAGACAAGGAAGGCAUCCCUCCUGACCAGCAGAGGUUGAUCUUUGCUGGGAAACAGCUGGAAGAUGGACGCACCCUGUCUGACUACAACAUCCAGAAAGAGUCCACCCUGCACCUGGUGCUCCGUCUUAGAGGUGGGAUGCAGAUCUUCGUGAAGACCCUGACUGGUAAGACCAUCACUCUCGAAGUGGAGCCGAGUGACACCAUUGAGAAUGUCAAGGCAAAGAUCCAAGACAAGGAAGGCAUCCCUCCUGACCAGCAGAGGUUGAUCUUUGCUGGGAAACAGCUGGAAGAUGGACGCACCCUGUCUGACUACAACAUCCAGAAAGAGUCCACCCUGCACCUGGUGCUCCGUCUCAGAGGUGGGAUGCAAAUCUUCGUGAAGACCCUGACUGGUAAGACCAUCACCCUCGAGGUGGAGCCCAGUGACACCAUCGAGAAUGUCAAGGCAAAGAUCCAAGAUAAGGAAGGCAUCCCUCCUGAUCAGCAGAGGUUGAUCUUUGCUGGGAAACAGCUGGAAGAUGGACGCACCCUGUCUGACUACAACAUCCAGAAAGAGUCCACUCUGCACUUGGUCCUGCGCUUGAGGGGGGGUGUCUAA



```python

```


## Translating RNA into Protein
A program that takes the RNA sequence created previously and translates it into the corresponding chain of amino acids (i.e. a protein)


```python
# Prompt user to enter the input RNA file name

input_name = input("Input the name of the file containing the RNA sequence to be translated: ")
```

    Input the name of the file containing the RNA sequence to be translated:  Ubiquitin_C



```python
# Open the file and read the RNA sequence

with open(input_name, "r") as input_file:
    rna_sequence = input_file.read().strip()
```


```python
# Define the codon table

codon_table = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G"
}
```


```python
# Translate RNA to protein

protein_sequence = ""
for i in range(0, len(rna_sequence), 3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid
```


```python
# Prompt user to enter output file name

output_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  UbiquitinC_Protein.txt



```python
# Save the protein sequence to a text file

with open(output_name, "w") as output_file:
    output_file.write(protein_sequence)
    print(f"The protein sequence has been saved to {output_name}")
```

    The protein sequence has been saved to UbiquitinC_Protein.txt



```python
# Print the protein sequence

print(protein_sequence)
```





































