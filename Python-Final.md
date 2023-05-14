# Python-Final
 Final for BISC 450
## Useing Jupiter Notebooks (1) 

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


