# Real Estate Price Predictor

- Data set taken for the city of Bangalore from [kaggle.com](https://www.kaggle.com/datasets/snooptosh/bangalore-real-estate-price)

### Understanding the Table Schema:

![Untitled](Real%20Estate%20Price%20Predictor%20281b3d6e7c3c46b3bf04427a4a43b170/Untitled.png)

### Cleaning data

For the sake of simplicity we drop the columns like: area_type ,society , availability

![Untitled](Real%20Estate%20Price%20Predictor%20281b3d6e7c3c46b3bf04427a4a43b170/Untitled%201.png)

1. Then drop the null values.
2. Clean the size (as everything is in BHK) 

```python
df3['bhk'] = df3['size'].apply(lambda x : int(x.split(' ')[0]))
```

1. Cleaning the total_sqft.

```python
def convert(x):
    tokens =x.split('-')
    if(len(tokens) == 2):
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df4['total_sqft'] = df4['total_sqft'].apply(convert)

```

1. Cleaning location.

```python
df5.location = df5.location.apply(lambda x: x.strip())
```

1. Defining new column: Price Per square ft.

```python
df6 = df5[~(df5.total_sqft / df5.bhk < 30)]
```

![Untitled](Real%20Estate%20Price%20Predictor%20281b3d6e7c3c46b3bf04427a4a43b170/Untitled%202.png)

1. Removing outliers

```python
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sq_ft)
        st = np.std(subdf.price_per_sq_ft)
        reduced_df = subdf[((subdf.price_per_sq_ft) > (m-st)) & ((subdf.price_per_sq_ft) < (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index = True)
    return df_out

df7 = remove_pps_outliers(df6)
df7.shape
```

1. Removing BHK outliers

```python
def remove_bhk_outliers(df):
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            bhk_stats = {}
            for bhk, bhk_df in location_df.groupby('bhk'):
                bhk_stats[bhk] = {
                    'mean': np.mean(bhk_df.price_per_sq_ft),
                    'std' : np.std(bhk_df.price_per_sq_ft),
                    'count' : bhk_df.shape[0]
                }
            for bhk, bhk_df in location_df.groupby('bhk'):
                stats = bhk_stats.get(bhk-1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices , bhk_df[bhk_df.price_per_sq_ft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis ="index")
df8 = remove_bhk_outliers(df7)
df8.head()

```

## Visualising the data:

```python
def plot_scatter_chart(df , location):
    bhk2 = df[(df.location == location) & (df.bhk == 2)]
    bhk3 = df[(df.location == location) & (df.bhk == 3)]
    
    matplotlib.rcParams['figure.figsize'] = (15,10) #??
    
    plt.scatter(bhk2.total_sqft , bhk2.price, color = 'blue' , label = '2 BHK' , s = 50)
    plt.scatter(bhk3.total_sqft , bhk3.price,marker = '+', color = 'red' , label = '3 BHK' , s = 50)
    
    plt.xlabel("Total square feet area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7 , "Rajaji Nagar")
```

![Untitled](Real%20Estate%20Price%20Predictor%20281b3d6e7c3c46b3bf04427a4a43b170/Untitled%203.png)

We can clearly see there are some flats in Rajaji nagar with 2 BHK, which are expansive than 3 BHK.
Also, since we dropped the society column which is clearly affecting the results here.

## Preparing data for feeding to models.

M**aking dumies for location data as regression model can't work on strings**

![Untitled](Real%20Estate%20Price%20Predictor%20281b3d6e7c3c46b3bf04427a4a43b170/Untitled%204.png)

```python

df11 = pd.concat([df10, dummies.drop('other' , axis = 'columns')] , axis = "columns")

df12 = df11.drop("location" , axis= "columns")

```

```python
X = df12.drop('price' , axis= "columns")
Y= df12.price
```

Splitting the data into train and test sets.

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y , test_size = 0.2 , random_state = 10)
```

## Training Data and calculating score:

Trying linear regression first.

```python
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits = 5 , test_size =0.2 , random_state = 0)
cross_val_score(LinearRegression(), X , Y , cv = cv)
```

![Output](Real%20Estate%20Price%20Predictor%20281b3d6e7c3c46b3bf04427a4a43b170/Untitled%205.png)

Output

Our linear regression model scored around 83%.

Trying other models (linear regression , Lasso , Decision Tree) and comparing results using gridSearchCV

```python
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchCV(X , Y):
    algos = {
        'linear_regression' : {
            'model' : LinearRegression(),
            'params': {
                'normalize' : [True, False]  #grid CV also does hyper parameter tuning
            }
        },
        'lasso': {
            'model' : Lasso(),
            'params' : {
                'alpha' : [1,2],
                'selection' : ['random' , 'cyclic']
            }
        },
        'decision_tree': {
            'model' : DecisionTreeRegressor(),
            'params' : {
                'criterion' : ['mse' ,'friedman_mse'],
                'splitter' : ['best', 'random']
            }
        }
        
    }
    
    scores = []
    cv = ShuffleSplit(n_splits = 5 , test_size = 0.2 , random_state = 0)
    for algo_name , config in algos.items():
        gs = GridSearchCV(config['model'] , config['params'] , cv= cv , return_train_score= False)
        gs.fit(X, Y)
        scores.append({
            'model': algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        })
        return pd.DataFrame(scores , columns = {'model', 'best_score' , 'best_params'})

find_best_model_using_gridsearchCV(X,Y)
```

![Output](Real%20Estate%20Price%20Predictor%20281b3d6e7c3c46b3bf04427a4a43b170/Untitled%206.png)

Output

Out of three linear regression scored the best: 82.5%

Now using linear regression we predict prices.

```python
def predict_price(location , sqft , bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] =1
    return lr_clf.predict([x])[0]
```

```python
predict_price('Indira Nagar', 1000 ,3 ,3)
>>> 159.49972189323393

predict_price('1st Block Koramangala', 1000 ,2 ,2)
>>> 132.6375398339503

predict_price('1st Phase JP Nagar', 1000 ,2 ,2)
>>> 72.3298497049784
```

## Conclusion:

So from the results we can concluded, Data was highly scattered and the best model which was linear regression performed 83%. Still practical scores was accurate about 95%.

In this project:

- Cleaned data
- Visualised data
- Training model and calculated score
- Predict prices.
