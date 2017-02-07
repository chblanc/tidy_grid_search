# Tidy Grid Search with `pipelearner`
Carlos Blancarte  
`r format(Sys.Date(), "%B %d, %Y")`  

going off the tutorial on:
'https://drsimonj.svbtle.com/how-to-grid-search-with-pipelearner'  

`pipelearner` is a package for making machine learning pipelines, aiding
in the grid search of hyperparameters for a model. The example uses
a decision tree, so we'll start with that.



```r
# libraries
library(pipelearner)
library(tidyverse)
library(rpart)
library(scales)
```

##  The data

Begin my reading in the `mtcars`` dataset and converting the outcome
variable, `am` into a factor.


```r
dat <- mtcars %>%
  mutate(
    am = factor(am, labels = c('automatic', 'manual'))
    )

with(dat, table(am))
```

```
## am
## automatic    manual 
##        19        13
```

```r
head(dat)
```

```
##    mpg cyl disp  hp drat    wt  qsec vs        am gear carb
## 1 21.0   6  160 110 3.90 2.620 16.46  0    manual    4    4
## 2 21.0   6  160 110 3.90 2.875 17.02  0    manual    4    4
## 3 22.8   4  108  93 3.85 2.320 18.61  1    manual    4    1
## 4 21.4   6  258 110 3.08 3.215 19.44  1 automatic    3    1
## 5 18.7   8  360 175 3.15 3.440 17.02  0 automatic    3    2
## 6 18.1   6  225 105 2.76 3.460 20.22  1 automatic    3    1
```

### `pipelearning` object

Create a `pipelearner` object that uses the *default* hyperparameters
of the decision tree (parameters such as minspit, minbucket, cp, maxcompete,
maxsurrogate, etc).  

the *pipelearner* class puts everything into a neat list which includes the
data, `$data`, cross validation pairs, `train_ps` (not really sure what this
is), and a tibble of models (so the `models` param can take multiple models?)


```r
# pipelearner object
pl <- dat %>%
  pipelearner(data= . , 
              models = rpart, 
              formulas = am ~ .
  )
```

with the `pipelearner` object in place we can now fit the model using
`learn`:


```r
# fit the model
results <- pl %>%
  learn()
```

the `results` object holds a treasure trove of information in a compact and
tidy format. We have an individual id for the model, `cv_pairs`, the actual
model object `fit`, and includes the model type, the params used to fit the
model, and training and testing datasets. 




### model results

in order to assess the results of the model we'll need to write a function
to assess the results given a tidy results object.


```r
# function to compute accuracy
accuracy <- function(fit, data, target_var) {
  
  # coerce to data.frame, predict output, and take the mean
  data <- as.data.frame(data)
  predicted <- predict(fit, data, type = 'class')
  mean(predicted == data[[target_var]])
  
}
```

now we can apply the function in order to assess model accuracy in both
training and testing:


```
## training accuracy: 92%
```

```
## testing accuracy: 85.7%
```


---
title: "tidy_grid_search.R"
author: "carlosblancarte"
date: "Sun Feb  5 09:57:49 2017"
---
