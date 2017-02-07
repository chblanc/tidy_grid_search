#' ---
#' title: Tidy Grid Search with `pipelearner`
#' author: Carlos Blancarte
#' date: '`r format(Sys.Date(), "%B %d, %Y")`'
#' output:
#'  html_document:
#'    keep_md: true
#'    theme: lumen
#'    highlight: kate
#' ---
#' 

#' going off the tutorial on:
#' 'https://drsimonj.svbtle.com/how-to-grid-search-with-pipelearner'  
#' 
#' `pipelearner` is a package for making machine learning pipelines, aiding
#' in the grid search of hyperparameters for a model. The example uses
#' a decision tree, so we'll start with that.
#' 

#+ message = FALSE, warning = FALSE
# libraries
library(pipelearner)
library(tidyverse)
library(rpart)
library(scales)

#' ##  The data
#' 
#' Begin my reading in the `mtcars`` dataset and converting the outcome
#' variable, `am` into a factor.

dat <- mtcars %>%
  mutate(
    am = factor(am, labels = c('automatic', 'manual'))
    )

with(dat, table(am))
head(dat)

#' ### `pipelearning` object
#' 
#' Create a `pipelearner` object that uses the *default* hyperparameters
#' of the decision tree (parameters such as minspit, minbucket, cp, maxcompete,
#' maxsurrogate, etc).  
#' 
#' the *pipelearner* class puts everything into a neat list which includes the
#' data, `$data`, cross validation pairs, `train_ps` (not really sure what this
#' is), and a tibble of models (so the `models` param can take multiple models?)

# pipelearner object
pl <- dat %>%
  pipelearner(data= . , 
              models = rpart, 
              formulas = am ~ .
  )

#' with the `pipelearner` object in place we can now fit the model using
#' `learn`:

# fit the model
results <- pl %>%
  learn()

#' the `results` object holds a treasure trove of information in a compact and
#' tidy format. We have an individual id for the model, `cv_pairs`, the actual
#' model object `fit`, and includes the model type, the params used to fit the
#' model, and training and testing datasets. 
#' 
#+ include=F

#' ### model results
#' 
#' in order to assess the results of the model we'll need to write a function
#' to assess the results given a tidy results object.

# function to compute accuracy
accuracy <- function(fit, data, target_var) {
  
  # coerce to data.frame, predict output, and take the mean
  data <- as.data.frame(data)
  predicted <- predict(fit, data, type = 'class')
  mean(predicted == data[[target_var]])
  
}

# function to output accuracy
printAccuracy <- function(results) {
  
  cat('training accuracy: ',
      percent(with(results, accuracy(first(fit), first(train), first(target)))),
  '\n',
  
  'testing accuracy: ',
      percent(with(results, accuracy(first(fit), first(test), first(target)))),
  sep=""
  )
  
}

#' now we can apply the function in order to assess model accuracy in both
#' training and testing:
#+ echo=FALSE
printAccuracy(results)

#' ### adding hyperparameters
#' 
#' adding hyperparameters into the pipelearner is as easy as adding
#' the parameters after the model formula (or however they're included
#' in the implementation of the package).  
#' 
#' In this example we'll search of various levels of `minsplit`, which
#' controls the minimum number of observations that must exist in a node
#' in order for a split to be attempted.

results <- dat %>%
  pipelearner(data= . , 
              models = rpart, 
              formulas = am ~ .,
              minsplit= seq(1, 10, 1)) %>%
  learn()

#' a few notes about what's going on here:
#' using `map_dbl` searches within the list, `params`, and extracts the
#' values of `minsplit` as a dbl. The second portion uses `pmap_dbl` to
#' iteratie over multiple functions - in this case, we're passing all of the
#' info we need to compute the accuracy. 

results <- results %>% 
  mutate(
    minsplit = map_dbl(params, "minsplit"),
    accuracy_train = pmap_dbl(list(fit, train, target), accuracy),
    accuracy_test  = pmap_dbl(list(fit, test,  target), accuracy)
  )

#' ### adding multiple hyperparameters

results <- dat %>%
  pipelearner(data= . , 
              models = rpart, 
              formulas = am ~ .,
              minsplit= seq(1, 20, 2),
              maxdepth = seq(2, 5, 1),
              xval = seq(5, 10, 5)) %>%
  learn()

results <- results %>% 
  mutate(
    minsplit = map_dbl(params, "minsplit"),
    maxdepth = map_dbl(params, "maxdepth"),
    xval = map_dbl(params, "xval"),
    accuracy_train = pmap_dbl(list(fit, train, target), accuracy),
    accuracy_test  = pmap_dbl(list(fit, test,  target), accuracy)
  )
