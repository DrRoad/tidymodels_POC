
library(tidymodels)
library(tidyverse)
library(lime)
library(tune)


# Data --------------------------------------------------------------------

data("gss_cat")
raw.data <- gss_cat %>%
  tidyr::drop_na() %>%
  dplyr::filter(marital != 'No answer') %>%
  dplyr::filter(marital != 'Separated') %>%
  droplevels()


# RSample -----------------------------------------------------------------

set.seed(1)
train_test_split <- rsample::initial_split(raw.data)
train <- rsample::training(train_test_split)
test <- rsample::testing(train_test_split)


# Recipes -----------------------------------------------------------------

marital.recipe <- recipes::recipe(marital ~ tvhours + age + relig, data = train) %>%
  recipes::step_center(recipes::all_numeric()) %>%
  recipes::step_scale(recipes::all_numeric()) %>%
  recipes::step_dummy(relig, one_hot = T) %>%
  recipes::prep()


train.data <- recipes::bake(marital.recipe, train)
test.data <- recipes::bake(marital.recipe, test)

# Parsnip -----------------------------------------------------------------

rf.model <- parsnip::rand_forest(mtry = 6, trees = 1530 , mode = "classification") %>%
  parsnip::set_engine("randomForest") %>%
  parsnip::fit(marital ~ ., 
               data = train.data)

rf.unfitted <- parsnip::rand_forest(mtry = tune::tune(), trees = tune::tune() ,mode = "classification") %>%
  parsnip::set_engine("randomForest")

predictions <- parsnip::predict.model_fit(object = rf.model, new_data = test.data)

results <- test.data %>%
  dplyr::bind_cols(predictions)

# Yardstick ---------------------------------------------------------------

yardstick::metrics(data = results, truth = 'marital', estimate = '.pred_class')
yardstick::precision(data = results, truth = 'marital', estimate = '.pred_class')
yardstick::conf_mat(results, marital, .pred_class)


# Tune --------------------------------------------------------------------

# resample bootstrap
set.seed(1)
resam.train <- rsample::bootstraps(train.data, times = 20)

roc.vals <- yardstick::metric_set(roc_auc)

ctrl <- tune::control_grid(verbose = FALSE)

# grid search
# set.seed(1)
# grid_form <- tune::tune_grid(marital ~ .,
#                             model = rf.unfitted,
#                             resamples = resam.train,
#                             metrics = roc.vals
# )
#                           
# tune::show_best(grid_form)


# Workflows ---------------------------------------------------------------



# Lime --------------------------------------------------------------------

# create explainer object
explainer <- lime::lime(train.data, rf.model)

# explain new obs
explanation <- lime::explain(test.data[1:3,], explainer, n_labels = 1, n_features = 40)

lime::plot_features(explanation)
lime::plot_explanations(explanation)
