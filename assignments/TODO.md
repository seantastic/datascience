How to test the predictions of each individual feature used in the predicted SARIMAX model?
Can I exclude a feature based in the SARIMA results? or the results compared against the df_test values? (i,e, future values?)

df_train (training): - Oct
df_test (validation): Nov
    * can be used to select the model (type of model + hyperparameters + feature selection)
final test: Dec

# Coding philosophy
* functions are only for when you repeat code with different values (parameters)
* code repetition can often be avoided with other techniques -- reorganisation, loops, OOP
* can organise code into files / functions to reduce file size and make it easier to navigate

# Data exploration

* view/handle missing data
* view correlation (would want to remove correlated features above a certain threshold for correlation coefficient)

# ARIMA

* screen a target/feature to see if it's random
* visualisation to determine seasonality of each target/feature
* ARIMA function
    * target vs individual feature
    * scaled always better
    * ARIMA vs SARIMA (can choose seasonality)
        * ARIMA: 7.7% error
        * SARIMA: 3.5% error
    * SARIMAX
        * with test_X (future leakage) all : 1.7% error
        * with predicted features from SARIMA (seasonalities [7,7,-,30,30,30] excluding C): 6.5% error

* open questions
    * is scaled ARIMA / SARIMA better than unscaled?
    * does including a random feature in test_X improve SARIMAX?
        * run SARIMAX with test_X -- all features vs features left out?
    * does including a poorly predicted feature from SARIMA improve SARIMAX?
        * run SARIMAX with predicted features -- all features vs with random/poor (percentage error >10%) features left out
    * for the above: does improving the predicitons for predicted_test_X (ie, including or excluding features) improve overall SARIMAX?

# ML: manual recursive multistep forecasting

* manual recursive multistep forecasting
    * chosen target (not including features A-F)
    * target lag 
    * additional features
    * grid search with time series cross validation

* results
    * MLP, lag 1-28, additional features, no grid search: 4-6%
    * DT, lag 1-28, additional features, no grid search: 6-8%

* open questions
    * is there an error with grid search?
        * with a fixed random_state, grid search with default parameters should give the same result as just MLP with default parameters
    * can we get better results with grid search?
        * small param grid to speed up investigation
        * can also try downloading and running on own machine to run faster
        * try getting best values for one parameter at a time before combining        
    * will scaling / normalisation improve results?
    * would DT give better results?


# Darts
* 1. as a starting point, code in Darts what we've already done for ML manually
    * hook up the output of Darts to the plotting function we have
    * use MLP with the same random_state as before -- should give same results

* a. try adding additional_features as before -- should give same results
* b. try to do recursive multistep forecasting with features like in the video