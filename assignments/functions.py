# default imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


# data exploration
# 2.0 Visual inspection using lineplots
def get_visualisation_plots(df:pd.DataFrame, figsize=(7,10), style='default'):
    ncharts = len(df.columns)
    if ncharts%2 > 0:
        rows = int(ncharts/2 + 0.5)
    else:
        rows = int(ncharts/2)
    with plt.style.context(style=style):
        plt.rcParams["font.size"] = 7
        plt.rcParams['lines.linewidth'] = 0.3
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1f77b4", "#e377c2", "#2ca02c", "#bcbd22"])
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(nrows=rows, ncols=2)
        feat = 0
        for row in range(rows):
            for col in range(2):
                ax = fig.add_subplot(grid[row,col])
                ax.tick_params(axis='x', labelrotation=90)
                sns.lineplot(df, x="date", y=df.columns[feat], color="lightgray", ax=ax)
                sns.lineplot(df[df.columns[feat]].rolling(window=7).mean(), color="#1f77b4", ax=ax)
                ax.fill_between(df.index, df[df.columns[feat]].rolling(window=7).mean() - df[df.columns[feat]].rolling(window=7).std(), df[df.columns[feat]].rolling(window=7).mean() + df[df.columns[feat]].rolling(window=7).std(), color="#e377c2", alpha=0.3)
                if feat < ncharts - 1:
                    feat += 1
                else:
                    break
    plt.show()

# 2.1 Check feature correlations
def get_feature_correlations(df:pd.DataFrame, figsize=(5,4)):
    df_matrix = df.corr().round(2)
    sns.set(font_scale=0.6)
    plt.figure(figsize=(5,4))
    sns.heatmap(df_matrix, annot=True, cmap='Blues', linecolor='white', linewidth=0.5)
    plt.show()

# 2.2 Check time series for seasonality
def get_seasonal_plots(dataset:pd.Series, figsize=(7,6), style='default'):
    df_seasonal = pd.DataFrame(dataset).copy()
    df_seasonal['day_of_year'] = pd.DatetimeIndex(dataset.index).dayofyear
    df_seasonal['week_of_year'] = pd.DatetimeIndex.isocalendar(dataset.index).week
    df_seasonal['month_of_year'] = pd.DatetimeIndex(dataset.index).month
    df_seasonal['year'] = pd.DatetimeIndex(dataset.index).year

    if not isinstance(dataset, pd.Series):
        dataset = pd.Series(dataset)
    with plt.style.context(style=style):
        plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 7})
        plt.rcParams['lines.linewidth'] = 0.5
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1f77b4", "#e377c2", "#2ca02c", "#bcbd22"])
        ticks = [
            [30,60,90,120,150,180,210,240,270,300,330,360],
            [4,8,12,16,20,24,28,32,36,40,44,48,52],
            [1,2,3,4,5,6,7,8,9,10,11,12]
        ]
        labels = [
            [30,60,90,120,150,180,210,240,270,300,330,360],
            ['Wk_4', 'Wk_8', 'Wk_12', 'Wk_16', 'Wk_20', 'Wk_24', 'Wk_28', 'Wk_32', 'Wk_36', 'Wk_40', 'Wk_44', 'Wk_48', 'Wk_52'],
            ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ]
        plots = {}
        for i,period in enumerate(df_seasonal.iloc[:,1:4]):
            plots[period] = pd.pivot_table(df_seasonal, values=dataset.name, index=period, columns='year', aggfunc=np.sum)
            plt.subplot(3,1,i+1)
            plt.plot(plots[period], label=plots[period].columns)
            plt.grid(axis="x")
            plt.xticks(ticks=ticks[i], labels=labels[i])
            plt.title(f"Feature: {dataset.name}\nYearly seasonal chart ({period})")
            plt.legend()
            plt.xlabel(period)
        plt.tight_layout(h_pad=1.0)
        plt.show()

# 2.3 Decompose seasonality
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def get_decompose_seasonality(dataset, model='additive', period=None, figsize=(12,14), style='default'):
    if not isinstance(dataset, pd.Series):
        dataset = pd.Series(dataset)
    decompose = seasonal_decompose(dataset, model=model, period=period)
    if model == "multiplicative":
        seasonally_adj = (dataset/decompose.seasonal).astype(int)
    else:
        seasonally_adj = (dataset - decompose.seasonal).astype(int)
    data = {}
    datasets = {'og': {'data': dataset}, 
                'sa': {'data': seasonally_adj}}
    for series in datasets:
        results = {}
        adf = adfuller(datasets[series]['data'])
        results['statistic'] = adf[0]
        results['p_value'] = adf[1]
        results['critical'] = adf[4]['5%']
        if results['p_value'] > 0.05:
            results['p_stationarity'] = "time series is NOT stationary"
        else:
            results['p_stationarity'] = "time series is stationary"
        if abs(results['critical']) > abs(results['statistic']):
            results['c_stationarity'] = "time series is NOT stationary"
        else:
            results['c_stationarity'] = "time series is stationary"
        datasets[series]['results'] = results
    ljungbox = sm.stats.acorr_ljungbox(decompose.resid.fillna(0), return_df=True)['lb_pvalue'].iloc[0]
    if ljungbox > 0.05:
        independent = "residuals are NOT independent"
    else:
        independent = "residuals are independent"
    with plt.style.context(style=style):
        fig = plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 7})
        plt.rcParams['lines.linewidth'] = 0.5
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1f77b4", "#e377c2", "#2ca02c", "#bcbd22"])
        layout = (7,2)                    
        observed_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        seasonally_adj_ax = plt.subplot2grid(layout, (1,0), colspan=2)
        trend_ax = plt.subplot2grid(layout, (2,0), colspan=2)
        seasonal_ax = plt.subplot2grid(layout, (3,0), colspan=2)
        resid_ax = plt.subplot2grid(layout, (4,0), colspan=2)
        acf_orig_ax = plt.subplot2grid(layout, (5,0), rowspan=2)
        pacf_orig_ax = plt.subplot2grid(layout, (5,1), rowspan=2)

        decompose.observed.plot(xlabel="", ylabel='observed', ax=observed_ax)
        observed_ax.set_title(f"Feature: {dataset.name}\nTime Series Seasonality Plots (model: {model}, period: {period})\n Dickey-Fuller: p={datasets['og']['results']['p_value']:.5f} ({datasets['og']['results']['p_stationarity']})\n ADF Statistic: {datasets['og']['results']['statistic']:.5f} vs Critical Value (5%): {datasets['og']['results']['critical']:.5f} ({datasets['og']['results']['c_stationarity']})".format())
        decompose.trend.plot(xlabel="", ylabel='trend', ax=trend_ax)
        decompose.seasonal.plot(xlabel="", ylabel='seasonal', ax=seasonal_ax)
        seasonally_adj.plot(xlabel="", ylabel='seasonally_adj', ax=seasonally_adj_ax)
        seasonally_adj_ax.set_title(f"Dickey-Fuller: p={datasets['sa']['results']['p_value']:.5f} ({datasets['sa']['results']['p_stationarity']})\n ADF Statistic: {datasets['sa']['results']['statistic']:.5f} vs Critical Value (5%): {datasets['sa']['results']['critical']:.5f} ({datasets['sa']['results']['c_stationarity']})".format())
        decompose.resid.plot(xlabel="", ylabel='residuals', ax=resid_ax)
        resid_ax.set_title(f"Residuals:\n Ljung-Box: p={ljungbox:.5g} ({independent})".format())
        plot_acf(dataset, title='Observed ACF', ax=acf_orig_ax)
        plot_pacf(dataset, method='ywm', title='Observed PACF', ax=pacf_orig_ax)
        plt.tight_layout()
        plt.show()

# 2.4 Find seasonality period using autocorrelation plots
from pandas.plotting import autocorrelation_plot
import pmdarima as pmd

def get_various_seasonality(dataset, model='additive', periods=[7,30,120,365], figsize=(8,6), style='default'):
    ncharts = len(periods)
    if ncharts%2 > 0:
        rows = int(ncharts/2 + 0.5)
    else:
        rows = int(ncharts/2)
    if not isinstance(dataset, pd.Series):
        dataset = pd.Series(dataset)
    with plt.style.context(style=style):
        plt.rcParams["font.size"] = 7
        plt.rcParams['lines.linewidth'] = 0.5
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1f77b4", "#e377c2", "#2ca02c", "#bcbd22"])
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        grid = fig.add_gridspec(nrows=rows, ncols=2)
        num_periods = 0
        for row in range(rows):
            for col in range(2):
                ax = fig.add_subplot(grid[row,col])
                decompose = seasonal_decompose(dataset, model=model, period=periods[num_periods])
                if model == 'multiplicative':
                    deseason = (decompose.observed / decompose.seasonal).astype(int)
                else:
                    deseason = (decompose.observed - decompose.seasonal).astype(int)
                autocorrelation_plot(deseason.tolist(), ax=ax)
                ax.set_title(f"Feature: {dataset.name}\nModel: {model}\n Period: {periods[num_periods]}")
                if num_periods < ncharts - 1:
                    num_periods += 1
                else:
                    break
        plt.show()

# 2.5 Transform to stationary data using differencing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def get_adf(dataset, adf_alpha=0.05):
    adf = adfuller(dataset)
    adf_statistic = adf[0]
    p_value = adf[1]
    critical_value = adf[4][str(int(adf_alpha*100))+"%"]
    if p_value > adf_alpha:
        p_stationarity = "time series is NOT stationary"
    else:
        p_stationarity = "time series is stationary"
    if adf_statistic > critical_value:
        adf_stationarity = "time series is NOT stationary"
    else:
        adf_stationarity = "time series is stationary"
    return adf_statistic, p_value, critical_value, p_stationarity, adf_stationarity

def get_moving_avg(dataset, difference=7):
    df_mva = pd.DataFrame(dataset.copy())
    df_mva.columns = [dataset.name]
    df_mva[f"{difference}-day moving_avg"] = dataset.rolling(difference).mean()
    return df_mva

def get_various_stationarity(dataset, differencing=[1], figsize=(8,4), deseasonalise=False, model='additive', style='default'):
    dataset = pd.Series(dataset, name=dataset.name)
    if deseasonalise:
        decompose = seasonal_decompose(dataset, model=model)
        name = dataset.name
        seasonal_adjustment = "adjusted for seasonality"
        if model == "multiplicative":
            dataset = (dataset/decompose.seasonal).astype(int)
        else:
            dataset = (dataset - decompose.seasonal).astype(int)
        dataset.name = name
    else:
        seasonal_adjustment = "unadjusted for seasonality"

    with plt.style.context(style=style):
        fig = plt.figure(figsize=figsize)
        plt.rcParams.update({'font.size': 7})
        plt.rcParams['lines.linewidth'] = 0.5
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#1f77b4", "#e377c2", "#2ca02c", "#bcbd22"])
        layout = ((1+len(differencing))*2,2)

        statistic, p_value, critical, p_stationarity, c_stationarity = get_adf(dataset)
        ax_og = plt.subplot2grid(layout, (0,0), colspan=2)
        ax_acf = plt.subplot2grid(layout, (1,0))
        ax_pacf = plt.subplot2grid(layout, (1,1))
        og = get_moving_avg(dataset)
        og.plot(ax=ax_og)
        ax_og.set_title(f"Feature: {dataset.name}\nTime Series Stationarity Plots ({seasonal_adjustment})\n Dickey-Fuller: p={p_value:.5f} ({p_stationarity})\n ADF Statistic: {statistic:.5f} vs Critical Value (5%): {critical:.5f} ({c_stationarity})".format())
        plot_acf(dataset, ax=ax_acf)
        plot_pacf(dataset, method='ywm', ax=ax_pacf)
        
        for i in range(len(differencing)):
            ax_diff = plt.subplot2grid(layout, (i*2+2,0), colspan=2)
            ax_diff_acf = plt.subplot2grid(layout, (i*2+3,0))
            ax_diff_pacf = plt.subplot2grid(layout, (i*2+3,1))
            
            dataset_diff = dataset.diff(differencing[i]).dropna()
            og_diff = get_moving_avg(dataset_diff, difference=differencing[i])
            og_diff.plot(ax=ax_diff)
            adf_statistic, p_value, critical_value, p_stationarity, adf_stationarity = get_adf(dataset_diff, adf_alpha=0.05)
            ax_diff.set_title(f"Feature: {dataset.name}\nDifferencing Plots (differencing = {differencing[i]})\n Dickey-Fuller: p={p_value:.5f} ({p_stationarity})\n ADF Statistic: {adf_statistic:.5f} vs Critical Value (5%): {critical_value:.5f} ({adf_stationarity})".format())    
            plot_acf(dataset_diff, ax=ax_diff_acf)
            plot_pacf(dataset_diff, method='ywm', ax=ax_diff_pacf)
        plt.tight_layout()
        plt.show()

# graph color key 
# print(plt.rcParams['axes.prop_cycle'].by_key()['color'])


# utilities
def plot(y_test_pred, test_series, train_series, service, filename, lower=None, upper=None):
    rmse = mean_squared_error(test_series, y_test_pred, squared=False)
    mape = mean_absolute_percentage_error(test_series, y_test_pred)
    
    # Plot
    fig, axs = plt.subplots(2,1)
    ax1, ax2 = axs
    for ax in axs:
        ax.plot(test_series, color="blue", label="actual")
        ax.plot(y_test_pred, color="red", label="predicted")
        if lower is not None and upper is not None:
            ax.fill_between(y_test_pred.index, 
                            lower, 
                            upper, 
                            color="k", alpha=.15)
    train_series.plot.line(color="blue")
    ax1.plot([], [], ' ', label=f"RMSE: {rmse}")
    ax1.plot([], [], ' ', label=f"MAPE: {mape}")
    ax1.legend()
    filename = service + "_" + filename
    ax1.set_title(filename)
    plt.savefig(filename + ".png")
    plt.show()
    return mape


# ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ARIMA
def get_X_y(target, df_train):
    X_train = df_train.drop(target, axis='columns')
    y_train = df_train[target]
    return X_train, y_train

def do_arima(service, dataset, model_type, target, seasonal, m=1, scaled=False, X_forecasts=None, features=None):
    
    train_X, train_y = get_X_y(target, dataset['df_train'])
    test_X, test_y = get_X_y(target, dataset['df_test'])
    if X_forecasts is not None:
        train_X = train_X[features]
        if scaled:
            scaler_X = MinMaxScaler()
            train_X = scaler_X.fit_transform(train_X)
            scaler_y = MinMaxScaler()
            train_y = scaler_y.fit_transform(train_y.to_frame())
        model = auto_arima(train_y, X=train_X, trace=True, seasonal=seasonal, m=m, suppress_warnings=True)
        forecasts, conf_int = model.predict(n_periods=len(dataset['test_period']), X=X_forecasts, return_conf_int=True)
    else:
        if scaled:
            scaler_y = MinMaxScaler()
            train_y = scaler_y.fit_transform(train_y.to_frame())
        model = auto_arima(train_y, X=None, trace=True, seasonal=seasonal, m=m, suppress_warnings=True)
        forecasts, conf_int = model.predict(n_periods=len(dataset['test_period']), X=None, return_conf_int=True)
    model.summary()
    if scaled:
        scaled_forecasts = pd.Series(forecasts.flatten(), index=dataset['test_period'], name=f"scaled_{model_type}_forecasts")
        train_y = pd.Series(scaler_y.inverse_transform(train_y).flatten(), index=dataset['train_period'], name=target).astype(int)
        forecasts = pd.Series(scaler_y.inverse_transform(scaled_forecasts.values.reshape(-1,1)).flatten(), index=dataset['test_period'], name=f"scaled_{model_type}_forecasts").astype(int)
        lower = pd.Series(scaler_y.inverse_transform(conf_int[:,0].reshape(-1, 1)).flatten(), index=dataset['test_period']).astype(int)
        upper = pd.Series(scaler_y.inverse_transform(conf_int[:,1].reshape(-1, 1)).flatten(), index=dataset['test_period']).astype(int)
    else:        
        forecasts = pd.Series(forecasts.astype(int), index=dataset['test_period'], name=f"{model_type}_forecasts")
        lower = pd.Series(conf_int[:,0], index=dataset['test_period'])
        upper = pd.Series(conf_int[:,1], index=dataset['test_period'])        
    
    mape = plot(forecasts, test_y, train_y, service, model_type.upper(), lower, upper)
    if scaled:
        return scaled_forecasts, forecasts, mape
    else:
        return forecasts, mape


# machine learning
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

# machine learning
def get_best_model(model, n_splits:int, param_grid:dict, scoring:str):
    tscv = TimeSeriesSplit(n_splits)
    best_model = GridSearchCV(estimator=model,
                                cv=tscv,
                                param_grid=param_grid,
                                scoring=scoring)
    return best_model
    
def do_ml(service, dataset, selected_model, target, n = 28, additional_features=True, grid_search=False):
    df_train = dataset['df_train'].copy()
    df_test = dataset['df_test'].copy()
    feats = [col for col in df_train.columns if col != target]

    # create distributed lags for each feature
    columns_to_lag = [target]
    for col_name in columns_to_lag:        
        for lag in range(1, n + 1):
            col_lag = f"{col_name}_lag{lag}" 
            df_train[col_lag] = df_train[col_name].shift(lag)
    if additional_features:
        window = 30
        df_train.insert(0, target+"_rolling_avg", df_train[target].rolling(window=window, closed='left').mean())
        df_train.insert(0, target+"_month_of_year", df_train.index.month)
        df_train.insert(0, target+"_week_of_year", df_train.index.isocalendar().week)
        df_train.insert(0, target+"_day_of_week", df_train.index.dayofweek)
    df_train = df_train.dropna().astype(int) # drop rows with missing values
    
    # Separate features and target
    train_X = df_train.drop(columns=[target]+feats)
    train_y = df_train[target]
    test_y = df_test[target]

    if grid_search:
        model = get_best_model(selected_model['model'], 
                                n_splits=n,
                                param_grid=selected_model['param_grid'],
                                scoring='neg_mean_squared_error')
        model.fit(train_X, train_y) # train best model
        print(model.best_params_) 
    else:
        model = selected_model['model'] # create selected model
        model.fit(train_X, train_y) # train model

    # Make recursive predictions
    X_t = train_X.loc[:,'sales_lag1':].tail(1)
    X_t = X_t.shift(periods=1, axis="columns")
    X_t['sales_lag1'] = train_y.tail(1)

    if additional_features:
        rolling_sales_calc = train_y.tail(window + 1)
        pred_date = df_test.index[0]
        X_t.insert(0, target+"_rolling_avg", rolling_sales_calc.tail(window).mean().astype(int))
        X_t.insert(0, target+"_month_of_year", pred_date.month) 
        X_t.insert(0, target+"_week_of_year", pred_date.week)
        X_t.insert(0, target+"_day_of_week", pred_date.dayofweek)

    forecasts = pd.Series(dtype=int)
    for t in range(len(df_test)):
        y_t = int(model.predict(X_t))  # Predict next step
        forecasts[df_test.index[t]] = y_t  # Store prediction
        X_t = X_t.loc[:,'sales_lag1':].shift(periods=1, axis="columns")
        X_t['sales_lag1'] = y_t
        if additional_features:
            rolling_sales_calc[df_test.index[t]] = y_t
            X_t.insert(0, target+"_rolling_avg", rolling_sales_calc.tail(window).mean().astype(int))
            X_t.insert(0, target+"_month_of_year", df_test.index[t].month) 
            X_t.insert(0, target+"_week_of_year", df_test.index[t].week)
            X_t.insert(0, target+"_day_of_week", df_test.index[t].dayofweek)
    plot(forecasts, test_y, train_y, service, f"{selected_model['name']}_additional_features_{additional_features}_grid_search_{grid_search}_lag{n}")


# darts
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.models import ARIMA
from darts.utils.statistics import check_seasonality
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler

# convert darts array to series
def darts_to_series(dates, darts_times_series):
    series = pd.Series(darts_times_series.values().flatten(), index=dates)
    return series

def do_darts_arima(service, target, train_series, test_series, train_period, test_period, model_type, m=None, scaled=False, exogenous=None, exogenous_features=None):
    if m:
        model = AutoARIMA(random_state=42, m=m, trace=True)
    else:
        model = AutoARIMA(random_state=42, trace=True)
    if scaled:
        scaler = MinMaxScaler(feature_range=(0, 1))
        transformer = Scaler(scaler)
        series = transformer.fit_transform(train_series[target])
        if exogenous:
            scaler_covariates = MinMaxScaler(feature_range=(0, 1))
            transformer_covariates = Scaler(scaler_covariates)
            covariates = transformer_covariates.fit_transform(exogenous[exogenous_features])
            model.fit(series=series, future_covariates=covariates)
            scaled_forecasts = model.predict(n=31, future_covariates=covariates)
        else:
            model.fit(series=series)
            scaled_forecasts = model.predict(n=31)
        forecasts = transformer.inverse_transform(scaled_forecasts)
        train_series = transformer.inverse_transform(series)
    else:
        if exogenous:
            model.fit(series=train_series[target], future_covariates=exogenous[exogenous_features])
            forecasts = model.predict(n=31, future_covariates=exogenous[exogenous_features])
        else:
            model.fit(series=train_series[target])
            forecasts = model.predict(n=31)
            
    mape = plot(darts_to_series(test_period, forecasts), darts_to_series(test_period, test_series[target]), darts_to_series(train_period, train_series[target]), service, model_type)
    return forecasts, mape

def get_future_covariates_autoarima(train_data, test_data, train_period, test_period, features, seasonality, scaled=False):
    future_covariates = None
    for i, feat in enumerate(features):
        if scaled:
            forecasts, mape = do_darts_arima(service, train_data[feat], test_data[feat], train_period, test_period, model_type=f"darts_sarima_scaled_{feat}", m=seasonality[i], scaled=True)
        else:
            forecasts, mape = do_darts_arima(service, train_data[feat], test_data[feat], train_period, test_period, model_type=f"darts_sarima_{feat}", m=seasonality[i], scaled=False)
        if isinstance(future_covariates, TimeSeries):
            future_covariates = future_covariates.stack(train_data[feat].append(forecasts))
        else:
            future_covariates = train_data[feat].append(forecasts)
    return future_covariates

def get_future_covariates_arima(params, train_data, test_data, train_period, test_period, features, seasonality, scaled=False):
    future_covariates = None
    for i, feat in enumerate(features):
        model = ARIMA(*params[i])
        if scaled:
            scaler = MinMaxScaler(feature_range=(0, 1))
            transformer = Scaler(scaler)
            series = transformer.fit_transform(train_data[feat])
        else:
            series = train_data[feat]            
        model.fit(series)
        predictions = model.predict(n=31)
        test_series = test_data[feat]
        if scaled:
            forecasts = transformer.inverse_transform(predictions)
            train_series = transformer.inverse_transform(series)
            mape = plot(darts_to_series(test_period, forecasts), darts_to_series(test_period, test_series), darts_to_series(train_period, train_series), "A", f"darts_sarima_scaled_{feat}")
        else:
            forecasts = predictions
            train_series = series
            mape = plot(darts_to_series(test_period, forecasts), darts_to_series(test_period, test_series), darts_to_series(train_period, train_series), "A", f"darts_sarima_{feat}")
        if isinstance(future_covariates, TimeSeries): 
            future_covariates = future_covariates.stack(train_series.append(forecasts))
        else:
            future_covariates = train_series.append(forecasts)
    return future_covariates

from darts.models import ExponentialSmoothing

def do_darts_stats_model(service, stats_model, target, train_series, test_series, train_period, test_period, model_type, scaled=False):    
    if scaled:
        scaler = MinMaxScaler(feature_range=(0, 1))
        transformer = Scaler(scaler)
        series = transformer.fit_transform(train_series[target])
    else:
        series = train_series[target]

    model = stats_model
    model.fit(series)
    forecasts = model.predict(n=31)

    if scaled:
        forecasts = transformer.inverse_transform(forecasts)
        train_series = transformer.inverse_transform(series)        
    plot(darts_to_series(test_period, forecasts[target]), darts_to_series(test_period, test_series[target]), darts_to_series(train_period, train_series[target]), service, model_type)

from darts.models import RegressionModel
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

def do_darts_regressor(service, regressor_model, lags, target, features, train_series, test_series, train_period, test_period, model_type, scaled=False, exogenous=None, exogenous_lags=None):
    if exogenous:
        if scaled:
            scaler = MinMaxScaler(feature_range=(0,1))
            transformer = Scaler(scaler)
            series = transformer.fit_transform(train_series[target])
            
            scaler_covariates = MinMaxScaler(feature_range=(0,1))
            transformer_covariates = Scaler(scaler_covariates)
            future_covariates = transformer_covariates.fit_transform(exogenous[features])
        else:
            series = train_series[target]
            future_covariates = exogenous[features]

        model = RegressionModel(
            lags=lags,
            lags_future_covariates=exogenous_lags,
            model=regressor_model,
            output_chunk_length=max(abs(num) for num in exogenous_lags)+1
        )
        model.fit(series, future_covariates=future_covariates)
        forecasts = model.predict(n=31, series=series, future_covariates=future_covariates)
    
        if scaled:
            forecasts = transformer.inverse_transform(forecasts)
            train_series = transformer.inverse_transform(series)
    else:
        if scaled:
            scaler = MinMaxScaler(feature_range=(0,1))
            transformer = Scaler(scaler)
            series = transformer.fit_transform(train_series[features].stack(train_series[target]))
        else:
            series = train_series[features].stack(train_series[target])

        model = RegressionModel(
            lags=lags,
            model=regressor_model,
            output_chunk_length=1
        )
        model.fit(series=series)
        forecasts = model.predict(n=31, series=series)
        if scaled:
            forecasts = transformer.inverse_transform(forecasts)
            train_series = transformer.inverse_transform(series)

    plot(darts_to_series(test_period, forecasts[target]), darts_to_series(test_period, test_series[target]), darts_to_series(train_period, train_series[target]), service, model_type)

