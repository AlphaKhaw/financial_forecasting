import pandas as pd
import statsmodels.api as sm

class ForecastDataFrame:
    def __init__(self, path, metric):
        self.path = path
        self.metric = metric
        
    def read_and_preprocess_data(self):
        data = pd.read_excel(self.path)
        df = data.loc[data['Account Group']=='Profit & Loss'].drop(
            columns='Account Group').T
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        return df
    
    def get_forecast_df(self):
        df = self.read_and_preprocess_data()
        forecast_df = pd.DataFrame(df[self.metric])
        forecast_df = forecast_df.reset_index(drop=True).reset_index()
        forecast_df = forecast_df.rename_axis('', axis='columns').rename(columns={'index':'t'})
        return forecast_df

class Forecast(ForecastDataFrame):
    def __init__(self, path, metric):
        super().__init__(path, metric)
        
    def mrv(self, forecast_df):
        self.df = forecast_df
        last_date_period = self.df.index.max()
        forecasted_value = self.df[self.metric][last_date_period]
        print(f'The forecasted value using most recent value (MRV) is ${forecasted_value:,.0f}.')
        
    def average(self, forecast_df):
        self.df = forecast_df
        forecasted_value = self.df[self.metric].mean()
        print(f'The forecasted value using average is ${forecasted_value:,.0f}.')
    
    def get_intercept_beta_and_t(self, forecast_df):
        model = sm.OLS(forecast_df[self.metric].astype(float), 
                       sm.add_constant(forecast_df['t']), hasconst=True)
        results = model.fit()
        intercept = results.params['const'] 
        beta = results.params['t']
        t = len(forecast_df)
        return intercept, beta, t
    
    def regression(self, forecast_df, future_nperiod):
        intercept, beta, t = self.get_intercept_beta_and_t(forecast_df)
        self.df = forecast_df
        self.nperiod = future_nperiod
        t += self.nperiod
        forecasted_value = intercept + beta * t
        print(f'The forecasted value using regression (OLS) is ${forecasted_value:,.0f}.')
        
    def get_cagr(self, forecast_df):
        self.df = forecast_df 
        y_0 = self.df[self.metric].iloc[0]
        y_T = self.df[self.metric].iloc[-1]
        n = len(self.df)-1
        cagr = (y_T/y_0)**(1/n) - 1
        return cagr
    
    def cagr(self, forecast_df, future_nperiod):
        cagr = self.get_cagr(forecast_df)
        self.df = forecast_df
        self.nperiod = future_nperiod
        y_T = self.df[self.metric].iloc[-1]
        forecasted_value = y_T*(cagr+1)**self.nperiod
        print(f'The forecasted value using compound annual growth rate (CAGR) is ${forecasted_value:,.0f}.')