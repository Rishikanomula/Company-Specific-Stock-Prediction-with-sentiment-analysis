import yfinance as yf
import pandas as pd
from datetime import date

def get_stock_data(ticker, start='2020-01-01', end=None):
    if end is None:
        end = date.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    data.reset_index(inplace=True)
    return data

if __name__ == "__main__":
    companies = ['RELIANCE.NS', 'INFY.NS', 'TCS.NS', 'HDFCBANK.NS']
    
    for company in companies:
        df = get_stock_data(company)
        df.to_csv(f'../data/{company.replace(".NS","")}_data.csv', index=False)
        print(f"{company} data saved up to {date.today()}")
