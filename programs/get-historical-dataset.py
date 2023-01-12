# import libraries
from datetime import date
import pandas as pd
import datetime
import time
import requests
import json

from selenium import webdriver
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.ui import WebDriverWait



# ----- scrape data from investing.com -----
# ----- run this program, daily every morning to get previous dataset -----


def get_crude_dataset():
        # load historical
        try:
                hist_crude = pd.read_csv("../historical-dataset/Crude Oil Prices.csv")
        except FileNotFoundError:
                hist_crude = pd.DataFrame()
        hist_crude["Date"] = pd.to_datetime(hist_crude["Date"])

        # hist_crude = pd.read_csv("../historical-dataset/Crude Oil Prices.csv")
        # hist_crude["Date"] = pd.to_datetime(hist_crude["Date"])

        # driver = webdriver.Chrome(ChromeDriverManager().install())
        driver = webdriver.Edge(EdgeChromiumDriverManager().install())
        driver.get("https://www.investing.com/commodities/crude-oil-historical-data")

        soup = BeautifulSoup(driver.page_source, 'lxml')
        tables = soup.find_all('table')

        dfs = pd.read_html(str(tables))
        new_crude = pd.DataFrame(dfs[1])

        driver.close()
        
        new_crude["Date"] = pd.to_datetime(new_crude["Date"])
        new_crude["Date"] = new_crude["Date"].dt.strftime('%Y-%m-%d')
        new_crude["Date"] = pd.to_datetime(new_crude["Date"])
        new_crude.sort_values(by="Date", inplace=True)
        new_crude = new_crude.reset_index(drop=True)

        concat_df = pd.concat([hist_crude, new_crude])
        concat_df.drop_duplicates('Date', inplace=True)
        
        return concat_df.to_csv("../historical-dataset/Crude Oil Prices.csv", index=False)


# call the function
get_crude_dataset()