"""Wikipedia is scraped to establish universes of equities, whose ticker information is
then saved to file.

The current 
"""

import requests
import bs4 as bs
import pandas as pd

resp = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find("table", {"class": "wikitable sortable"})

df = pd.DataFrame(columns=["symbol", "security", "GICS sector", "GICS sub-industry"])
for row in table.findAll("tr")[1:]:
    # Read in the Symbol, Security, GICS sector, GICS Sub-industry
    e = row.text.split("\n")
    df.loc[len(df), :] = [e[1], e[3], e[4], e[5]]

df.to_csv("sandp500.csv")
