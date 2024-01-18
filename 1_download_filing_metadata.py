from sec_api import QueryApi
import json
from constants import TICKERS

API_KEY = "7c93c8f319d956f0d5548306499f31d721bcbe482eabb0e051e85864862f954d"

queryApi = QueryApi(api_key=API_KEY)

START_DATE = "2010-01-01"
END_DATE = '2019-12-31'

for ticker in TICKERS:
    print(f'Download filings for {ticker}')
    query = {
        "query": { "query_string": { 
            "query": f"ticker:{ticker} AND filedAt:[{START_DATE} TO {END_DATE}] AND formType:\"10-K\"",
            "time_zone": "America/New_York"
        } },
        "from": "0",
        "size": "10000",
        "sort": [{ "filedAt": { "order": "asc" } }]
    }

    response = queryApi.get_filings(query)    
    # response.keys: total, query, filings
    
    print(f'Ticker {ticker} has {response["total"]} filings.')
    
    with open(f"filing-metadata/{ticker}.json", "w") as f:
        f.write(json.dumps(response))





