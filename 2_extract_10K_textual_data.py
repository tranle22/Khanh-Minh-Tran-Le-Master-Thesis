from sec_api import ExtractorApi
import re
import json
from constants import TICKERS, ITEM_SECTIONS


API_KEY = "7c93c8f319d956f0d5548306499f31d721bcbe482eabb0e051e85864862f954d"

extractorApi = ExtractorApi(API_KEY)

# Extract Filing URL
for ticker in TICKERS:
    print(f'Extract textual data for ticker {ticker}')
    with open(f"filing-metadata/{ticker}.json", "r") as f:
        filing_metadata = json.load(f)
    
    filings = filing_metadata['filings']
    filing_details_list = []
    for filing in filings:
        filing_details = {}
        filing_details['ticker'] = ticker
        filing_details['filedAt'] = filing['filedAt']
        filing_details['periodOfReport'] = filing.get('periodOfReport', '')
        filing_details['linkToTxt'] = filing['linkToTxt']
        filing_details['companyName'] = filing['companyName']
        filing_details['textual_data'] = {}
        print(f"Extract textual data for ticker {ticker} filed at {filing_details['filedAt']}")
        for item_section in ITEM_SECTIONS: # ExtractorApi only works with one section at a time
            print(f"Extract section {item_section}..")
            extracted_text = extractorApi.get_section(filing_details["linkToTxt"], item_section, "text")
            cleaned_extracted_text = re.sub(r"\n|&#[0-9]+;", "", extracted_text)
            filing_details['textual_data'][item_section] = cleaned_extracted_text
        filing_details_list.append(filing_details)
        
    with open(f"filing-textual-data/{ticker}.json", "w") as f:
        f.write(json.dumps(filing_details_list))
    