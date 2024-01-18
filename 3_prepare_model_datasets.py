import pandas as pd
from constants import TICKERS, ITEM_SECTIONS

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)


RESULT_FOLDER = 'clean_data'

#import word list of economy
with open('sustainability_word_lists/Table S2 economy_word_list.txt', 'r') as file:
  economy_words = file.read()
  economy_words = [each.lower() for each in economy_words.split("\n")]
  
#import word list of environoment
with open('sustainability_word_lists/Table S3 environment_word_list.txt', 'r') as file:
  environment_words = file.read()
  environment_words = [each.lower() for each in environment_words.split("\n")]

#import word list of social
with open('sustainability_word_lists/Table S4 social_word_list.txt', 'r') as file:
  social_words = file.read()
  social_words = [each.lower() for each in social_words.split("\n")]
  
def extract_sentences(textual_data_dict):
    filing_sentences = []
    for section in ITEM_SECTIONS:
        section_text = textual_data_dict[section].lower()
        sentences = sent_tokenize(section_text)
        filing_sentences.extend(sentences)
    return filing_sentences

def check_if_sentence_contains_environment_word(sentence_in_lower_case):
    for word in environment_words:
        if word in sentence_in_lower_case:
            return True
        
def check_if_sentence_contains_social_word(sentence_in_lower_case):
    for word in social_words:
        if word in sentence_in_lower_case:
            return True
        
def check_if_sentence_contains_economy_word(sentence_in_lower_case):
    for word in economy_words:
        if word in sentence_in_lower_case:
            return True

def check_if_sentence_contains_csr_word(sentence_in_lower_case):
    if check_if_sentence_contains_environment_word(sentence_in_lower_case):
        return True
    if check_if_sentence_contains_social_word(sentence_in_lower_case):
        return True
    if check_if_sentence_contains_economy_word(sentence_in_lower_case):
        return True
    return False
    
def calculate_csr_sentences_and_sentiment_scores(df):
    filing_sentences = df.iloc[0]['sentences']
    filedAt = df.iloc[0]['filedAt']
    ticker = df.iloc[0]['Ticker']
    
    environment_sentences = [sentence for sentence in filing_sentences if check_if_sentence_contains_environment_word(sentence)]
    social_sentences = [sentence for sentence in filing_sentences if check_if_sentence_contains_social_word(sentence)]
    economy_sentences = [sentence for sentence in filing_sentences if check_if_sentence_contains_economy_word(sentence)]
    csr_sentences = [sentence for sentence in filing_sentences if check_if_sentence_contains_csr_word(sentence)]
    
    num_positive = 0
    positive_score = 0
    num_negative = 0
    negative_score = 0
    
    num_environment_positive = 0
    environment_positive_score = 0
    num_environment_negative = 0
    environment_negative_score = 0
    
    num_social_positive = 0
    social_positive_score = 0
    num_social_negative = 0
    social_negative_score = 0
    
    num_economy_positive = 0
    economy_positive_score = 0
    num_economy_negative = 0
    economy_negative_score = 0
    
    num_failed_sentences = 0
    count = 0
    for sentence in filing_sentences:
        count += 1
        if (count % 50 == 0):
            print(f'{ticker} filing at {filedAt} - Calculated CSR sentences and sentiments for {count}/{len(filing_sentences)} sentences')
        try:
            finbert_response = nlp(sentence)[0]
            
            
            if finbert_response['label'] == 'Positive': 
                # Overall
                if sentence in csr_sentences:
                    positive_score += finbert_response['score']
                    num_positive += 1
                # Enviroment
                if sentence in environment_sentences:
                    environment_positive_score += finbert_response['score']
                    num_environment_positive += 1
                # Social    
                if sentence in social_sentences:
                    social_positive_score += finbert_response['score']
                    num_social_positive += 1
                # Economy
                if sentence in economy_sentences:
                    economy_positive_score += finbert_response['score']
                    num_economy_positive += 1
                
            if finbert_response['label'] == 'Negative': 
                # Overall
                if sentence in csr_sentences:
                    negative_score += finbert_response['score']
                    num_negative += 1
                # Enviroment
                if sentence in environment_sentences:
                    environment_negative_score += finbert_response['score']
                    num_environment_negative += 1
                # Social    
                if sentence in social_sentences:
                    social_negative_score += finbert_response['score']
                    num_social_negative += 1
                # Economy
                if sentence in economy_sentences:
                    economy_negative_score += finbert_response['score']
                    num_economy_negative += 1
                
            
        except:
            num_failed_sentences += 1
            pass
    print(f'The number of failed sentences: {num_failed_sentences}')
    return (
        # Overall
        len(csr_sentences), # 0
        num_positive, # 1
        num_negative, # 2
        positive_score, # 3
        negative_score, # 4
        (positive_score - negative_score) / (num_positive + num_negative) if (num_positive + num_negative) > 0 else 0, # 5
        
        # Enviroment
        len(environment_sentences), # 6
        num_environment_positive, # 7
        num_environment_negative, # 8
        environment_positive_score, # 9
        environment_negative_score, # 10
        (environment_positive_score - environment_negative_score) / (num_environment_positive + num_environment_negative) 
            if (num_environment_positive + num_environment_negative) > 0 else 0, # 11
        
        # Social
        len(social_sentences), # 12
        num_social_positive, # 13
        num_social_negative, # 14
        social_positive_score, # 15
        social_negative_score, # 16
        (social_positive_score - social_negative_score) / (num_social_positive + num_social_negative) 
            if (num_social_positive + num_social_negative) > 0 else 0, # 17
        
        # Economy
        len(economy_sentences), # 18
        num_economy_positive, # 19
        num_economy_negative, # 20
        economy_positive_score, # 21
        economy_negative_score, # 22
        (economy_positive_score - economy_negative_score) / (num_economy_positive + num_economy_negative) 
            if (num_economy_positive + num_economy_negative) > 0 else 0, # 23
                
        num_failed_sentences,  # 24
        
        
    )
    
def calculate_future_return(df):
    # Offset is used to account for cases where no price available on filing date
    offset = 0
    while len(df[df.days_from_filing==offset]) == 0:
        offset += 1
    
    price_on_filing_date = df[df.days_from_filing == offset].Price.values[0]
    
    return_df = pd.DataFrame(columns = [*df.columns, 'benchmark', 'return'])
    for benchmark in [0, 1, 7, 30, 60]:
        days = benchmark + offset
        while not (days in df.days_from_filing.values):
            days += 1
        extracted = df[df.days_from_filing == days]
        extracted['benchmark'] = benchmark
        extracted['return'] = (extracted.Price - price_on_filing_date) / price_on_filing_date
        extracted['price_on_filing_date'] = price_on_filing_date
        return_df = pd.concat([return_df, extracted], axis = 0)
    return_df = return_df.drop(["filedAt"], axis = 1)
    return return_df


def calculate_historical_n_day_return(df, n):
    # Offset is used to account for cases where no price available on filing date
    offset = 0
    while len(df[df.days_from_filing==offset]) == 0:
        offset += 1
        
    price_on_filing_date = df[df.days_from_filing == offset].Price.values[0]
    
    # If stock price not available at n days, go back one day at a time to find the closest price
    days = n + offset
    while not(days in df.days_from_filing.values):
        days -= 1
    df = df[df.days_from_filing == days]
    
    df[f'hist_{abs(n)}_day_date'] = df.Date
    df[f'hist_{abs(n)}_day_price'] = df.Price
    df[f'hist_{abs(n)}_day_return'] = (price_on_filing_date - df.Price) / df.Price
    df['price_on_filing_date'] = price_on_filing_date # To merge back to the original Data Frame
    df = df.reset_index()
    df = df[[
        'Ticker',
        'filedAt',
        'price_on_filing_date',
        f'hist_{abs(n)}_day_date',
        f'hist_{abs(n)}_day_price',
        f'hist_{abs(n)}_day_return'
    ]]
    return df


for ticker in TICKERS:
    print(f'Computing features and targets for ticker {ticker}')
    # Read stock data downloaded from Yahoo Finance
    stock_data = pd.read_csv(f"daily_stock_price/{ticker}.csv")
    stock_data['Ticker'] = ticker
    stock_data['Price'] = stock_data['Adj Close']
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data[[
        'Ticker',
        'Date',
        'Price'
    ]]
    
    # Read filling textual data
    report_text_data = pd.read_json(f"filing-textual-data/{ticker}.json")
    tzname = pd.to_datetime(report_text_data.filedAt.iloc[0]).tzname() # E.g. 'UTC-05:00'
    report_text_data['filedAt'] = pd.to_datetime(report_text_data.filedAt).astype(f'datetime64[ns, {tzname}]')
    report_text_data = report_text_data.rename({'ticker': 'Ticker'}, axis = 1)
    report_text_data = report_text_data[[
        'Ticker',
        'filedAt',
        'periodOfReport',
        'linkToTxt',
        'companyName',
        'textual_data'
    ]]
    
    # Merge texttual data and stock data with a cross join.
    # This means that one filing row will be joined with many stock price rows.
    merged_df = pd.merge(left=report_text_data, right=stock_data, on=['Ticker'], indicator=True)    
    merged_df = merged_df.sort_values('filedAt')
    # days_from_filing: The number of days between the stock-price-date and the filing-date
    merged_df['days_from_filing'] = pd.to_timedelta(merged_df.Date.dt.date - merged_df.filedAt.dt.date).dt.days

    # Future stock price and return
    benchmark_df = merged_df.groupby('filedAt').apply(calculate_future_return).reset_index()
    
    # Historical stock price and return
    minus_1_day_df = merged_df.groupby('filedAt').apply(lambda group_df: calculate_historical_n_day_return(group_df, -1)).reset_index(drop=True)
    minus_7_day_df = merged_df.groupby('filedAt').apply(lambda group_df: calculate_historical_n_day_return(group_df, -7)).reset_index(drop=True)
    minus_30_day_df = merged_df.groupby('filedAt').apply(lambda group_df: calculate_historical_n_day_return(group_df, -30)).reset_index(drop=True)
    minus_60_day_df = merged_df.groupby('filedAt').apply(lambda group_df: calculate_historical_n_day_return(group_df, -60)).reset_index(drop=True)
    benchmark_df = (
        benchmark_df
            .merge(minus_1_day_df, on=['Ticker', 'filedAt', 'price_on_filing_date'])
            .merge(minus_7_day_df, on=['Ticker', 'filedAt', 'price_on_filing_date'])
            .merge(minus_30_day_df, on=['Ticker', 'filedAt', 'price_on_filing_date'])
            .merge(minus_60_day_df, on=['Ticker', 'filedAt', 'price_on_filing_date'])
    )
    
    # Extract sentences from textual data
    benchmark_df['sentences'] = benchmark_df['textual_data'].apply(extract_sentences)
    benchmark_df['num_sentences'] = benchmark_df.sentences.apply(len)
    
    # Calculate number of CSR sentences and sentiment scores
    csr_sentences_and_sentiment_scores_df = benchmark_df.groupby(['Ticker', 'filedAt']).apply(calculate_csr_sentences_and_sentiment_scores).reset_index()
    benchmark_df = benchmark_df.merge(csr_sentences_and_sentiment_scores_df, on=['Ticker', 'filedAt'])
    # Overall
    benchmark_df['num_csr_sentences'] = benchmark_df[0].apply(lambda x: x[0])
    benchmark_df['num_positive_sentences'] = benchmark_df[0].apply(lambda x: x[1])
    benchmark_df['num_negative_sentences'] = benchmark_df[0].apply(lambda x: x[2])
    benchmark_df['sum_positive_score'] = benchmark_df[0].apply(lambda x: x[3])
    benchmark_df['sum_negative_score'] = benchmark_df[0].apply(lambda x: x[4])
    benchmark_df['sentiment_score'] = benchmark_df[0].apply(lambda x: x[5])
    # Enviroment
    benchmark_df['num_environment_sentences'] = benchmark_df[0].apply(lambda x: x[6])
    benchmark_df['num_positive_environment_sentences'] = benchmark_df[0].apply(lambda x: x[7])
    benchmark_df['num_negative_environment_sentences'] = benchmark_df[0].apply(lambda x: x[8])
    benchmark_df['sum_environment_positive_score'] = benchmark_df[0].apply(lambda x: x[9])
    benchmark_df['sum_environment_negative_score'] = benchmark_df[0].apply(lambda x: x[10])
    benchmark_df['environment_sentiment_score'] = benchmark_df[0].apply(lambda x: x[11])
    # Social
    benchmark_df['num_social_sentences'] = benchmark_df[0].apply(lambda x: x[12])
    benchmark_df['num_positive_social_sentences'] = benchmark_df[0].apply(lambda x: x[13])
    benchmark_df['num_negative_social_sentences'] = benchmark_df[0].apply(lambda x: x[14])
    benchmark_df['sum_social_positive_score'] = benchmark_df[0].apply(lambda x: x[15])
    benchmark_df['sum_social_negative_score'] = benchmark_df[0].apply(lambda x: x[16])
    benchmark_df['social_sentiment_score'] = benchmark_df[0].apply(lambda x: x[17])
    # Economy
    benchmark_df['num_economy_sentences'] = benchmark_df[0].apply(lambda x: x[18])
    benchmark_df['num_positive_economy_sentences'] = benchmark_df[0].apply(lambda x: x[19])
    benchmark_df['num_negative_economy_sentences'] = benchmark_df[0].apply(lambda x: x[20])
    benchmark_df['sum_economy_positive_score'] = benchmark_df[0].apply(lambda x: x[21])
    benchmark_df['sum_economy_negative_score'] = benchmark_df[0].apply(lambda x: x[22])
    benchmark_df['economy_sentiment_score'] = benchmark_df[0].apply(lambda x: x[23])


    benchmark_df = benchmark_df[[
        'Ticker', 'filedAt', 'price_on_filing_date', 
        'num_sentences', 'num_csr_sentences',
        'num_positive_sentences', 'num_negative_sentences', 'sum_positive_score', 'sum_negative_score', 'sentiment_score',
        'num_environment_sentences', 'num_positive_environment_sentences', 'num_negative_environment_sentences', 
            'sum_environment_positive_score', 'sum_environment_negative_score', 'environment_sentiment_score',
        'num_social_sentences', 'num_positive_social_sentences', 'num_negative_social_sentences',
            'sum_social_positive_score', 'sum_social_negative_score', 'social_sentiment_score',
        'num_economy_sentences', 'num_positive_economy_sentences', 'num_negative_economy_sentences', 
            'sum_economy_positive_score', 'sum_economy_negative_score', 'economy_sentiment_score',
        'hist_1_day_date', 'hist_1_day_price', 'hist_1_day_return', 
        'hist_7_day_date', 'hist_7_day_price', 'hist_7_day_return',
        'hist_30_day_date', 'hist_30_day_price', 'hist_30_day_return',
        'hist_60_day_date', 'hist_60_day_price', 'hist_60_day_return',
        'Price', 'Date', 'benchmark', 'days_from_filing', 'return'
    ]]
    
    benchmark_df.to_csv(f'{RESULT_FOLDER}/{ticker}.csv', index=False)    



# Put all data into ALL_TICKERS_{benchmark}_DAYS.csv for ease of use
for benchmark in [1, 7, 30, 60]:
    df = pd.DataFrame(columns = ['Ticker', 'filedAt', 'price_on_filing_date', 
        'num_sentences', 'num_csr_sentences', 
       'num_positive_sentences', 'num_negative_sentences', 'sum_positive_score', 'sum_negative_score','sentiment_score', 
       'num_environment_sentences', 'num_positive_environment_sentences', 'num_negative_environment_sentences', 
            'sum_environment_positive_score', 'sum_environment_negative_score', 'environment_sentiment_score',
        'num_social_sentences', 'num_positive_social_sentences', 'num_negative_social_sentences', 
            'sum_social_positive_score', 'sum_social_negative_score', 'social_sentiment_score',
        'num_economy_sentences', 'num_positive_economy_sentences', 'num_negative_economy_sentences', 
            'sum_economy_positive_score', 'sum_economy_negative_score', 'economy_sentiment_score',
       'hist_1_day_date', 'hist_1_day_price', 'hist_1_day_return', 
       'hist_7_day_date', 'hist_7_day_price', 'hist_7_day_return', 
       'hist_30_day_date', 'hist_30_day_price', 'hist_30_day_return', 
       'hist_60_day_date', 'hist_60_day_price', 'hist_60_day_return', 
       'Price', 'Date', 'benchmark', 'days_from_filing', 'return'])
    for ticker in TICKERS:
        temp = pd.read_csv(f"./{RESULT_FOLDER}/{ticker}.csv")
        temp = temp[temp.num_sentences > 100]
        temp = temp[temp.num_csr_sentences > 0]
        temp = temp[temp.benchmark == benchmark]
        df = pd.concat([df, temp])
    df.to_csv(f"./{RESULT_FOLDER}/ALL_TICKERS_{benchmark}_DAYS.csv", index=False)
        