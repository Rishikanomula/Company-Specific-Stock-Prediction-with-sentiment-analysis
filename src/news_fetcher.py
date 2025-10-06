# import requests
# import pandas as pd
# from datetime import datetime, timedelta

# def fetch_news(company, api_key, days=7):
#     to_date = datetime.today().strftime('%Y-%m-%d')
#     from_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

#     # ✅ Use GNews API
#     url = (
#         f"https://gnews.io/api/v4/search?"
#         f"q={company}&lang=en&country=in&max=50&"
#         f"from={from_date}&to={to_date}&token={api_key}"
#     )

#     response = requests.get(url)
#     data = response.json()

#     if 'articles' not in data:
#         print(f"❌ Error fetching data for {company}:", data)
#         return None

#     articles = data.get('articles', [])
#     df = pd.DataFrame([{
#         'date': a['publishedAt'][:10],
#         'headline': a['title'],
#         'source': a['source']['name'],
#         'url': a['url'],
#         'company': company  # ✅ Add company column
#     } for a in articles])

#     print(f"✅ Fetched {len(df)} articles for {company}")
#     return df

# if __name__ == "__main__":
#     API_KEY = "c4d63d1a15afbed83577555e7bbc26ee"  
#     companies = ["Reliance Industries", "Infosys", "TCS", "HDFC Bank"]

#     all_news = pd.DataFrame()

#     for company in companies:
#         df = fetch_news(company, API_KEY)
#         if df is not None and not df.empty:
#             all_news = pd.concat([all_news, df], ignore_index=True)

#     output_file = r"C:\Rishika\SPP\data\live_news.csv"
#     all_news.to_csv(output_file, index=False)
#     print(f"\n📰 All company news saved to {output_file}")




# import requests
# import pandas as pd
# from datetime import datetime, timedelta
# import time

# def fetch_news(company, api_key, days=7, max_articles=100):
#     to_date = datetime.today().strftime('%Y-%m-%d')
#     from_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

#     url = (
#         f"https://gnews.io/api/v4/search?"
#         f"q={company}&lang=en&country=in&max={min(max_articles, 100)}&"
#         f"from={from_date}&to={to_date}&token={api_key}"
#     )

#     try:
#         response = requests.get(url)
#         data = response.json()
#     except Exception as e:
#         print(f"❌ Request error for {company}: {e}")
#         return None

#     if 'articles' not in data or not data['articles']:
#         print(f"⚠️ No articles found for {company}")
#         return None

#     articles = data['articles']
#     df = pd.DataFrame([{
#         'date': a.get('publishedAt', '')[:10],
#         'headline': a.get('title', ''),
#         'source': a.get('source', {}).get('name', ''),
#         'url': a.get('url', ''),
#         'company': company
#     } for a in articles])

#     print(f"✅ Fetched {len(df)} articles for {company}")
#     return df

# if __name__ == "__main__":
#     API_KEY = "c4d63d1a15afbed83577555e7bbc26ee"  
#     companies = ["TCS"]

#     all_news = pd.DataFrame()

#     for company in companies:
#         df = fetch_news(company, API_KEY, days=7, max_articles=100)
#         if df is not None and not df.empty:
#             all_news = pd.concat([all_news, df], ignore_index=True)
#         else:
#             print(f"⚠️ Skipping {company}, no articles fetched")
#         time.sleep(1)  # ✅ small delay to avoid API rate limits

#     output_file = r"C:\Rishika\SPP\data\tcs_live_news.csv"
#     all_news.to_csv(output_file, index=False)
#     print(f"\n📰 All company news saved to {output_file}")



import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def fetch_news(company, api_key, days=30, max_articles=100):
    to_date = datetime.today().strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={company}&"
        f"from={from_date}&to={to_date}&"
        f"language=en&"
        f"sortBy=publishedAt&"
        f"pageSize={min(max_articles, 100)}&"
        f"apiKey={api_key}"
    )

    try:
        response = requests.get(url)
        data = response.json()
    except Exception as e:
        print(f"❌ Request error for {company}: {e}")
        return None

    if data.get('status') != 'ok' or not data.get('articles'):
        print(f"⚠️ No articles found for {company}: {data.get('message', '')}")
        return None

    articles = data['articles']
    df = pd.DataFrame([{
        'date': a.get('publishedAt', '')[:10],
        'headline': a.get('title', ''),
        'source': a.get('source', {}).get('name', ''),
        'url': a.get('url', ''),
        'company': company
    } for a in articles])

    print(f"✅ Fetched {len(df)} articles for {company}")
    return df

if __name__ == "__main__":
    API_KEY = "22e06e0e0d8d4bf6bf55e4303e836b9b"  # Replace with your NewsAPI key
    companies = ["TCS"]

    all_news = pd.DataFrame()

    for company in companies:
        df = fetch_news(company, API_KEY, days=30, max_articles=100)
        if df is not None and not df.empty:
            all_news = pd.concat([all_news, df], ignore_index=True)
        else:
            print(f"⚠️ Skipping {company}, no articles fetched")
        time.sleep(1)  # small delay to avoid API rate limits

    output_file = r"C:\Rishika\SPP\data\tcs_live_news.csv"
    all_news.to_csv(output_file, index=False)
    print(f"\n📰 All company news saved to {output_file}")
