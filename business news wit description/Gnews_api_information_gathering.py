import json
import urllib.request
from datetime import datetime, timedelta
import time

apikey = ""  # Replace this with your API key
category = "business"

# Set the start and end date for the entire year
start_date = datetime(#year, #month, #day)
end_date = datetime(#year, #month, #day)  # Exclusive end date

while start_date < end_date:
    from_time = start_date.strftime('%Y-%m-%dT12:00:00Z')
    to_time = start_date.strftime('%Y-%m-%dT23:59:59Z')

    try:
        # Read existing content to avoid duplicates
        try:
            with open("Gnew_list.txt", "r", encoding="utf-8") as file_read:
                file_content = set(line.split(",")[0].strip() for line in file_read if line.strip())
        except FileNotFoundError:
            file_content = set()

        url = (
            f"https://gnews.io/api/v4/top-headlines?category={category}"
            f"&lang=en&country=in&max=50&from={from_time}&to={to_time}&apikey={apikey}"
        )

        news_desc = set()

        with urllib.request.urlopen(url) as response:
            status_code = response.getcode()
            if status_code == 200:
                data = json.loads(response.read().decode("utf-8"))
                articles = data.get("articles", [])

                for article in articles:
                    desc = article.get("description")
                    if desc:
                        desc = desc.strip()
                        print(f"Description: {desc}")
                        news_desc.add(desc)

                with open("Gnew_list.txt", "a", encoding="utf-8") as file:
                    for desc in news_desc:
                        if desc not in file_content:
                            file.write(desc + "," + start_date.strftime('%Y-%m-%d') + "\n")

            else:
                print(f"Error: Status code {status_code} on {start_date.strftime('%Y-%m-%d')}")

    except Exception as e:
        print(f"Exception occurred on {start_date.strftime('%Y-%m-%d')}: {e}")
        break

    # Wait 20 seconds between API calls to avoid rate limiting
    time.sleep(200)

    # Go to next day
    start_date += timedelta(days=1)
