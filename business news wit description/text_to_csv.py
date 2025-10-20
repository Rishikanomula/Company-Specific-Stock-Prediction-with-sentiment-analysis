processed_rows = []
with open('Gnew_list_2024.txt', 'r', encoding='utf-8') as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            text, date = line.rsplit(',', 1)
            processed_rows.append([text, date])
        else:
            processed_rows.append([line, ''])  # In case no date is present

# To write to a new CSV file:
import csv
with open('Gnew_list_2024_processed.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['Text', 'Date'])
    writer.writerows(processed_rows)