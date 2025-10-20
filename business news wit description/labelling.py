from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd

model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

data = pd.read_csv("name of the file")
#data = data.drop("Unnamed: 2",axis =1)

l = []
for i in data["Text"]:
    headline = i
    result = nlp(headline)
    #print(result[0]["label"])
    l.append(result[0]["label"])
data["label"] = l

data.to_csv("name of the file",index = False)