'''
@File    :   testing_distilbert.py
@Time    :   12/2024
@Author  :   nikifori
@Version :   -
'''
from transformers import DistilBertTokenizer, DistilBertModel


def main():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    text = "Replace me by any text you'd like. Eimai tasos."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input) # output["last_hidden_state"] --> tensor requiring grad
    print(1)


if __name__ == '__main__':
    main()