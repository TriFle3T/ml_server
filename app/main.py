from flask import Flask, jsonify,request
from flask_restx import Resource,Api
import random
import logging
import torch
import gluonnlp as nlp
from utils.predict import BERTpredict

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from utils.classifier import BERTClassifier


app = Flask(__name__)


@app.route("/analyze",methods=["POST"])
def analyzeDiary():
    data = request.get_json(silent=True, cache=False, force=True)
    return analyze(data["content"][0])


def analyze(content):

    result = model_predict.predict(content)
    print(result)
    
    return {
        "happy" : result[0],
        "angry" : result[1],
        "disgust" : result[2],
        "fear" : result[3],
        "neutral" : result[4],
        "sad" : result[5],
        "surprise" : result[6],
        "resultIndex" : result[7],
        "quoteIndex" : result[8]
    }


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
tok = tokenizer.tokenize
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

## Setting parameters
max_len = 64
batch_size = 64
model = BERTClassifier(bertmodel,  dr_rate=0.5).to(device)
model.load_state_dict(torch.load('./my_path/model.pth',map_location=device))
model_predict = BERTpredict(model,tok,vocab,max_len,batch_size,device)

application = app  

if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port = 80)