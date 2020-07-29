import logging
import azure.functions as func
from transformers import pipeline
from transformers import BertTokenizer, BertModel
import json

unmasker = pipeline('fill-mask', model='bert-base-cased')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    text = req.params.get('text')
    if not text:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            text = req_body.get('text')

    if text:
        encoded_input = tokenizer(text, return_tensors='pt')
        return func.HttpResponse(json.dumps({ "tokenizer": unmasker(text) }, ensure_ascii=False), mimetype="application/json")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass text in the query string or in the request body for a personalized response.",
             status_code=200
        )
