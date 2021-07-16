from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import os
import torch
from transformers import BertTokenizerFast

DATA_DIRECTORY = os.path.join(os.getcwd(), 'models')
CACHE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'pretrained_models_cache')
CHECKPOINT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'trained_models')
MODEL = 'voidful/albert_chinese_tiny'
MAX_LEN = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class BertInferenceModel(torch.nn.Module):
    def __init__(self):
        super(BertInferenceModel, self).__init__()
        self.max_len = MAX_LEN
        self.device = DEVICE
        # 由於 albert_chinese_tiny 模型沒有用 sentencepiece，用AlbertTokenizer會載不進詞表，因此需要改用BertTokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(MODEL, cache_dir=CACHE_DIRECTORY, local_files_only=True)
        # 此model是pytorch模型的子类，可以保存成支持C++调用的形式
        self.bert = torch.load(f'{CHECKPOINT_DIRECTORY}/albert-tiny-zh-finetuned.pth').to(DEVICE)
        self.bert.eval()
        # self.bert = AlbertForSequenceClassification.from_pretrained(MODEL, cache_dir=CACHE_DIRECTORY, local_files_only=True, num_labels=4).to(DEVICE)
    def forward(self, sentence1, sentence2s):
        X = self.tokenizer.batch_encode_plus([(sentence1,sentence2) for sentence2 in sentence2s], max_length=self.max_len, padding='max_length', truncation='longest_first', return_tensors='pt')
        input_ids = X['input_ids'].to(self.device)
        token_type_ids = X['token_type_ids'].to(self.device)
        attention_mask = X['attention_mask'].to(self.device)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        _, results = torch.max(outputs[0], dim=1)
        return results


class Item(BaseModel):
    query: str
    docs: List[str]


app = FastAPI()

model = BertInferenceModel()

warmup_result = model('北京', ['北京', '上海'] * 8)
print(warmup_result)


@app.get("/")
async def test():
    return "lyj"
    
@app.post("/inference")
async def inference(item: Item):
    return model(item.query, item.docs).tolist()

if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=8080)