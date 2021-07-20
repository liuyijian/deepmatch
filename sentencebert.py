import sentence_transformers
from sentence_transformers import SentenceTransformer, util, InputExample, losses
import torch
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
import time
import faiss
import numpy as np
import math
from collections import defaultdict
from itertools import chain, product
import multiprocessing
import pickle

DATA_DIRECTORY = os.path.join(os.getcwd(), 'models')
CACHE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'pretrained_models_cache')
CHECKPOINT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'trained_models')
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
MAX_LEN = 128
EPOCHS = 1


def finetune():


    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder=CACHE_DIRECTORY)

    train_samples = []

    # with open('visible_test_data.txt', 'r') as f:
    #     content = [eval(line) for line in f.readlines()]

    with open('/mnt/datadisk0/visible_test_data.pkl', 'rb') as f_in:
        content = pickle.load(f_in)

    query_dict = defaultdict(dict)
    query_set = set()

    for item in content:
        query_set.add(item["query"])
        query_dict[item["query"]] = {'0': [], '1':[], '2':[], '3':[]}

    for item in content:
        query_dict[item["query"]][str(item["label"])].append((item["title"] + '@' + item["summary"])[:MAX_LEN-2])

    for query in query_set:
        query_info = query_dict[query]
        for positive_text, negative_text in chain(product(query_info['3'], query_info['0']), product(query_info['3'], query_info['1']), product(query_info['2'], query_info['0'])):
            train_samples.append(InputExample(texts=[query, positive_text, negative_text]))


    train_dataloader = sentence_transformers.datasets.NoDuplicatesDataLoader(train_samples, batch_size=TRAIN_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # test_samples = [InputExample(texts=[ line['query'], line['summary']], label=float(line['label']) / 3) for line in content]
    # test_evaluator = sentence_transformers.evaluation.EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=TEST_BATCH_SIZE, name='testEvaluator')
    # test_evaluator = sentence_transformers.evaluation.TripletEvaluator.from_input_examples(test_samples, batch_size=TEST_BATCH_SIZE, name='testEvaluator')

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs = EPOCHS,
        warmup_steps=math.ceil(len(train_dataloader) * EPOCHS * 0.1),
        use_amp=True,
        # evaluator=None,
        # evaluation_steps=int(len(train_dataloader)*0.1),
        output_path=f'/mnt/datadisk0/sentencebert-all-multinegativesrankingloss-title+summary-finetuned.pth',
        checkpoint_path=f'/mnt/datadisk0/sentencebert-all-multinegativesrankingloss-title+summary-ckpt.pth',
        checkpoint_save_steps=1000,
        checkpoint_save_total_limit=1
    )

def load_all_docs_to_faiss_index(model, build_index, trained):

    if build_index:
        print('start reading file')
        with open('visible_test_data.txt', 'r') as f:
            docs = [eval(line)['summary'] for line in f.readlines()]

        print('start encoding')
        T1 = time.perf_counter()
        # pool = model.start_multi_process_pool()
        # docs_split = [ docs[i: i + 320] for i in range(0, int(len(docs)/100), 320)]

        # with multiprocessing.Pool() as pool:
        #     results = pool.map(worker, docs_split)

        # print(results)
        # print(len(results))
        # print(results[0])

        docs_embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

        T2 = time.perf_counter()
        print(T2-T1)
        faiss_index = faiss.IndexFlatIP(384)  # vector dimension 归一化后的点积等价于余弦相似度
        faiss_index.add(docs_embeddings)
        faiss.write_index(faiss_index, '/mnt/datadisk0/170w-summary-finetuned.index' if trained else '/mnt/datadisk0/170w-summary.index')
        T3 = time.perf_counter()
        print(T3-T2)

    else:
        faiss_index = faiss.read_index('/mnt/datadisk0/170w-summary-finetuned.index' if trained else '/mnt/datadisk0/170w-summary.index')
        faiss_index.ntotal
    
    return faiss_index, faiss_index.ntotal



# class Item(BaseModel):
#     query: List[str]

# model1 = SentenceTransformer('./models/trained_models/sentencebert-full-tripletloss-finetuned.pth')
# # model2 = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_folder=CACHE_DIRECTORY)
# faiss_index1, docs_count1 = load_all_docs_to_faiss_index(model=model1, build_index=False, trained=True)
# # faiss_index2, docs_count2 = load_all_docs_to_faiss_index(model=model2, build_index=False, trained=False)

# # with open('visible_test_data.txt', 'r') as f:
# #     docs = [eval(line) for line in f.readlines()]

# # with open('/mnt/datadisk0/visible_test_data.pkl', 'wb') as f_out:
# #     pickle.dump(docs, f_out)

# with open('/mnt/datadisk0/visible_test_data.pkl', 'rb') as f_in:
#     docs = pickle.load(f_in)

# app = FastAPI()

# @app.post("/inference")
# async def inference(item: Item):
#     T_start = time.perf_counter()

#     query_embedding1 = model1.encode(item.query, convert_to_numpy=True, normalize_embeddings=True)
#     distance1, index1 = faiss_index1.search(query_embedding1, min(docs_count1, 10))

#     # query_embedding2 = model2.encode(item.query, convert_to_numpy=True, normalize_embeddings=True)
#     # distance2, index2 = faiss_index2.search(query_embedding2, min(docs_count2, 10))
    
#     T_end = time.perf_counter()
#     print(f'inference time: {1000 * (T_end - T_start) } ms') # 20ms 17224文本
#     result = list(zip([docs[k]['title'] for k in index1.tolist()[0]], [docs[k]['summary'] for k in index1.tolist()[0]], distance1.tolist()[0]))
#     return result

if __name__ == '__main__':
    finetune()
    # uvicorn.run(app=app, host="0.0.0.0", port=8080)