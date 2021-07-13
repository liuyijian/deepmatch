from transformers import BertTokenizer, BertTokenizerFast, BertConfig
from transformers import AlbertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
import torch
import numpy as np
import os
import time
import pickle

DATA_DIRECTORY = os.path.join(os.getcwd(), 'models')
MODEL = 'voidful/albert_chinese_tiny'
CACHE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'pretrained_models_cache')
CHECKPOINT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'trained_models')
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 256
MAX_LEN = 128
EPOCHS = 1
WEIGHT_DECAY = 0.01
LRATE = 2e-5
LOG_INTERVAL = 100
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

# 由於 albert_chinese_tiny 模型沒有用 sentencepiece，用AlbertTokenizer會載不進詞表，因此需要改用BertTokenizer
tokenizer = BertTokenizerFast.from_pretrained(MODEL, cache_dir=CACHE_DIRECTORY)
# 此model是pytorch模型的子类，可以保存成支持C++调用的形式
model = AlbertForSequenceClassification.from_pretrained(MODEL, cache_dir=CACHE_DIRECTORY, num_labels=4)
model.to(DEVICE)
print(model)

# 保存成支持C++调用的模型

# example = torch.rand(1, 3, 224, 224)
# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
# traced_script_module = torch.jit.trace(model, example)
# Once you have a ScriptModule in your hands, either from tracing or annotating a PyTorch model, you are ready to serialize it to a file. 
# Later on, you’ll be able to load the module from this file in C++ and execute it without any dependency on Python. 
# traced_script_module.save('traced_albert_chinese_tiny.pt')



def generate_data_loaders(load):

    if load == True:
        with open('train_dataloader.pkl', 'rb') as f1:
            train_dataloader = pickle.load(f1)
        
        with open('test_dataloader.pkl', 'rb') as f2:
            test_dataloader = pickle.load(f2)
    else:
        with open('query.txt.ground_truth', 'r') as f:
            lines = [eval(line) for line in f.readlines()] # 变成字典
        
        # 自动构造[CLS] SEQ_A [SEP] SEQ_B [SEP] 的输入格式 
        # https://blog.csdn.net/qq_34418352/article/details/106069042
        # https://huggingface.co/transformers/internal/tokenization_utils.html?highlight=encode_plus#transformers.tokenization_utils_base.PreTrainedTokenizerBase.encode_plus
        # 返回一个字典 d
        # d['input_ids'] == [101, x..., 102, y..., 102]
        # d['token_type_ids'] = [0,..., 0, 1, ...,1]
        # d['attention_mask'] = [1,1,1, ..., 0] pad的为0
        X = [ tokenizer.encode_plus(line['query'], line['summary'], max_length=MAX_LEN, padding='max_length', truncation='longest_first', return_tensors=None) for line in lines]
        X = [ [dic['input_ids'], dic['token_type_ids'], dic['attention_mask']] for dic in X]
        Y = [ int(line['label']) for line in lines ]
        
        # stratify=Y,使训练集中Y的分布和全量数据中Y的分布一致，用于类分布不平衡的情况，不指定则类标签比例随机
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y) 

        X_train_input_ids = torch.tensor([item[0] for item in X_train], dtype=torch.long)
        X_train_token_type_ids = torch.tensor([item[1] for item in X_train], dtype=torch.long)
        X_train_attention_mask = torch.tensor([item[2] for item in X_train], dtype=torch.long)
        Y_train = torch.tensor(Y_train, dtype=torch.long)

        X_test_input_ids = torch.tensor([item[0] for item in X_test], dtype=torch.long)
        X_test_token_type_ids = torch.tensor([item[1] for item in X_test], dtype=torch.long)
        X_test_attention_mask = torch.tensor([item[2] for item in X_test], dtype=torch.long)
        Y_test = torch.tensor(Y_test)

        train_dataset = TensorDataset(X_train_input_ids, X_train_token_type_ids, X_train_attention_mask, Y_train)
        test_dataset = TensorDataset(X_test_input_ids, X_test_token_type_ids, X_test_attention_mask, Y_test)
        
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=TRAIN_BATCH_SIZE)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=TEST_BATCH_SIZE)

        with open('train_dataloader.pkl', 'wb') as f1:
            pickle.dump(train_dataloader, f1)
        
        with open('test_dataloader.pkl', 'wb') as f2:
            pickle.dump(test_dataloader, f2)

    return train_dataloader, test_dataloader

def train():
    # 获取数据
    print('start getting data')

    # train_dataloader, test_dataloader = generate_data_loaders(load=False)
    # exit()

    train_dataloader, test_dataloader = generate_data_loaders(load=True)

    # 配置模型信息
    print('start configuring model')
    loss_function = torch.nn.CrossEntropyLoss()
    warmup_steps = int(0.1 * len(train_dataloader))
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_parameters, lr=LRATE, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * EPOCHS)
    # 迭代模型
    print('start training model')
    f1_benchmark = 0
    model_to_save = None
    for epoch in range(1, EPOCHS + 1):
        # 训练
        model.train()
        loss_vec = []
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(DEVICE, dtype=torch.long) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss_vec.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % LOG_INTERVAL == 0:
                print(f'Epoch: {epoch} \t Step:{step} \t Average Loss:{np.mean(loss_vec):.4f}')
                loss_vec = []
        # 测试
        model.eval()
        pred_correct_samples, total_samples = 0, 0
        original_labels = []
        predicted_labels = []
        for batch in test_dataloader:
            T_start = time.perf_counter()
            batch = tuple(t.to(DEVICE, dtype=torch.long) for t in batch)
            input_ids, token_type_ids, attention_mask, labels = batch
            with torch.no_grad():
                # T_start = time.perf_counter()
                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                # T_end = time.perf_counter()
                # print(f'inference time (batchsize {TEST_BATCH_SIZE}): {1000 * (T_end - T_start) } ms')
            out_logits = outputs[0].detach().cpu().numpy()
            predictions = np.argmax(out_logits, axis=1)
            T_end = time.perf_counter()
            print(f'inference time (batchsize {TEST_BATCH_SIZE}): {1000 * (T_end - T_start) } ms')
            labels = labels.to('cpu').numpy()
            pred_correct_samples += np.sum(predictions == labels)
            total_samples += len(labels)
            original_labels.extend(labels)
            predicted_labels.extend(predictions)
    
        accuracy = pred_correct_samples / total_samples
        f1 = f1_score(original_labels, predicted_labels, average="macro")
        print(f'Accuracy: [{pred_correct_samples}/{total_samples}] {accuracy:.4f}\n')
        print(f'f1_score: {f1:.4f}\n')

       
        if f1 > f1_benchmark:
            model_to_save = model
    # 保存
    # if model_to_save:
    #     model_to_save.save(f'.pt')   

def test():
    pass


    
if __name__ == '__main__':
    train()
    test()







    

    