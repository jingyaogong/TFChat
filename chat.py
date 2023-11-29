import json
import torch
from torch.utils.data import Dataset
import torch.utils.data
from models import *
from utils import *
from handle_dataset.生成汉字字典 import *

load_checkpoint = True
ckpt_path = 'checkpoint_2_20.pth'
checkpoint = torch.load(ckpt_path)
transformer = checkpoint['transformer']


def evaluate(transformer, question, question_mask, max_len, word_map):
    """
    Performs Greedy Decoding with a batch size of 1
    """
    rev_word_map = {v: k for k, v in word_map.items()}
    transformer.eval()
    start_token = word_map['<start>']
    encoded = transformer.encode(question, question_mask)
    words = torch.LongTensor([[start_token]]).to(device)

    for step in range(max_len - 1):
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)
        decoded = transformer.decode(words, target_mask, encoded, question_mask)
        predictions = transformer.logit(decoded[:, -1])
        _, next_word = torch.max(predictions, dim=1)
        next_word = next_word.item()
        if next_word == word_map['<end>'] or next_word == word_map['<unk>']:
            break
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim=1)  # (1,step+2)

    # Construct Sentence
    if words.dim() == 2:
        words = words.squeeze(0)
        words = words.tolist()

    sen_idx = [w for w in words if w not in {word_map['<start>']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])

    return sentence


def chat_with_input():
    while True:
        question = input("问题: ")
        if question == 'exit':
            break
        max_len = input("Maximum Reply Length: ")
        enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
        question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
        question_mask = (question != 0).to(device).unsqueeze(1).unsqueeze(1)
        sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)
        print(sentence)



def chat_with_pre():
    questions = [
        "你是男的女的？",
        "你好啊！",
        "你看到我的电脑了吗？",
    ]

    for question in questions:
        # question = input("Question: ")
        if question == 'exit':
            break
        # 使用空格拆分字符串
        split_words = [_ for _ in question]
        # 使用空格拼接拆分后的单词
        question = ' '.join(split_words)

        # max_len = input("Maximum Reply Length: ")
        max_len = 256
        enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
        question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
        question_mask = (question != 0).to(device).unsqueeze(1).unsqueeze(1)
        sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)
        print(sentence)




def eval_from_train_dataset():
    import json
    max_len = 256
    input_file = "./dataset/dataset/data_zh.json"
    with open(input_file, "r", encoding="utf-8") as file:  # 指定编码格式为utf-8
        data_zh = json.load(file)
    next_index=0
    for idx, item in enumerate(data_zh):
        instruction = item["instruction"]
        desc = instruction.replace('\n\n', '\n')
        if len(desc) <= 0 or len(desc) >= max_len:
            continue
        split_words = [_ for _ in desc]
        # 使用空格拼接拆分后的单词
        question = ' '.join(split_words)
        enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]
        question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)
        question_mask = (question != 0).to(device).unsqueeze(1).unsqueeze(1)
        sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)
        if sentence == '':
            continue

        print(
            "问题" + str(next_index) + "：" + desc.replace('\n', '') + "\n回答" + str(next_index) + "：" + sentence.replace('\n',
                                                                                                                      '').replace(
                ' ', '') + "\n\n")
        next_index += 1




if __name__=='__main__':
    eval_from_train_dataset()