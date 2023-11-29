import json

max_len = 100
# 打开JSON文件以读取数据
with open('../dataset/dataset/dataset_train.json', 'r', encoding='utf-8') as json_file:
    data = json_file.read()

# 分割JSON对象并处理每个对象
data = data.split('\n')  # 使用换行符分割JSON对象

# 初始化空字典来存储汉字映射
chinese_character_mapping = {}
next_index = 1  # 下一个可用的索引

pairs_encoded = []  # 存储编码后的对话对

# 打开WORDMAP_corpus_.json文件以读取汉字映射数据
with open('../dataset/WORDMAP_corpus.json', 'r', encoding='utf-8') as word_map_file:
    word_map = json.load(word_map_file)


def encode_question(words, word_map):
    enc_c = [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<pad>']] * (max_len - len(words))
    return enc_c


def encode_reply(words, word_map):
    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<end>']] + [
        word_map['<pad>']] * (max_len - len(words))
    return enc_c


pairs_encoded = []
for idx, json_object in enumerate(data):
    if not json_object.strip():
        continue  # 跳过空行

    # 解析JSON对象
    obj = json.loads(json_object)

    # 从描述和答案中获取文本
    desc = obj.get("desc", "")
    answer = obj.get("answer", "")

    desc = [_ for _ in desc]
    answer = [_ for _ in answer]
    qus = encode_question(desc[:max_len], word_map)
    ans = encode_reply(answer[:max_len], word_map)
    pairs_encoded.append([qus, ans])

if __name__ == '__main__':
    print('开始写pairs_encoded')
    with open('pairs_encoded.json', 'w') as p:
        json.dump(pairs_encoded, p)
