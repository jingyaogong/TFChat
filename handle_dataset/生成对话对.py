import json

max_len = 256
# # 打开JSON文件以读取数据
# with open('../dataset/dataset/dataset_train.json', 'r', encoding='utf-8') as json_file:
#     data = json_file.read()
#
# # 分割JSON对象并处理每个对象
# data = data.split('\n')  # 使用换行符分割JSON对象


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

input_file = "../dataset/dataset/data_zh.json"
with open(input_file, "r", encoding="utf-8") as file:  # 指定编码格式为utf-8
    data_zh = json.load(file)

new_data = []
for idx, item in enumerate(data_zh):
    query_history = []
    instruction = item["instruction"]
    output = item["output"]

    instruction = instruction.replace('\n\n', '\n')
    output = output[0].replace('\n\n', '\n')

    output = output[:max_len - 1].ljust(max_len - 1)

    qus = encode_question(instruction[:max_len-1], word_map)
    ans = encode_reply(output, word_map)
    pairs_encoded.append([qus, ans])

# for idx, json_object in enumerate(data):
#     if not json_object.strip():
#         continue  # 跳过空行
#
#     # 解析JSON对象
#     obj = json.loads(json_object)
#
#     # 从描述和答案中获取文本
#     desc = obj.get("desc", "")
#     answer = obj.get("answer", "")
#
#     desc = [_ for _ in desc]
#     answer = [_ for _ in answer]
#     qus = encode_question(desc[:max_len], word_map)
#     ans = encode_reply(answer[:max_len], word_map)
#     pairs_encoded.append([qus, ans])

if __name__ == '__main__':
    print('开始写pairs_encoded')
    with open('../pairs_encoded.json', 'w') as p:
        json.dump(pairs_encoded, p)
