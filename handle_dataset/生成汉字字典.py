import json

# 常见的中文字符和英文标点符号
common_characters = "，。！？；：“”‘’()[]&#;``【】``;《》1234567890"
common_english_punctuation = ".,!?;:\"'()[]{}<>qwertyuiopasdfghjklzxcvbnm"

# 打开JSON文件以读取汉字映射数据
with open('../dataset/dataset/char_common.json', 'r', encoding='utf-8') as json_file:
    char_data = json.load(json_file)

# 创建汉字映射字典
word_map = {}
# 遍历char_data列表并将汉字及其对应的索引添加到字典中
for item in char_data:
    char = item.get("char", "")
    if char not in word_map:
        word_map[char] = len(word_map) + 1

# 遍历常见字符并将它们添加到字典中
for char in common_characters:
    if char not in word_map:
        word_map[char] = len(word_map) + 1

# 遍历常见英文标点符号并将它们添加到字典中
for char in common_english_punctuation:
    if char not in word_map:
        word_map[char] = len(word_map) + 1

# 添加特殊标记
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

if __name__ == '__main__':
    # 保存汉字映射字典到文件
    with open('../dataset/WORDMAP_corpus.json', 'w', encoding='utf-8') as map_file:
        json.dump(word_map, map_file, ensure_ascii=False)
