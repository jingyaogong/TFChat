
![logo](md_imgs/lg.png)
# 实现聊天机器人TFChat，本质上是个麻雀虽小五脏俱全的ChatGPT

* 这是一个使用Transformer构建的问答GPT，代码很简化，有注释，是一个教程和备忘录。
* 本项目可以快速帮助小白了解到大模型工作原理，例如自然语言在深度神经网络中如何进行编码和推理传播，如何进行中文分词，
  Transformer的输入和输出细节到底是什么，传说中的Embeddings如何嵌入式编码向量，MLP层，FeedForward层和多头注意力MUL-Attention的实现细节等等。
* 训练效果在英文数据集上表现勉强能看，中文语料库上惨不忍睹（基本无法正常对话）。首先训练集只够大模型微调，太少，其次模型复杂度太低，远没达到“顿悟现象”所需最低参数量和数据量。用于学习和了解大模型的工作原理足够，上手难度低。
* 构建不易，留个star
* 作者：jingyaogong

### 预训练的TFChat模型由2层Encoder-Decoder层组成，结构如下

![img_1.png](md_imgs/img_1.png)
### Netron可视化前向推理结构
![img_1.png](md_imgs/model_visual.png)

## 配置
PyTorch 1.1+, Python 3.7+, NVIDIA 4090单卡


## 快速开始
需要下载dataset文件夹包含[数据集+中文分词](https://pan.baidu.com/s/1e76erEeq5WUiufB3VnOCYA?pwd=8765) 到根目录

生成中文字典
```
python handle_dataset/生成汉字字典.py
```
生成训练用的对话对
```
python handle_dataset/生成对话对.py
```
训练
```
python train.py --d_model 512 --heads 8 --num_layers 2 --device cuda --epochs 100
```
测试聊天（附加：5轮小规模预训练[checkpoint.pth](https://pan.baidu.com/s/1RP-JhyMuNa7N-cYEqS_GVA?pwd=8765 )）
```
python chat.py
```


## 运行测试

| 层数 | 参数量    | 推理显存 | 速度(token/S) |
|----|--------|------|-------------|
| 1  | 0.011B | 0.8G | 170+        |
| 2  | 0.018B | 1G   | 150+        |

| batch-size | 层数  | 参数量    | 训练显存 | 收敛时间 |
|------------|-----|--------|------|------|
| 50         | 2   | 0.018B | 2.8G | 3h+  |

## 文件



`./handle_dataset/*.py:` 数据集生成（首先执行） <br/>
`utils.py:` 工具类  <br/>
`models.py:` 模型文件 <br/>
> 常见的LLama，GPT，国产大模型清一色是Decoder-only系列，因为Chat-Model本质上并不需要编码即可完美地生成对话
> 
> Encoder-only代表有Bert
> 
> TFChat完成了一个标准的Seq2Seq编码-解码结构，起到学习作用
> 1. **Embeddings层：** 该层将输入的词汇索引映射为对应的词嵌入向量，并添加了位置编码。这有助于网络理解词汇的语义和位置信息。
> 2. **Encoder层：** 包含多个相同结构的编码器层，每个层由自注意力机制（Multi-Head
   Attention）和前馈神经网络组成。自注意力机制使模型能够在输入序列中捕捉不同位置的关系，而前馈神经网络则对每个位置的特征进行非线性变换。
> 3. **Decoder层：** 同样包含多个相同结构的解码器层，每个层由自注意力机制、编码-解码注意力机制（用于获取源序列的信息）和前馈神经网络组成。解码器通过对先前预测的词进行自注意力和对编码器输出进行注意力来生成输出序列。
> 4. **输出层：** 将解码器的输出通过线性变换映射到词汇空间，最后通过softmax函数产生概率分布，表示每个词的可能性。
> 整个模型的训练过程涉及对编码器和解码器的多层结构进行交替训练，通过最小化输出序列与目标序列之间的差异来优化模型参数。这种结构使得Transformer能够处理不同长度的输入和输出序列，并在捕捉长距离依赖关系方面表现优越。


`train.py:` 训练脚本文件 <br/>
`chat.py:` 测试模型的对话能力脚本



## 中文测试集效果
![img_2.png](md_imgs/img_2.png)
![img_3.png](md_imgs/img_3.png)


