import torch.utils.data as data
from models import *
from utils import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

train_loader = data.DataLoader(Dataset(),
                               batch_size=50,
                               shuffle=True,
                               pin_memory=True)

d_model = 512
heads = 8
num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 100

with open('./dataset/WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)

# 创建Transformer模型
transformer = Transformer(d_model=d_model, heads=heads, num_layers=num_layers, word_map=word_map, max_len=120)
transformer = transformer.to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size=d_model, warmup_steps=4000, optimizer=adam_optimizer)
criterion = LossWithLS(len(word_map), 0.1)

# 打印模型参数数量
pfm = count_parameters(transformer)
print(f'模型参数数量: {pfm} ({(pfm / 1e6):.2f} 百万)')


def train(train_loader, transformer, criterion, epoch):
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):

        samples = question.shape[0]

        # 移动到设备
        question = question.to(device)
        reply = reply.to(device)

        # 准备目标数据
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        # 创建掩码并添加维度
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        # 获取Transformer输出
        out = transformer(question, question_mask, reply_input, reply_input_mask)

        # 计算损失
        loss = criterion(out, reply_target, reply_target_mask)

        # 反向传播
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()

        sum_loss += loss.item() * samples
        count += samples

        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\t损失: {:.3f}".format(epoch, i, len(train_loader), sum_loss / count))


for epoch in range(epochs):
    train(train_loader, transformer, criterion, epoch)

    if epoch % 4 == 0:
        state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}
        torch.save(state, 'checkpoint_2_' + str(epoch) + '.pth')