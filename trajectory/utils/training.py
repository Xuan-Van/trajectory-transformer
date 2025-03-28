import math
import torch
from torch.utils.data.dataloader import DataLoader
import pdb

from .timer import Timer

# 将数据列表中的每个元素移动到指定设备上
def to(xs, device):
    return [x.to(device) for x in xs]

# 训练器类
class Trainer:

    # 初始化训练器
    def __init__(self, config):
        self.config = config  # 配置参数
        self.device = config.device  # 设备（CPU/GPU）

        self.n_epochs = 0  # 当前训练的epoch数
        self.n_tokens = 0  # 已处理的token数量，用于学习率衰减
        self.optimizer = None  # 优化器

    # 获取优化器，如果尚未创建则创建
    def get_optimizer(self, model):
        if self.optimizer is None:
            print(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}') # 打印一条消息，指示正在创建优化器，并提供当前的epoch数
            self.optimizer = model.configure_optimizers(self.config) # 创建优化器
        return self.optimizer

    # 训练模型
    def train(self, model, dataset, n_epochs=1, log_freq=100): # $ log_freq：控制训练过程中日志输出的频率

        config = self.config
        optimizer = self.get_optimizer(model)  # 获取优化器
        model.train(True)  # 设置模型为训练模式
        vocab_size = dataset.N  # 获取词汇表大小

        # 创建数据加载器
        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers) # $ shuffle：打乱数据；pin_memory：加速数据传输

        # 训练多个epoch
        for _ in range(n_epochs):

            losses = []  # 存储每个batch的损失
            timer = Timer()  # 计时器
            for it, batch in enumerate(loader):

                batch = to(batch, self.device)  # 将batch数据移动到指定设备

                # 前向传播
                with torch.set_grad_enabled(True): # 启用梯度计算
                    logits, loss = model(*batch) # 调用模型进行前向传播
                    losses.append(loss.item()) # 添加损失

                # 反向传播和参数更新
                model.zero_grad() # 清零模型梯度
                loss.backward() # 计算损失的梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip) # 对梯度进行裁剪，防止梯度爆炸
                optimizer.step() # 更新模型参数

                # 根据进度衰减学习率
                if config.lr_decay: # 根据处理的token数量调整学习率
                    y = batch[-2]
                    self.n_tokens += (y != vocab_size).sum()  # 计算当前处理的token数量
                    if self.n_tokens < config.warmup_tokens:
                        # 线性warmup
                        lr_mult = float(self.n_tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # 余弦学习率衰减
                        progress = float(self.n_tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    lr = config.learning_rate * lr_mult # 更新优化器的学习率
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    lr = config.learning_rate

                # 报告训练进度
                if it % log_freq == 0:
                    print( # 打印训练进度信息，包括当前epoch、batch索引、训练损失、学习率、学习率乘数和计时器时间
                        f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                        f'train loss {loss.item():.5f} | lr {lr:.3e} | lr_mult: {lr_mult:.4f} | '
                        f't: {timer():.2f}')

            self.n_epochs += 1  # 更新epoch计数