# 1 加载库
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid


def image_show(inp, title=None):
    plt.figure(figsize=(14, 3))
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    plt.show()


def main():
    # 2 定义超参数
    BATCH_SIZE = 8  # 每批处理的数据数量
    EPOCHS = 10  # 训练轮数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LEARNING_RATE = 0.001  # 学习率

    # 3 图片转换
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize(300),
                transforms.RandomResizedCrop(300),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ]),

        'val':
            transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
    }

    # 4 操作数据集
    data_path = "chest_xray"  # 请根据实际数据集路径进行修改
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                              data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], shuffle=True,
                                 batch_size=BATCH_SIZE) for x in ['train', 'val']}
    data_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    target_names = image_datasets['train'].classes
    # 显示一个batch_size的图片（8张图片）
    datas, targets = next(iter(dataloaders['train']))  # 修正这里的拼写错误
    # 将若干张图片拼成一幅图像
    out = make_grid(datas, nrow=4, padding=10)
    # 显示图片
    image_show(out, title=[target_names[x] for x in targets])
    # 检查数据加载
    print("正在检查数据加载...")
    for phase in ['train', 'val']:
        inputs, labels = next(iter(dataloaders[phase]))
        print(f"{phase} 数据加载成功: 输入尺寸 {inputs.shape}, 标签尺寸 {labels.shape}")

    # 5 迁移学习：拿到一个成熟的模型，进行模型微调
    model = get_model().to(DEVICE)  # 调用模型构建函数并移动到设备
    criterion = nn.NLLLoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # 定义优化器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # 学习率调度

    # 6 训练过程记录
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    best_acc = 0.0  # 初始化最佳准确率
    best_model_file = "best_model.pth"  # 定义保存模型的文件名

    # 6 模型训练
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # 使用 tqdm 显示进度条
            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase} phase"):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / data_sizes[phase]
            epoch_acc = running_corrects.double() / data_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录损失和准确率
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
                scheduler.step()
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

                # 检查是否是最佳模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_file)  # 保存最佳模型参数到文件
                    print(f"保存最佳模型参数到 {best_model_file}")

                print("Validation Classification Report:")
                print(classification_report(all_labels, all_preds, target_names=target_names))

    print('训练完成!')

    # 加载最佳模型权重
    model.load_state_dict(torch.load(best_model_file))  # 从文件中加载最佳模型权重

    # 绘制损失和准确率曲线
    plt.figure(figsize=(12, 5))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# 7 自定义的池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super().__init__()
        size = size or (1, 1)
        self.pool_one = nn.AdaptiveAvgPool2d(size)
        self.pool_two = nn.AdaptiveAvgPool2d(size)

    def forward(self, x):
        return torch.cat([self.pool_one(x), self.pool_two(x)], dim=1)


# 8 模型构建
def get_model():
    model_pre = models.resnet50(pretrained=True)
    for param in model_pre.parameters():
        param.requires_grad = False
    model_pre.avgpool = AdaptiveConcatPool2d()
    model_pre.fc = nn.Sequential(
        nn.Flatten(),
        nn.BatchNorm1d(4096),
        nn.Dropout(0.5),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 2),
        nn.LogSoftmax(dim=1),
    )
    return model_pre


if __name__ == '__main__':
    main()
