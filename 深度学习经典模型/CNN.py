import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
from torch import nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#########################################MNIST数据集###########################################
####parameters####

####parameters####
# dataset
input_shape = 28
num_classes = 10#图片的类型数，到了全连接层才要使用这个参数

# hyper
batch_size = 64
num_epochs = 5
# 梯度是用来指导模型参数更新的
learning_rate = 1e-3

# gpu擅长于执行高度线程化的并行处理任务（如大型矩阵运算）
#设置GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#对于MNIST手写的数据集，使用pytorch自带的数据库
#下面是在导入数据集
train_dataset = datasets.MNIST(root='../data/',
                               download=True,
                               train=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='../data/',
                              download=True,
                              train=False,
                              transform=transforms.ToTensor())
#下面是在进行数据的加载
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               shuffle=True,
                                               batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              shuffle=False,
                                              batch_size=batch_size)
images, labels = next(iter(train_dataloader))
# iter会获取一个对象并返回与该参数对应的迭代器对象
# next使用迭代器并返回输入中的下一个元素。如果我们再次调用它，它会记住之前返回的值并输入下一个值


# batch_size, channels, h, w
images.shape
torch.Size([64, 1, 28, 28])

# 数据可视化
import numpy as np

# 指定图片大小，图像大小为20宽、5高的绘图(单位为英寸inch)
plt.figure('数据可视化', figsize=(20, 5))
for i, images in enumerate(images[:20]):
    # 维度缩减
    npimg = np.squeeze(images.numpy())
    # 将整个figure分成2行10列，绘制第i+1个子图。
    plt.subplot(2, 10, i + 1)
    plt.imshow(npimg, cmap=plt.cm.binary)
    plt.axis('off')
plt.show()
####model arch####
# cnn: channel 不断增加，shape 不断减少的过程
# 最好是 *2
class CNN(nn.Module):
    def __init__(self, input_shape, in_channels, num_classes):
        super(CNN, self).__init__()
        # super是一个调用父类方法的函数（反虚函数？）
        # conv2d: (b, 1, 28, 28) => (b, 16, 28, 28)
        # maxpool2d: (b, 16, 28, 28) => (b, 16, 14, 14)
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16,
                                            kernel_size=5, padding="same", stride=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))

        # conv2d: (b, 16, 14, 14) => (b, 32, 14, 14)
        # maxpool2d: (b, 32, 14, 14) => (b, 32, 7, 7)
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32,
                                            kernel_size=5, padding="same", stride=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        # (b, 32, 7, 7) => (b, 32*7*7)
        # (b, 32*7*7) => (b, 10)

        self.fc = nn.Linear(32 * (input_shape // 4) * (input_shape // 4), num_classes)

    def forward(self, x):
        # (b, 1, 28, 28) => (b, 16, 14, 14)
        out = self.cnn1(x)
        # (b, 16, 14, 14) => (b, 32, 7, 7)
        out = self.cnn2(out)
        # (b, 32, 7, 7) => (b, 32*7*7)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


####torchsummary

from torchsummary import summary

model = CNN(input_shape=input_shape, num_classes=num_classes, in_channels=1).to(device)
summary(model, input_size=(1, 28, 28), batch_size=batch_size)

# model
# train
criterion = nn.CrossEntropyLoss()#创建损失函数，起到一个造型上的作用谢谢
optimzer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(train_dataloader)
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        out = model(images)
        loss = criterion(out, labels)

        # backward
        optimzer.zero_grad()
        loss.backward()
        optimzer.step()  # 更细 模型参数

        if (batch_idx + 1) % 100 == 0:
            print(f'{epoch + 1}/{num_epochs}, {batch_idx + 1}/{total_batch}: {loss.item():.4f}')

####model evaluation
total = 0
correct = 0
for images, labels in test_dataloader:
    images = images.to(device)
    labels = labels.to(device)
    out = model(images)
    preds = torch.argmax(out, dim=1)

    total += images.size(0)
    correct += (preds == labels).sum().item()
print(f'{correct}/{total}={correct / total}')

####model save
torch.save(model.state_dict(), 'cnn_mnist.ckpt')

######################################### CIFAR10数据集###########################################
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from tensorflow.keras import datasets, layers, models
# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#
# # Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

class CNN(nn.Module):
    def __init__(self, input_shape, in_channels, num_classes):
        super(CNN, self).__init__()
        # super是一个调用父类方法的函数（反虚函数？）
        # conv2d: (b, 1, 28, 28) => (b, 16, 28, 28)
        # maxpool2d: (b, 16, 28, 28) => (b, 16, 14, 14)
        self.cnn1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=16,
                                            kernel_size=5, padding=2, stride=1),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))

        # conv2d: (b, 16, 14, 14) => (b, 32, 14, 14)
        # maxpool2d: (b, 32, 14, 14) => (b, 32, 7, 7)
        self.cnn2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32,
                                            kernel_size=5, padding=2, stride=1),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=2, stride=2))
        # (b, 32, 7, 7) => (b, 32*7*7)
        # (b, 32*7*7) => (b, 10)
        self.fc = nn.Linear(32 * (input_shape // 4) * (input_shape // 4), num_classes)

    def forward(self, x):
        # (b, 1, 28, 28) => (b, 16, 14, 14)
        out = self.cnn1(x)
        # (b, 16, 14, 14) => (b, 32, 7, 7)
        out = self.cnn2(out)
        # (b, 32, 7, 7) => (b, 32*7*7)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


####torchsummary

from torchsummary import summary

model = CNN( in_channels=3).to(device)

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)


############################MNIST可视化##############################################
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = True

# 获取训练集dataset
training_data = torchvision.datasets.MNIST(
    root='./data/',  # dataset存储路径
    train=True,  # True表示是train训练集，False表示test测试集
    transform=torchvision.transforms.ToTensor(),  # 将原数据规范化到（0,1）区间
    download=DOWNLOAD_MNIST,
)

# 打印MNIST数据集的训练集及测试集的尺寸
print(training_data.data.size())
print(training_data.targets.size())
# torch.Size([60000, 28, 28])
# torch.Size([60000])
plt.imshow(training_data.data[0].numpy(), cmap='gray')
plt.title('simple')
plt.show()

# 通过torchvision.datasets获取的dataset格式可直接可置于DataLoader
train_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE,
                               shuffle=True)

# 获取测试集dataset
test_data = torchvision.datasets.MNIST(root='./data/', train=False)
# 取前2000个测试集样本

test_x = Variable(torch.unsqueeze(test_data.data, dim=1),
                  volatile=True).type(torch.FloatTensor)[:2000] / 255
# (2000, 28, 28) to (2000, 1, 28, 28), in range(0,1)
test_y = test_data.targets[:2000]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1,28,28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                      stride=1, padding=2),  # (16,28,28)
            # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # (16,14,14)
        )
        self.conv2 = nn.Sequential(  # (16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # (32,7,7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 将（batch，32,7,7）展平为（batch，32*7*7）
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(b_x)
        loss = loss_function(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            s1=sum(pred_y == test_y)
            s2=test_y.size(0)
            accuracy = s1/(s2*1.0)
            print('Epoch:', epoch, '|Step:', step,
                  '|train loss:%.4f' % loss.item(), '|test accuracy:%.4f' % accuracy)

test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

for n in range(10):
    plt.imshow(test_data.data[n].numpy(), cmap='gray')
    plt.title('data[%i' % n+']:   test:%i' % test_data.targets[n]+'   pred:%i' % pred_y[n])
    plt.show()


















