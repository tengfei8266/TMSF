# from PIL import Image
# import torchvision.transforms as transforms
# import torch
# # A_path = "/home/cgd/Pictures/1.png"
# # A_img = Image.open(A_path).convert('RGB')
# # a = transforms.ToTensor()(A_img)
# # A = 1

# line = "path1 path2 path3"
# feats = line.strip().split(' ')[2:]
# x = torch.randn(16,1,192,192)
# mask = (x > 0.5)
# # mask = mask.bool()
# y0 = x[mask]
# y = torch.masked_select(x, mask)
# y1 = torch.where(mask, x, torch.zeros_like(x))

# print(x[mask])
from time import time
import multiprocessing as mp
import torch
import torchvision
from torchvision import transforms


transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,  # 如果为True，从 training.pt 创建数据，否则从 test.pt 创建数据。
    download=True,  # 如果为true，则从 Internet 下载数据集并将其放在根目录中。 如果已下载数据集，则不会再次下载。
    transform=transform
)

print(f"num of CPU: {mp.cpu_count()}")
for num_workers in range(2, mp.cpu_count(), 2):
    train_loader = torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
    start = time()
    for epoch in range(1, 3):
        for i, data in enumerate(train_loader, 0):
            pass
    end = time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
