import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

# 定义添加椒盐噪声的函数
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    num_pepper = np.ceil(pepper_prob * image.size)

    # 添加盐噪声
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 1

    # 添加胡椒噪声
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# 1. 加载 Fashion-MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])  # 将图像转换为 PyTorch 张量
fashion_mnist = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

# 2. 提取10张 T 恤衫图片 (类别 0 表示 T-shirt/top)
tshirt_images = []  # 存放图片
for image, label in fashion_mnist:
    if label == 0:  # 如果标签是 T-shirt/top
        tshirt_images.append(image)  # 提取到的 T 恤衫图片
    if len(tshirt_images) == 10:  # 找到 10 张后停止
        break

# # 3. 将图片转换成 NumPy 格式并生成噪声
# mean = 0             # 噪声均值
# std_dev = 0.3      # 噪声标准差
# original_images = []   # 原始图片列表
# noisy_images = []      # 加噪图片列表
# 3. 将图片转换成 NumPy 格式并生成噪声
mean = 0             # 噪声均值
std_dev = 0.3      # 噪声标准差
salt_prob = 0.05   # 盐噪声概率
pepper_prob = 0.05 # 胡椒噪声概率
original_images = []   # 原始图片列表
noisy_images = []      # 加噪图片列表

# for img in tshirt_images:
#     # 转换 Tensor 到 NumPy 格式 (1x28x28 -> 28x28)
#     img_np = img.squeeze().numpy()
#     original_images.append(img_np)

#     # 生成噪声并叠加到图片上
#     noise = np.random.normal(mean, std_dev, img_np.shape)  # 生成高斯噪声
#     noisy_img = img_np + noise  # 图片与噪声相加
#     #noisy_img = np.clip(noisy_img, 0, 1)  # 将像素值裁剪到 [0,1]
#     noisy_images.append(noisy_img)

for img in tshirt_images:
    # 转换 Tensor 到 NumPy 格式 (1x28x28 -> 28x28)
    img_np = img.squeeze().numpy()
    original_images.append(img_np)

    # 生成高斯噪声并叠加到图片上
    noise = np.random.normal(mean, std_dev, img_np.shape)  # 生成高斯噪声
    noisy_img = img_np + noise  # 图片与噪声相加
    noisy_img = np.clip(noisy_img, 0, 1)  # 将像素值裁剪到 [0,1]

    # 生成椒盐噪声并叠加到图片上
    noisy_img = add_salt_and_pepper_noise(noisy_img, salt_prob, pepper_prob)

    noisy_images.append(noisy_img)

# 4. 可视化：输出原始图片和添加噪声后的图片
plt.figure(figsize=(12, 6))

for i in range(10):
    # 显示原始图片
    plt.subplot(2, 10, i + 1)
    plt.imshow(original_images[i], cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # 显示加噪图片
    plt.subplot(2, 10, i + 11)
    plt.imshow(noisy_images[i], cmap='gray')
    plt.title("Noisy")
    plt.axis('off')

plt.tight_layout()
plt.show()