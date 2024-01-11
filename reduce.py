from PIL import Image
import os


# 因为没有数据了，所以用这个模拟缩略图加密
def reduce_image_quality(path, quality=5):
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(path, filename)
            with Image.open(image_path) as img:
                reduced_quality_path = os.path.join(path, "reduced_" + filename)
                img.save(reduced_quality_path, quality=quality)


# 替换为您的文件夹路径
folder_path = '/Users/binary/Desktop/val2017/'
reduce_image_quality(folder_path)
