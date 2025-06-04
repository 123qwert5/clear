import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class RainKITTI2012Dataset(Dataset):
    def __init__(self, rainy_dir, clean_dir_image_2, clean_dir_image_3, transform=None):
        self.rainy_dir = rainy_dir
        self.clean_dir_image_2 = clean_dir_image_2
        self.clean_dir_image_3 = clean_dir_image_3
        self.transform = transform
        self.rainy_images = sorted([f for f in os.listdir(rainy_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.rainy_images)

    def __getitem__(self, idx):
        rainy_path = os.path.join(self.rainy_dir, self.rainy_images[idx])
        rainy_img = Image.open(rainy_path).convert('RGB')

        if '_rain_2_50' in self.rainy_images[idx]:
            clean_filename = self.rainy_images[idx].replace('_rain_2_50.jpg', '_norain_2.png')
            clean_path = os.path.join(self.clean_dir_image_2, clean_filename)
        elif '_rain_3_50' in self.rainy_images[idx]:
            clean_filename = self.rainy_images[idx].replace('_rain_3_50.jpg', '_norain_3.png')
            clean_path = os.path.join(self.clean_dir_image_3, clean_filename)
        else:
            clean_filename = self.rainy_images[idx].replace('.jpg', '_norain_3.png')
            clean_path = os.path.join(self.clean_dir_image_3, clean_filename)

        clean_img = Image.open(clean_path).convert('RGB')

        if self.transform:
            rainy_img = self.transform(rainy_img)
            clean_img = self.transform(clean_img)

        return rainy_img, clean_img

def get_dataloader(data_dir, batch_size=8, mode='train', shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    sub_dir = 'training' if mode == 'train' else 'testing'
    rainy_dir = os.path.join(data_dir, sub_dir, 'image_2_3_rain50')  # 修正为实际路径
    clean_dir_image_2 = os.path.join(data_dir, sub_dir, 'image_2')
    clean_dir_image_3 = os.path.join(data_dir, sub_dir, 'image_3')
    dataset = RainKITTI2012Dataset(
        rainy_dir, clean_dir_image_2, clean_dir_image_3, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)