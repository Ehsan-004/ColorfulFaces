import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision
import cv2
import numpy as np


def preprocess_dataset(root_path, ignore=None):
    """
    تابعی برای جمع‌آوری فایل‌ها از پوشه‌ها برای پروژه‌های بینایی ماشین (مثل segmentation)

    Parameters:
    -----------
    root_path : str or Path
        مسیر پوشه اصلی که زیرپوشه‌ها داخلش هستن (مثلاً 'data')
    ignore : list of str, optional
        نام پوشه‌هایی که باید نادیده گرفته بشن

    Returns:
    --------
    dict
        دیکشنری از فرمت {folder_name: [list of file paths]}
    pd.DataFrame
        دیتافریم مشابه که هر ستون نام یک پوشه است و ردیف‌ها مسیر فایل‌ها
    """
    root_path = Path(root_path)
    ignore = ignore or []
    # dataset_dict = {}
    
    # has_content = False

    # for folder in root_path.iterdir():
    #     if folder.is_dir() and folder.name not in ignore:
    #         has_content = True
    # files = [str(f.resolve()) for f in root_path.rglob("*") if f.is_file()]
    files = [p.name for p in root_path.rglob("*")]
    # files = [root_path.joinpath(p.name) for p in root_path.rglob("*")]
    
    
    if len(files) == 0:
        print("directory is empty!")
        return
        
            # dataset_dict[folder.name] = files
    # ساخت دیتافریم با کمترین طول ستون‌ها یکسان‌سازی شده
    
    # max_len = max(len(v) for v in dataset_dict.values())
    # for k in dataset_dict:
    #     # پر کردن با None تا طول‌ها برابر شوند
    #     dataset_dict[k] += [None] * (max_len - len(dataset_dict[k]))

    df = pd.DataFrame({"path": files})
    sep = os.path.sep
    df["path"] = str(root_path) + sep + df["path"]

    return df



def split_dataset(df, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, save_dir=".", random_state=42):
    """
    تقسیم دیتافریم به سه مجموعه train, valid, test و ذخیره هر کدوم به صورت CSV
    Parameters:
    -----------
    df : pd.DataFrame
        دیتافریم اصلی که مسیر فایل‌ها در ستون‌ها هستند
    train_ratio : float
        نسبت داده‌ها برای مجموعه آموزش
    valid_ratio : float
        نسبت داده‌ها برای مجموعه اعتبارسنجی
    test_ratio : float
        نسبت داده‌ها برای مجموعه تست
    save_dir : str or Path
        مسیر ذخیره CSV ها
    random_state : int
        عدد برای شافل کردن دیتافریم برای reproducibility

    Returns:
    --------
    None
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # شافل ردیف‌ها
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    n = len(df_shuffled)
    # محاسبه تعداد نمونه‌ها برای هر مجموعه
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    n_test = int(n * test_ratio)

    # تقسیم‌بندی
    train_df = df_shuffled.iloc[:n_train] if n_train > 0 else pd.DataFrame(columns=df.columns)
    valid_df = df_shuffled.iloc[n_train:n_train+n_valid] if n_valid > 0 else pd.DataFrame(columns=df.columns)
    test_df  = df_shuffled.iloc[n_train+n_valid:n_train+n_valid+n_test] if n_test > 0 else pd.DataFrame(columns=df.columns)

    # ذخیره CSV
    train_df.to_csv(save_dir / "train.csv", index=False)
    valid_df.to_csv(save_dir / "valid.csv", index=False)
    test_df.to_csv(save_dir / "test.csv", index=False)

    print(f"Datasets saved to {save_dir.resolve()}")
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")

# مثال استفاده:
# split_dataset(df, train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, save_dir="processed_data")




class ColorFaceDataset(Dataset):
    def __init__(self, csv_path, transforms):
        super().__init__()
        df = pd.read_csv(csv_path)
        self.images = list(df["path"])
        self.transforms = transforms
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as img:
            L, ab = self.transforms(img)
            return {
                "L": L,     # ورودی
                "ab": ab    # هدف
            }

    
class ColorFaceTransform:
    def __init__(self, is_train=True, image_size=(256, 256)):
        self.train = is_train
        self.image_size = image_size
        # اضافه کردن Resize برای یکسان‌سازی ابعاد
        self.resize = transforms.Resize(image_size)
        self.random_flip = transforms.RandomHorizontalFlip(p=0.5) if is_train else None
        
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = self.resize(img)
        
        if self.train:
            img = self.random_flip(img)

        img_np = np.array(img)  # uint8, 0..255
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)  # OpenCV LAB: L 0..255, a/b 0..255 (128 center)

        # Correct normalization:
        L = lab[..., 0] / 255.0                     # -> [0,1]
        ab = (lab[..., 1:3] - 128.0) / 128.0       # -> approx [-1,1]

        L_tensor = torch.from_numpy(L).unsqueeze(0).float()        # [1,H,W]
        ab_tensor = torch.from_numpy(ab).permute(2,0,1).float()   # [2,H,W]

        return L_tensor, ab_tensor
        # img_np = np.array(img)
        # lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # L = lab[..., 0] / 100.0             # [0, 1]
        # ab = (lab[..., 1:3] + 128) / 255.0   # انتقال به بازه [0, 1] برای پایداری بیشتر (اختیاری)
        # # یا همون روش خودت: lab[..., 1:3] / 128.0 (بازه -1 تا 1)
        
        # L_tensor = torch.from_numpy(L).unsqueeze(0)         
        # ab_tensor = torch.from_numpy(ab).permute(2,0,1)     
        # return L_tensor, ab_tensor
