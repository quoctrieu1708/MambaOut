# 📘 MambaOut Image Classification Project

---

## 1. Giới thiệu

Dự án này sử dụng mô hình **MambaOut** (tích hợp trong thư viện `timm`) để giải bài toán **phân loại ảnh**.

**Pipeline:**
Detection (XML) → Crop ảnh → Classification Dataset → Train → Test → Deploy

---

## 2. Cài đặt môi trường

### 2.1 Giải nén project

```bash
!unzip MambaOut.zip
%cd MambaOut
!ls
```

### 2.2 Cài thư viện

```bash
!pip install git+https://github.com/huggingface/pytorch-image-models.git
!pip install einops
```

---

## 3. Kiểm tra MambaOut

```python
import timm

mamba_models = [m for m in timm.list_models() if 'mambaout' in m]
print("Available models:", mamba_models)
```

---

## 4. Load model pretrained

```python
import torch
import timm

model_name = 'mambaout_tiny'

model = timm.create_model(model_name, pretrained=True)
model.eval()

print("Params:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
```

---

## 5. Inference ảnh mẫu

### 5.1 Preprocess

```python
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 5.2 Predict

```python
with torch.no_grad():
    output = model(input_batch)

prob = torch.softmax(output[0], dim=0)
```

---

## 6. Chuẩn bị dataset (XML → Classification)

### Classes

| Label | Mô tả |
|-------|-------|
| D00   | Loại 1 |
| D10   | Loại 2 |
| D20   | Loại 3 |
| D40   | Loại 4 |

### Quy trình xử lý

1. Đọc file XML
2. Lấy bounding box
3. Crop ảnh
4. Lưu theo folder class

### Fix lỗi thường gặp

- Clamp bbox vào kích thước ảnh
- Bỏ bbox không hợp lệ
- Bỏ ảnh rỗng sau khi crop

---

## 7. Load dataset

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("Japan_cls", transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

---

## 8. Train model

### 8.1 Fix version timm

```bash
!pip uninstall -y timm
!pip install timm==0.6.13
```

### 8.2 Chạy training

```bash
!python train.py /content/Japan_cls \
  --model mambaout_tiny \
  --pretrained \
  --num-classes 4 \
  --opt adamw \
  --lr 1e-4 \
  --sched cosine \
  --warmup-epochs 3 \
  -b 16 \
  -j 2 \
  --epochs 30 \
  --hflip 0.5 \
  --color-jitter 0.2 \
  --amp \
  --output /content/output
```

---

## 9. Test sau khi train

### 9.1 Load model

```python
import torch
from models.mambaout import mambaout_tiny

num_classes = 4

model = mambaout_tiny(pretrained=False, num_classes=num_classes)

checkpoint = torch.load(
    "/content/output/your_run/checkpoint-39.pth.tar",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint['state_dict'])
model.eval().cuda()
```

### 9.2 Preprocess

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 9.3 Hàm predict

```python
from PIL import Image

def predict_image(path):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()

    return pred, prob.cpu().numpy()
```

### 9.4 Class names

```python
class_names = ["asphalt", "gravel", "wet", "crack"]
```

### 9.5 Test 1 ảnh

```python
import matplotlib.pyplot as plt
from PIL import Image

img_path = "/content/test.jpg"

pred, prob = predict_image(img_path)

plt.imshow(Image.open(img_path))
plt.title(f"{class_names[pred]} ({prob[0][pred]:.2f})")
plt.axis("off")
plt.show()
```

---

## 10. Test nhiều ảnh

```python
import os

folder = "/content/test_folder"

for f in os.listdir(folder):
    p = os.path.join(folder, f)
    pred, prob = predict_image(p)
    print(f, "->", class_names[pred], prob[0][pred])
```

---

## 11. Đánh giá Accuracy

```python
correct = 0
total = 0

for imgs, labels in test_loader:
    imgs = imgs.cuda()
    labels = labels.cuda()

    with torch.no_grad():
        out = model(imgs)
        preds = torch.argmax(out, dim=1)

    correct += (preds == labels).sum().item()
    total += labels.size(0)

print("Accuracy:", correct / total)
```

---

> **Ghi chú:** Thay `your_run` trong đường dẫn checkpoint bằng tên thư mục thực tế được tạo ra sau khi train.