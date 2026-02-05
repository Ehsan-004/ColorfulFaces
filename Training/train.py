import os
from time import time
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
from model import get_model
from dataset import ColorFaceDataset, ColorFaceTransform, split_dataset, preprocess_dataset


torch.cuda.empty_cache()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VERSION = 1

print(f"training on device: {DEVICE}")


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


torch.cuda.empty_cache()


if not os.path.exists("data/train.csv"):
    df = preprocess_dataset("../datasets/celeba/")
    split_dataset(df, train_ratio=0.8, valid_ratio=0.15, test_ratio=0.05, save_dir="data/")
else:
    print("Dataset CSVs already exist. skipping split.")

train_transform = ColorFaceTransform(is_train=True)
valid_transform = ColorFaceTransform(is_train=False)


train_dataset = ColorFaceDataset("data/train.csv", train_transform)
valid_dataset = ColorFaceDataset("data/valid.csv", valid_transform)


num_workers = int(min(8, os.cpu_count()))
pin_memory = True


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=24, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=24, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

print("creating the model")

model = get_model()

model = model.to(DEVICE)

# criterion = FaceUNetLoss(lambda_ssim=0.3)
criterion = torch.nn.MSELoss()


# inc_params = filter(lambda p: p.requires_grad, model.inc.parameters())
# down_params = filter(lambda p: p.requires_grad, model.downs.parameters())
# ups_params = filter(lambda p: p.requires_grad, model.ups.parameters())

# optimizer = torch.optim.Adam([
#     {"params": inc_params, "lr": 1e-4},
#     {"params": down_params, "lr": 1e-4},
#     {"params": ups_params, "lr": 1e-3},
# ])

lr = 0.0005
weight_decay = 1e-5

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# if valid loss doesn't decrease after two epochs, decreases lr by a factor of 10%
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

scaler = torch.amp.GradScaler(DEVICE)


num_epochs = 15
train_losses = []
valid_losses = []
test_accuracies = []

os.makedirs("outputs", exist_ok=True)
os.makedirs(f"outputs/checkpoints/v{VERSION}/", exist_ok=True)

patience_number = 3
delta = 0.0001
best_val_loss = float("inf")
patience = 0


t1 = time()
print(f"Start training with {num_epochs} epochs ...")

torch.cuda.empty_cache()

for epoch in range(num_epochs):
    print(f"\n==== Epoch {epoch+1}/{num_epochs} ====")

    torch.cuda.empty_cache()

    # ---------- TRAIN ----------
    model.train()
    total_train_loss = 0.0
    total_train_samples = 0

    pbar_train = tqdm(train_loader, desc="Training", leave=True)
    for batch in pbar_train:
        gray_img, ab_img = batch["L"].to(DEVICE), batch["ab"].to(DEVICE)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda"):
            outputs = model(gray_img)
            loss = criterion(outputs, ab_img)

        # print("=== Test Values ===")
        # print("ab_img min/max (first 5 samples):", ab_img[:22].min().item(), ab_img[:22].max().item())
        # print("outputs min/max (first 5 samples):", outputs[:22].min().item(), outputs[:22].max().item())
        # print("==================")
        # exit()



        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()
        
        total_train_loss += loss.item() * ab_img.size(0)
        total_train_samples += ab_img.size(0)

        # updating progressbar
        pbar_train.set_postfix({
            "Loss": f"{total_train_loss/total_train_samples:.4f}",
            # "Acc": f"{total_train_correct/total_train_samples*100:.2f}%"
        })
    torch.cuda.empty_cache()

    avg_train_loss = total_train_loss / total_train_samples
    # avg_train_acc = total_train_correct / total_train_samples
    train_losses.append(avg_train_loss)
    # train_accuracies.append(avg_train_acc*100)

    # ---------- VALID ----------
    model.eval()
    total_valid_loss = 0.0
    total_valid_samples = 0

    pbar_valid = tqdm(valid_loader, desc="Validating", leave=True)
    with torch.no_grad():
        for batch in pbar_valid:
            gray_img, ab_img = batch["L"].to(DEVICE), batch["ab"].to(DEVICE)

            with torch.autocast(device_type="cuda"):
                outputs = model(gray_img)
                loss = criterion(outputs, ab_img)
            
            total_valid_loss += loss.item() * ab_img.size(0)
            total_valid_samples += ab_img.size(0)

            # updating progressbar
            pbar_valid.set_postfix({
                "Loss": f"{total_valid_loss/total_valid_samples:.4f}",
                # "Acc": f"{total_valid_correct/total_valid_samples*100:.2f}%"
            })

    

    avg_valid_loss = total_valid_loss / total_valid_samples
    valid_losses.append(avg_valid_loss)

    # ---------- SUMMARY ----------
    print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    # print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc*100:.2f}%")
    print(f"Valid Loss: {avg_valid_loss:.4f}")

    

    # lr scheduler
    scheduler.step(avg_valid_loss)

    # early stopping
    if avg_valid_loss < best_val_loss - delta:
        best_val_loss = avg_valid_loss
        patience = 0
        torch.save(model.state_dict(), f"outputs/checkpoints/v{VERSION}/faceNet_best.pth")
        print(f"✅ Validation improved. Saving model.")
    else:
        patience += 1
        print(f"⚠️ No improvement. Early stop counter: {patience}/{patience_number}")
        if patience >= patience_number:
            print("⛔ Early stopping triggered!")
            break

torch.save(model.state_dict(), f"outputs/checkpoints/v{VERSION}/faceNet_final.pth")
t2 = time()
print(f"Training Finished After {t2-t1:.2f} seconds!")

print("train losses:")
print(train_losses)
print("valid losses:")
print(valid_losses)

# matplotlib.use("Agg")
# plt.savefig(f"outputs/checkpoints/v{VERSION}/loss.png")


# plot_losses(train_losses, valid_losses)

save_dir = f"outputs/checkpoints/v{VERSION}"
os.makedirs(save_dir, exist_ok=True)

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker='o', label='Train Loss')
plt.plot(epochs, valid_losses, marker='o', label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)

save_path = os.path.join(save_dir, "loss_curve.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Results saved at: {save_path}")