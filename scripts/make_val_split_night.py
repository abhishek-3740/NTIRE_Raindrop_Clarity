import os
import random
import shutil

def main():
    # Setup paths - Update base_dir to your dataset location
    base_dir = os.getenv("DATA_ROOT", "data")
    s1_train = os.path.join(base_dir, "stage1", "train")
    s1_val = os.path.join(base_dir, "stage1", "val")
    gt_train = os.path.join(base_dir, "gt", "train")
    gt_val = os.path.join(base_dir, "gt", "val")

    os.makedirs(s1_val, exist_ok=True)
    os.makedirs(gt_val, exist_ok=True)

    # 🌙 Find ONLY the Night images
    night_images = [f for f in os.listdir(s1_train) if f.startswith("Night_") and f.endswith(('.png', '.jpg', '.jpeg'))]
    night_images.sort()

    if len(night_images) == 0:
        print("❌ No Night images found in the train folder!")
        return

    # Pick 5% for validation
    val_count = int(len(night_images) * 0.05)
    random.seed(42) # Keeps it reproducible
    val_images = random.sample(night_images, val_count)

    print(f"Found {len(night_images)} Night images in train directory.")
    print(f"Moving {val_count} Night images to validation folders...")

    for img in val_images:
        # Move Stage 1 predictions
        shutil.move(os.path.join(s1_train, img), os.path.join(s1_val, img))
        # Move Ground Truth
        shutil.move(os.path.join(gt_train, img), os.path.join(gt_val, img))

    print("✅ Night Validation split complete!")

if __name__ == "__main__":
    main()