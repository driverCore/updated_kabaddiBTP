import os
import random

random.seed(42)

dataset_root = r"C:\Users\aryan\Downloads\BTP\Updated_dataset"
output_dir = r"C:\Users\aryan\Downloads\BTP\mmaction2\data\kabaddi"

os.makedirs(output_dir, exist_ok=True)

classes = sorted([
    d for d in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, d))
])

class_to_idx = {cls: i for i, cls in enumerate(classes)}

train = []
val = []

for cls in classes:
    folder = os.path.join(dataset_root, cls)

    videos = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    random.shuffle(videos)

    if len(videos) == 1:
        train_videos = videos
        val_videos = []
    else:
        val_count = max(1, int(len(videos) * 0.2))
        val_videos = videos[:val_count]
        train_videos = videos[val_count:]

    for v in train_videos:
        train.append(
            f"{cls}/{v} {class_to_idx[cls]}"
        )

    for v in val_videos:
        val.append(
            f"{cls}/{v} {class_to_idx[cls]}"
        )

with open(os.path.join(output_dir, "train.txt"), "w") as f:
    f.write("\n".join(train))

with open(os.path.join(output_dir, "val.txt"), "w") as f:
    f.write("\n".join(val))

with open(os.path.join(output_dir, "classes.txt"), "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print("Done!")
print("Classes:", len(classes))
print("Training videos:", len(train))
print("Validation videos:", len(val))
print("\nClass Mapping:")
for c, i in class_to_idx.items():
    print(i, c)
