import tensorflow as tf
import matplotlib.pyplot as plt

# =========================
# 1️⃣ Load the image
# =========================
img_path = 'images/catdog.jpg'  # Update to your image path
img = plt.imread(img_path)       # Read the image into an array
img_height, img_width = img.shape[:2]

print(f"Image size: width={img_width}, height={img_height}")

# =========================
# 2️⃣ Define bounding boxes
# Format: [x1, y1, x2, y2] (upper-left, lower-right)
# These can be from any source; we will scale if needed
dog_bbox_orig = [60, 45, 378, 516]   # example coordinates
cat_bbox_orig = [400, 112, 655, 493] # example coordinates

# =========================
# 3️⃣ Scale bounding boxes (if needed)
# Assume original coordinates were made for an image of size 720x540
original_width = 720
original_height = 540

scale_x = img_width / original_width
scale_y = img_height / original_height

dog_bbox = [dog_bbox_orig[0]*scale_x, dog_bbox_orig[1]*scale_y,
            dog_bbox_orig[2]*scale_x, dog_bbox_orig[3]*scale_y]

cat_bbox = [cat_bbox_orig[0]*scale_x, cat_bbox_orig[1]*scale_y,
            cat_bbox_orig[2]*scale_x, cat_bbox_orig[3]*scale_y]

# =========================
# 4️⃣ Function to draw bounding boxes
# =========================
def draw_bbox(ax, bbox, color, label=None):
    """
    ax    : matplotlib axes
    bbox  : list of [x1, y1, x2, y2]
    color : box color
    label : optional text label
    """
    rect = plt.Rectangle(
        (bbox[0], bbox[1]),        # xy = upper-left corner
        bbox[2]-bbox[0],           # width
        bbox[3]-bbox[1],           # height
        fill=False, edgecolor=color, linewidth=2
    )
    ax.add_patch(rect)
    if label:
        ax.text(bbox[0], bbox[1]-5, label, color=color, fontsize=12, weight='bold')

# =========================
# 5️⃣ Display image with bounding boxes
# =========================
fig, ax = plt.subplots()
ax.imshow(img)

draw_bbox(ax, dog_bbox, 'blue', 'Dog')
draw_bbox(ax, cat_bbox, 'red', 'Cat')

plt.show()