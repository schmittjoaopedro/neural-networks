# Imports
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
import cv2
import easyocr
import imutils

annot = pd.read_parquet('dataset/annot.parquet')
images = pd.read_parquet('dataset/img.parquet')


def get_image(idx):
    image_id = images.iloc[idx].id
    image_path = f"dataset/train_val_images/train_images/{image_id}.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_annot = annot.query("image_id == @image_id")
    texts = image_annot["utf8_string"].values

    return image, texts


def rotate_image(image):
    results = []
    results.append(image)
    for angle in [45, 90, -45, -90]:
        results.append(imutils.rotate_bound(image, angle))
    return results


def predict_easyocr(image):
    easyocr_reader = easyocr.Reader(['en'])
    all_texts = []
    for im in rotate_image(image):
        texts = easyocr_reader.readtext(im, detail=0)
        texts = [pred.split() for pred in texts]
        texts = [item for sublist in texts for item in sublist]
        all_texts.extend(texts)
    all_texts = [i.lower() for i in all_texts]
    all_texts = list(set(all_texts))
    return all_texts


def compare_easyocr_output(expect_output, easyocr_output, log=False):
    expect_output = set([i.lower() for i in expect_output])
    easyocr_output = set([i.lower() for i in easyocr_output])
    if log:
        print(f"Dataset: {expect_output}")
        print(f"EasyOCR: {easyocr_output}")
    # Compare matching percentage
    matches_count = 0
    matches_text = []
    for i in expect_output:
        if i in easyocr_output:
            matches_count += 1
            matches_text.append(i)
    accuracy = matches_count / len(expect_output)
    if log:
        print(f"Matches: {matches_text}")
        print(f"Accuracy: {accuracy}")
    return matches_text, len(expect_output), matches_count, len(easyocr_output), accuracy


# image_idx = 25
# print(f"Image Index: {image_idx}")
# image, expected_output = get_image(image_idx)
# easyocr_output = predict_easyocr(image)
# accuracy, matches = compare_easyocr_output(expected_output, easyocr_output, log=True)
# plt.imshow(image)
# plt.grid(None)
# plt.show()

# For all images in the dataset
start_idx = 0
try:
    with open("results.csv", "r") as results_csv:
        for line in results_csv:
            start_idx = int(line.split(",")[0]) + 1
except:
    start_idx = 0

for i in range(start_idx, len(images)):
    image, expected_output = get_image(i)
    easyocr_output = predict_easyocr(image)
    texts, expected, matches, total, accuracy = compare_easyocr_output(expected_output, easyocr_output)
    print(f"Image Index: {i} | Accuracy: {accuracy}")
    with open("results.csv", "a") as results_csv:
        results_csv.write(f"{i},{expected},{matches},{total},{accuracy}\n")

accuracies = []
with open("results.csv", "r") as results_csv:
    for line in results_csv:
        accuracies.append(float(line.split(",")[4]))
print(f"Average Accuracy: {sum(accuracies) / len(accuracies)}")
