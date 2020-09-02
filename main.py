from easyocr.custom_easyocr import CustomReader
from easyocr.easyocr import Reader
import os
import cv2
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'
TEST_DATA_PATH = "./DATA"
file_list = os.listdir(TEST_DATA_PATH)

image_list = []
for file_name in file_list:
    file_path = TEST_DATA_PATH + "/" + file_name
    print(file_name)
    if file_name ==".ipynb_checkpoints":
        continue
    image = cv2.imread(file_path)
    image_list.append(image)

custom_reader = CustomReader(['ko', 'en'], )
reader = Reader(['ko', 'en'], )

start_time = time.time()

result = custom_reader.readtext(image_list, batch_size=128)
# print(result)
end_time = time.time()
print(end_time-start_time)

start_time = time.time()
for image in image_list:
    result = reader.readtext(image, batch_size=10)
    print(result)

end_time = time.time()
print(end_time - start_time)
