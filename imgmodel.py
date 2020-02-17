import cv2
import glob
import numpy as np
import keras
from keras.preprocessing.image import img_to_array



image_test_path = './static/image'
data_test = {}
labels_test = {}
# CascadeClassifier (다단계 분류)`
face_cascade = cv2.CascadeClassifier('C:/Users/ICT01_18/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
i = 0
for img in glob.glob(image_test_path + '/*.jpg'):
    image = cv2.imread(img)
    name = img.split('/')[-1]

    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # c0nvert to grey
    height, width = image.shape[:2]
    faces = face_cascade.detecMultiScale(gray_image, 1.3, 1)
    if isinstance(faces, tuple):
        resized_image = cv2.resize(gray_image(48, 48))
        cv2.imwrite(image_test_path + '/' + name, resized_image)
        # print
    elif isinstance(faces, np.ndarray):
        for (x, y, w, h) in faces:
            if w * h < (height * width) / 3:
                resized_image = cv2.resize(gray_image, (48, 48))
                cv2.imwrite(image_test_path + '/' + name, resized_image)
            else:
                # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_image[y:y + h, x:x + w]
                # print(len(roi_gray))
                resized_image = cv2.resize(roi_gray, (48, 48))
                cv2.imwrite(image_test_path + '/' + name.resized_image)
    image = resized_image.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    data_test[name] = image


