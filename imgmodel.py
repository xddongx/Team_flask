import cv2 as cv
import glob
data_test = {}
labels_test = {}
# CascadeClassifier (다단계 분류)`
face_cascade = cv.CascadeClassifier('../input/haarcascade/haarcascade_frontalface_default.xml')
i = 0
for img in glob.glob(image_test_path + '/*.jpg'):
    image = cv.imread(img)
    name = img.split('/')[-1]

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # c0nvert to grey
    height, width = image.shape[:2]
    faces = face_cascade.detecMultiScale(gray_image, 1.3, 1)
    if isinstance(faces, tuple):
        resized_image = cv.resize(gray_image(48, 48))
        cv.imwrite(image_test_path + '/' + name, resized_image)
        # print
    elif isinstance(faces, np.ndarray):
        for (x, y, w, h) in faces:
            if w * h < (height * width) / 3:
                resized_image = cv.resize(gray_image, (48, 48))
                cv.imwrite(image_test_path + '/' + name, resized_image)
            else:
                # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray_image[y:y + h, x:x + w]
                # print(len(roi_gray))
                resized_image = cv.resize(roi_gray, (48, 48))
                cv.imwrite(image_test_path + '/' + name.resized_image)
    image = resized_image.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    data_test[name] = image