from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 초기화할 GPU number
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth=True)



def model_play():
    test_dir = '../../static/image'
    image_w = 64
    image_h = 64

    result = []

    X = []
    filenames = []
    files = glob.glob(test_dir + '/*.*')
    print('files', files)
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert('RGB')
        img = img.resize((image_w, image_h))
        data = np.asarray(img)
        filenames.append(f)

        X.append(data)

    X = np.array(X)
    model = load_model("../../model/multi_img_classification.model", compile=False)
    print('model', model)
    prediction = model.predict(X)
    print('predict : {}'.format(prediction))
    np.set_printoptions(formatter={'float': lambda x: '{0:0.3f}'.format(x)})
    cnt = 0

    lists = []
    for i in prediction:
        pre_ans = i.argmax()  # 예측 레이블
        pre_ans_str = ''
        print(pre_ans)
        if pre_ans == 0:
            pre_ans_str = "캔"
        elif pre_ans == 1:
            pre_ans_str = "플라스틱"
        elif pre_ans == 2:
            pre_ans_str = "유리"
        else:
            pre_ans_str = "스티로품"

        if i[0] >= 0.8:
            df = "해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다."
            lists.append(df)

        if i[1] >= 0.8:
            df = "해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "으로 추정됩니다."
            lists.append(df)
        if i[2] >= 0.8:
            df = "해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다."
            lists.append(df)
        if i[3] >= 0.8:
            df = "해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "로 추정됩니다."
            lists.append(df)
        cnt += 1

    lists
    return lists

print(model_play())
#
# def search(dirname):
#     filenames = os.listdir(dirname)
#     list = []
#     for filename in filenames:
#         full_filename = os.path.join(dirname, filename)
#         list.append(full_filename)
#     print(len(list))
#
# search("./static/image/member")