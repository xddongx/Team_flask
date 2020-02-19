from PIL import Image
import os, glob, numpy as np
from keras.models import load_model
from keras import backend as K
import tensorflow as tf


os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 초기화할 GPU number
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(allow_growth=True)

K.clear_session()
tf.reset_default_graph()

test_dir = './image'
image_w =64
image_h =64

X = []
filenames =[]
files = glob.glob(test_dir+ '/' + '*.*')


for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((image_w, image_h))
    data = np.asarray(img)
    #     print(data)
    filenames.append(f)
    #     print(data)
    X.append(data)
#     print(T)

X = np.array(X)

model_dir = os.path.join('./model/multi_img_classification.model')
print(model_dir)
model = load_model(model_dir, compile=False)
X = X.reshape(X.shape[0], 64, 64, 3)

prediction = model.predict(X)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
# cnt = 0

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})  # >> 넘파이 출력옵션 변경하는것! (소수점3자리까지)
cnt = 0
for i in prediction:
    pre_ans = i.argmax()  # 예측 레이블  # argmax : 함수를 최대로 만들기 위한 x 값 --> 즉 첫번째에서는 3번째만 1이므로 2가 출력됨
    print(i)
    print(pre_ans)
    pre_ans_str = ''
    if pre_ans == 0:
        pre_ans_str = "afraid"
    elif pre_ans == 1:
        pre_ans_str = "anger"
    elif pre_ans == 2:
        pre_ans_str = "disgusted"
    elif pre_ans == 3:
        pre_ans_str = "happy"
    elif pre_ans == 4:
        pre_ans_str = "neutral"
    elif pre_ans == 5:
        pre_ans_str = "sad"

    else:
        pre_ans_str = "surprised"
    if i[0] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "한 기분으로 추정됩니다.")
    if i[1] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "한 기분으로 추정됩니다.")
    if i[2] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "한 기분으로 추정됩니다.")
    if i[3] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "한 기분으로 추정됩니다.")
    if i[4] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "한 기분으로 추정됩니다.")
    if i[5] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "한 기분으로 추정됩니다.")
    if i[3] >= 0.8: print("해당 " + filenames[cnt].split("\\")[1] + "이미지는 " + pre_ans_str + "한 기분으로 추정됩니다.")
    cnt += 1
