import tensorflow as tf
import numpy as np
import pandas as pd


music = pd.read_csv('./static/csv/music.csv', header=None)

music.columns = ['emotion', 'name', 'title', 'url']

emotions = music.emotion.unique()

print(emotions)

def emotion(e):
    if e == '불안한':
        emo = emotions[0]
    elif e == '화난':
        emo = emotions[1]
    elif e == '역겨운':
        emo = emotions[2]
    elif e == '행복한':
        emo = emotions[3]
    elif e == '슬픈':
        emo = emotions[5]
    elif e == '놀란':
        emo = emotions[6]
    else:
        emo = emotions[4]

    emotion_ = music[music.emotion == emo]
    rand_list = emotion_.sample()
    rand_list.emotion = e
    return rand_list




def emo(t):
    list = emotion(t).values[0]
    return list

# list = emo('슬픈')
# print(type(list))
# print(list[0])
# print(list[1])
# print(list[2])