from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import searchfile
import model, emotions, text

app = Flask(__name__)

face_dir = os.path.join('./static/image/face')
print(face_dir)

@app.route('/')
def render_file():
   name = searchfile.search("./static/image/team")
   return render_template('home.html', filelist=name)


@app.route('/image', methods = ['GET', 'POST'])
def upload_image():
   if request.method == 'POST':
      f = request.files['file']
      f.save(face_dir+'/'+secure_filename(f.filename))
      print(f.filename)
      result = model.model_play()
      mind = emotions.emo(result[1])
      print(mind)
      return render_template('image.html', lists=result, name=f.filename, emotion = mind)
   return render_template('image.html')

@app.route('/text', methods = ['GET', 'POST'])
def upload_text():
   if request.method == 'POST':
      print('POST 들어온다')
      feel = request.form.get('text_data')
      print('txt가 무었인가?',feel)
      result = text.model_text(feel)
      print(result)
      mind = emotions.emo(result[1])
      print(mind)
      return render_template('text.html', texts=result, emotion=mind)
   return render_template('text.html')

if __name__ == '__main__':
    #서버 실행
   app.run(debug = False)
