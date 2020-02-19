from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import models, searchfile


app = Flask(__name__)

upload_dir = os.path.join('./static/image')
print(upload_dir)

@app.route('/')
def render_file():
   name = searchfile.search("./static/image/team")
   return render_template('home.html', filelist=name)


@app.route('/image', methods = ['GET', 'POST'])
def upload_image():
   if request.method == 'POST':
      f = request.files['file']
      f.save(upload_dir+'/'+secure_filename(f.filename))
      print(f.filename)
      result = models.model_play()
      return render_template('image.html', lists=result, name=f.filename)
   return render_template('image.html')

@app.route('/text', methods = ['GET', 'POST'])
def upload_text():
   # if request.method == 'POST':
      # f = request.files['text']
      # f.save(upload_dir+'/'+secure_filename(f.filename))
      # print(f.filename)
      # result = models.model_play()
      # return render_template('image.html', lists=result, name=f.filename)
   return render_template('text.html')

if __name__ == '__main__':
    #서버 실행
   app.run(debug = True)
