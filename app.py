from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import model, searchfile
app = Flask(__name__)

upload_dir = os.path.join('./static/image')
print(upload_dir)

@app.route('/')
def render_file():
   name = searchfile.search("./static/image/team")
   return render_template('home.html', filelist=name)


@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(upload_dir+'/'+secure_filename(f.filename))
      result = model.model_play()
      return render_template('upload.html', lists=result)
   return render_template('upload.html')

if __name__ == '__main__':
    #서버 실행
   app.run(debug = True)
