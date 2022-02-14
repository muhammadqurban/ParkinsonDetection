from flask import Flask, request, render_template, make_response
from flask.helpers import url_for
from werkzeug.utils import secure_filename
import os
import predict

app = Flask(__name__)
uploadFolder = './uploads/'
app.config['UPLOAD_FOLDER'] = uploadFolder

@app.route('/', methods=['GET'])
def home():
    print('this is just a placeHolder')
    return make_response(render_template('fileUpload.html'))

@app.route('/detect', methods=['POST'])
def detect():
    print(request.files)
    file =  request.files['file1']
    nameA = secure_filename(file.filename)
    # os.path = './'
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], nameA))
    print("file has been uploaded.")
    # import predict
    return make_response(predict.predictFromImage('./uploads/'+nameA))


if __name__ == "__main__":
    app.run(debug=True)