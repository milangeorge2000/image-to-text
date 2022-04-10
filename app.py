from flask import Flask,redirect, url_for, request, render_template,jsonify
# from werkzeug.utils import secure_filename
import os,io
from google.cloud.vision_v1 import types
from google.cloud import vision
import pandas as pd
# from flask_ngrok import run_with_ngrok


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"credentials.json"


# Define a flask app
app = Flask(__name__)

# run_with_ngrok(app)


@app.route('/', methods=['GET'])
def index():
    # Main page
     return render_template('index.html')


def get_response(file):
    with io.open(file,"rb") as img_file:
       content = img_file.read()
    image = types.Image(content=content)
    client = vision.ImageAnnotatorClient()
    response = client.text_detection(image=image)
    text = response.text_annotations
    return text





@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        # Get the file from post request
        f.save("img.jpg")
        df = pd.DataFrame(columns=['locale','description'])
        text = get_response("img.jpg")
        for t in text:
            df = df.append(dict(locale=t.locale,description=t.description),ignore_index=True)
        
        return render_template('prediction.html',prediction=df['description'][0])

#         return jsonify(df['description'][0])
#         # print(df['description'][0])

        


if __name__ == '__main__':
    app.run(debug=False)
