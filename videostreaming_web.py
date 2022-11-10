from flask import Flask
from flask import render_template
from flask import Response
from flask import jsonify
import cv2
from camera_feed import generate

cap = cv2.VideoCapture(0)
sentence = ['']

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/_stuff',methods=['GET'])
def stuff():
    return jsonify(sentence_array=sentence)

@app.route("/video_feed")
def video_feed():
    return Response(generate(cap,sentence),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True)

cap.release()