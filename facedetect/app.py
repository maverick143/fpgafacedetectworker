"""
Face detection deephi caffe model with webcam frames via RESTful API on fpga
"""

import numpy as np
import cv2
import flask
import argparse
from detect_api import Detect
import base64
import json

det=Detect()
det_model_path = ''
det_def_path = ''
det.model_init(det_model_path,det_def_path)

app = flask.Flask(__name__)

@app.route("/", methods=["GET","POST"])
def predict():
  global det  
  data = {"response": [],"success": False}
 
  # Serve GETs from Browser 
  if flask.request.method == "GET":
    return flask.render_template("index.html")

  if flask.request.method == "POST":
    try:
      dat = json.loads(flask.request.form.get("image"))
      npimg = np.asarray(dat["data"],dtype=np.uint8)
      img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
      image_resize = cv2.resize(img, (320, 320), interpolation = cv2.INTER_LINEAR)
      face_rects = det.detect(image_resize)
      response = face_rects
      data["success"] = True
      data["response"] = response
    except:
      data["response"] = "Could Not Decode Image"

  return flask.jsonify(data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--port', default=8080)
  
  args = vars(parser.parse_args())

  print("Loading FPGA with image...")
  #det=Detect()
  #det_model_path = ''
  #det_def_path = ''
  #det.model_init(det_model_path,det_def_path)

  print("Starting Flask Server...")
  app.run(port=args["port"],host="0.0.0.0")
  
