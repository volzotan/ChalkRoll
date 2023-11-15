from flask import Flask
from flask import render_template
from flask import request

from . import generate

from PIL import Image
import io
import base64

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():

    uploaded_image = request.files["image"].read()
    uploaded_image = Image.open(io.BytesIO(uploaded_image))

    # result = generate.generate(Image.open(generate.INPUT_IMAGE))
    result = generate.generate(uploaded_image)
    
    img = result["image"]

    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_base64 = base64.b64encode(img_bytes.getvalue())

    gcode_base64 = base64.b64encode(bytearray(result["gcode"], "utf-8"))

    data = {
        "image_data": img_base64.decode("utf-8"),
        "gcode": result["gcode"],
        "gcode_base64": gcode_base64.decode("utf-8")
    }

    return render_template("result.html", data=data)
