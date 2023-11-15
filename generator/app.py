from flask import Flask
from flask import render_template
from flask import request
from PIL import Image

from . import generate

import io
import base64
from pathlib import Path

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():

    try:

        uploaded_filename = request.files["image"].filename

        uploaded_image = request.files["image"].read()
        uploaded_image = Image.open(io.BytesIO(uploaded_image))

        result = generate.generate(uploaded_image)
        
        img = result["image"]

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue())

        gcode_base64 = base64.b64encode(bytearray(result["gcode"], "utf-8"))

        gcode_filename = Path(uploaded_filename).stem + ".gcode"

        data = {
            "image_filename": uploaded_filename,
            "preview_image_data": img_base64.decode("utf-8"),
            "gcode_filename": gcode_filename,
            "gcode": result["gcode"],
            "gcode_base64": gcode_base64.decode("utf-8")
        }

        return render_template("result.html", data=data)

    except Exception as e:

        return render_template("error.html")