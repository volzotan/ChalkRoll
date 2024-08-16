from flask import Flask
from flask import render_template
from flask import request
from PIL import Image

try:
    import processing
except ImportError as e:
    from . import processing

import io
import base64
from pathlib import Path

app = Flask("generator")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload", methods=["POST"])
def upload():

    try:

        uploaded_filename = request.files["image"].filename
        uploaded_image = request.files["image"].read()

        if len(uploaded_image) == 0:
            raise Exception("image missing")

        uploaded_image = Image.open(io.BytesIO(uploaded_image))

        machine_type = request.form["machine_type"].upper()
        try:
            machine_type = processing.Gcode_type(machine_type)
        except Exception as e:
            raise Exception(f"unknown machine_type: {machine_type}")

        tool_type = request.form["tool_type"].upper()
        try:
            tool_type = processing.Tool_type(tool_type)
        except Exception as e:
            raise Exception(f"unknown tool_type: {tool_type}")

        option_triplescrubbing = False
        if "triplescrubbing" in request.form and request.form["triplescrubbing"].upper() in ["TRUE", "1"]:
            option_triplescrubbing = True

        option_highspeed = False
        if "highspeed" in request.form and request.form["highspeed"].upper() in ["TRUE", "1"]:
            option_highspeed = True

        result = processing.generate(
            uploaded_image, 
            gcode_type=machine_type, 
            tool_type=tool_type, 
            triple_scrubbing=option_triplescrubbing,
            highspeed=option_highspeed
        )
        
        img = result["image"]

        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue())

        gcode_base64 = base64.b64encode(bytearray(result["gcode"], "utf-8"))
        gcode_filename = f"{Path(uploaded_filename).stem}_{machine_type}_{tool_type}.gcode"

        data = {
            "image_filename": uploaded_filename,
            "preview_image_data": img_base64.decode("utf-8"),
            "gcode_filename": gcode_filename,
            "gcode": result["gcode"],
            "gcode_base64": gcode_base64.decode("utf-8"),
            "stats": result["stats"]
        }

        return render_template("result.html", data=data)

    except Exception as e:
        raise e
        # return render_template("error.html")


if __name__ == "__main__":
   app.run(debug=False, host="0.0.0.0")