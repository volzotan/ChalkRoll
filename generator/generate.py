from datetime import datetime

from PIL import Image, ImageOps
import numpy as np

from io import StringIO

# all units in mm

PEN_DIAMETER      = 15
GANTRY_LENGTH     = 800
RESOLUTION_X      = PEN_DIAMETER
RESOLUTION_Y      = 1

TRIPLE_SCRUBBING  = True
OFFSET_LIFTER2    = 37

TWO_COLORS        = None

INPUT_IMAGE       = "../test6.png"

def generate(img):

    # remove alpha channel

    if img.mode == "RGBA":
        img_rgb = Image.new("RGB", img.size, (255, 255, 255))
        img_rgb.paste(img, mask=img.split()[3])
        img = img_rgb

    # rotate if portrait orientation

    if img.width < img.height:
        img = img.transpose()

    # detect colors

    img_arr = np.array(img, dtype=float)
    # compute the diff between R-G, B-G, and R-B channels. Max value when [0, 255, 255] = 255*2
    diff = np.abs(img_arr[:,:,0] - img_arr[:,:,1]) + np.abs(img_arr[:,:,1] - img_arr[:,:,2]) + np.abs(img_arr[:,:,0] - img_arr[:,:,2])
    diff = diff / 255*2.0
    num_diff = (diff > 0.10).sum()

    # if more than 10% of all pixels have a color diff greater than 10%:
    if num_diff > img.width*img.height*0.10:
        TWO_COLORS = True
    else:
        TWO_COLORS = False # grayscale image

    # extract layer(s)

    layers = []
    if TWO_COLORS:

        img_arr = np.array(img)

        mask_red = np.logical_and(img_arr[:,:,0] > 200, np.sum(img_arr, axis=2) < 300)
        mask_green = np.logical_and(img_arr[:,:,1] > 200, np.sum(img_arr, axis=2) < 300)

        layer0 = np.zeros([img.height, img.width], dtype=np.uint8)
        layer1 = np.zeros([img.height, img.width], dtype=np.uint8)

        layer0[mask_red] = 255
        layer1[mask_green] = 255
        
        # make sure the layers are mutually exclusive
        #layer1[mask_red] = 0
        
        layer0 = Image.fromarray(layer0).convert(mode="1")
        layers.append(layer0)
        layer1 = Image.fromarray(layer1.astype(np.uint8)).convert(mode="1")
        layers.append(layer1)
        
    else:
        img = img.convert(mode="1")
        img = ImageOps.invert(img)
        layers.append(img)

    # for i in range(len(layers)):
    #     print("layer {}".format(i))
    #     display(layers[i])

    resize_height = int(GANTRY_LENGTH / RESOLUTION_Y)
    resize_width = int(img.width / img.height * GANTRY_LENGTH / RESOLUTION_X)

    combined_image = np.zeros([resize_height, resize_width], dtype=np.uint8)
    combined_image_debug = np.zeros([resize_height, resize_width, 3], dtype=np.uint8)

    for i in range(len(layers)):
        layers[i] = layers[i].resize((
            resize_width,
            resize_height
        ))
        #), resample=Image.Resampling.LANCZOS)

        # debug view

        layer_array = np.array(layers[i]) * (i+1)
        combined_image = np.add(combined_image, layer_array.astype(np.uint8))

        color = [0, 0, 0]
        color[i] = 255
        combined_image_debug[layer_array > 0] = color

    # combined_image_debug[combined_image == 0] = [255, 255, 255]

    # debug_display(Image.fromarray(combined_image_debug))

    segments = []
    segment_start = None
    for x in range(combined_image.shape[1]):
        segments_line = []
        
        segment_start = None
        last_color = 0
        
        for y in range(combined_image.shape[0]):
            current_color = combined_image[y, x]

            if current_color != last_color:
                if segment_start is None:
                    segment_start = [x*RESOLUTION_X, y*RESOLUTION_Y]
                else:
                    segment = [segment_start, [x*RESOLUTION_X, y*RESOLUTION_Y], last_color]

                    # ignore very short lines t
                    #segment_length = abs(segment[0][1]-segment[1][1])
                    #if segment_length >= PEN_DIAMETER:
                    #    segments_line.append(segment)

                    # if lifter2 is used an offset for the Y-axis needs to added
                    if segment[2] == 2:
                        segment[0][1] += OFFSET_LIFTER2 * RESOLUTION_Y
                        segment[1][1] += OFFSET_LIFTER2 * RESOLUTION_Y
                    
                    segments_line.append(segment)

                    if current_color == 0:
                        segment_start = None
                    else:
                        segment_start = [x*RESOLUTION_X, y*RESOLUTION_Y]
                        
                last_color = current_color

        # column ends but line segment is unfinished
        if segment_start is not None:
            segment = [segment_start, [x*RESOLUTION_X, y*RESOLUTION_Y], last_color]
            
            if segment[2] == 2:
                segment[0][2] += OFFSET_LIFTER2 * RESOLUTION_Y
                segment[1][2] += OFFSET_LIFTER2 * RESOLUTION_Y
                    
            segments_line.append(segment)
            segment_start = None
        
        if len(segments_line) > 0:
            segments.append(segments_line)

    #for l in segments:
    #    for s in l:
    #        print(s)
    #    print("---")

    segments_reversed = []
    for i in range(len(segments)):
        line = segments[i]
        if i%2==0:
            segments_reversed.append(line)
        else:
            segments_reversed.append(list(reversed([[s[1], s[0], s[2]] for s in line])))
    segments = segments_reversed

    gcode_str = gcode(segments)

    output_img = Image.fromarray(combined_image_debug)
    output_img = output_img.resize((
        int(600*img.width/img.height), 
        int(600)
    ), resample=Image.Resampling.NEAREST)

    return {
        "image": output_img,
        "gcode": gcode_str
    }

    # print("number of lines: {}".format(len(segments)))


def write_to_file(filename, gcode):

    with open(filename, "w") as f:
        f.write(gcode)


def gcode(segments, params={}):
    
    FEEDRATE_X        = 2000
    FEEDRATE_Y        = 10000
    FEEDRATE_Z_RAISE  = 800
    FEEDRATE_Z_LOWER  = 6000
    ACCELERATION_Z    = 90
    RAISE_DISTANCE    = 20
    
    START_CMD              = """
G90                                 
G21                     
G28  
G92 X0 Y0 Z0

MANUAL_STEPPER STEPPER=lifter1 ENABLE=1
MANUAL_STEPPER STEPPER=lifter1 SET_POSITION=0
MANUAL_STEPPER STEPPER=lifter2 ENABLE=1
MANUAL_STEPPER STEPPER=lifter2 SET_POSITION=0

G1 F{feedrate}
"""
    
    MOVEX_CMD             = """
G1 X{x:.4f} F{feedrate}
"""

    MOVEY_CMD             = """
G1 Y{y:.4f} F{feedrate}
"""

#    LOWER_1_CMD           = """
#G1 Z{pos:.4f} F{feedrate}
#"""

    LOWER_1_CMD           = """
MANUAL_STEPPER STEPPER=lifter1 MOVE={pos:.4f} SPEED={feedrate} ACCEL={accel}
"""
    
    LOWER_2_CMD           = """
MANUAL_STEPPER STEPPER=lifter2 MOVE={pos:.4f} SPEED={feedrate} ACCEL={accel}
"""

#    RAISE_1_CMD           = """
#G1 Z{pos:.4f} F{feedrate}
#"""

    RAISE_1_CMD           = """
MANUAL_STEPPER STEPPER=lifter1 MOVE={pos:.4f} SPEED={feedrate} ACCEL={accel}
"""
    
    RAISE_2_CMD           = """
MANUAL_STEPPER STEPPER=lifter2 MOVE={pos:.4f} SPEED={feedrate} ACCEL={accel}
"""

    END_CMD               = """
G1 F{feedrate}
G1 X{x:.4f} 
G1 Y0 
G92 X0 Y0 Z0

MANUAL_STEPPER STEPPER=lifter1 ENABLE=0
MANUAL_STEPPER STEPPER=lifter2 ENABLE=0
"""
    
    f = StringIO()

    f.write("% ChalkRoll --- date: {} \n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    f.write("% gantry length: {}mm \n".format(GANTRY_LENGTH))
    f.write("% feedrate: X {} | Y {} | Z ^{} °{} @ {} mm/sec)\n".format(
        FEEDRATE_X, 
        FEEDRATE_Y, 
        FEEDRATE_Z_RAISE, 
        FEEDRATE_Z_LOWER,
        ACCELERATION_Z
    ))

    for key in params.keys():
        f.write("& info: {} : {}\n".format(key, params[key]))

    f.write(START_CMD.format(feedrate=FEEDRATE_X))
    f.write(RAISE_1_CMD.format(pos=RAISE_DISTANCE, feedrate=FEEDRATE_Z_RAISE, accel=ACCELERATION_Z))
    f.write(RAISE_2_CMD.format(pos=RAISE_DISTANCE, feedrate=FEEDRATE_Z_RAISE, accel=ACCELERATION_Z))

    for line in segments:

        if len(line) == 0:
            continue
        
        first_segment_in_line = line[0]

        f.write(MOVEX_CMD.format(
            x=first_segment_in_line[0][0],
            feedrate=FEEDRATE_X
        ))
        
        for s in line:

            f.write(MOVEY_CMD.format(
                y=s[0][1], feedrate=FEEDRATE_Y
            ))

            if s[2] == 1:
                f.write(LOWER_1_CMD.format(
                    pos=0, feedrate=FEEDRATE_Z_LOWER, accel=ACCELERATION_Z
                ))
            elif s[2] == 2:
                f.write(LOWER_2_CMD.format(
                    pos=0, feedrate=FEEDRATE_Z_LOWER, accel=ACCELERATION_Z
                ))
            else:
                raise Exception("unknown lifter head: {}".format(s[2]))
    
            f.write(MOVEY_CMD.format(
                y=s[1][1],
                feedrate=FEEDRATE_Y
            ))

            if TRIPLE_SCRUBBING:
                f.write(MOVEY_CMD.format(
                    y=s[0][1], feedrate=FEEDRATE_Y
                ))
                f.write(MOVEY_CMD.format(
                    y=s[1][1], feedrate=FEEDRATE_Y
                ))

            if s[2] == 1:
                f.write(RAISE_1_CMD.format(
                    pos=RAISE_DISTANCE, feedrate=FEEDRATE_Z_RAISE, accel=ACCELERATION_Z
                ))
            elif s[2] == 2:
                f.write(RAISE_2_CMD.format(
                    pos=RAISE_DISTANCE, feedrate=FEEDRATE_Z_RAISE, accel=ACCELERATION_Z
                ))
            else:
                raise Exception("unknown lifter head: {}".format(s[2]))

    last_line = segments[-1][-1]

    f.write(END_CMD.format(
        feedrate=FEEDRATE_X,
        x=last_line[1][0] + 100
    ))

    gcode_str = f.getvalue()
    f.close()
    return gcode_str


if __name__ == "__main__":
    result = generate(Image.open(INPUT_IMAGE))
    write_to_file("output.gcode", result["gcode"])
