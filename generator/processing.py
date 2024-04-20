from datetime import datetime
from io import StringIO
import math

from PIL import Image, ImageOps, ImageDraw
import numpy as np

GCODE_TYPE_KLIPPER      = "MACHINE_KLIPPER"
GCODE_TYPE_FLUIDNC      = "MACHINE_FLUIDNC"
GCODE_TYPES             = [GCODE_TYPE_KLIPPER, GCODE_TYPE_FLUIDNC]
TOOL_TYPE_STICK         = "TOOL_STICK"
TOOL_TYPE_CAN           = "TOOL_CAN"
TOOL_TYPES              = [TOOL_TYPE_STICK, TOOL_TYPE_CAN]

# all units in mm

CONFIG_STICK = {
    "TOOL_NAME"         : "chalk stick",
    "TOOL_DIAMETER"     : 15,
    "OFFSET_LIFTER2"    : 37,
    "GANTRY_LENGTH"     : 800, # 840 - OFFSET_LIFTER2
    "RESOLUTION_X"      : 15,
    "RESOLUTION_Y"      : 2,
}

CONFIG_CAN = {
    "TOOL_NAME"         : "chalk can",
    "TOOL_DIAMETER"     : 32,
    "OFFSET_LIFTER2"    : 50,
    "GANTRY_LENGTH"     : 800,
    "RESOLUTION_X"      : 32,
    "RESOLUTION_Y"      : 1,
}

# TRIPLE_SCRUBBING    = True
MAX_DISTANCE_X_OPTIMIZATION = 15
TWO_COLORS              = None

SERVO_POS_WRITE         = 80
SERVO_POS_IDLE          = 0
SERVO_WAIT_TIME         = 150

DEBUG_INPUT_IMAGE       = "../test6.png"
DEBUG_INPUT_IMAGE       = "../HelloWorld_long.png"


def generate(img, gcode_type=GCODE_TYPE_KLIPPER, tool_type=TOOL_TYPE_STICK, triple_scrubbing=False, highspeed=False):

    config = None

    if tool_type == TOOL_TYPE_STICK:
        config = CONFIG_STICK
    elif tool_type == TOOL_TYPE_CAN:
        config = CONFIG_CAN
    else:
        raise Exception("unknown tool_type: {}".format(tool_type))

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

    resize_height = int(config["GANTRY_LENGTH"] / config["RESOLUTION_Y"])
    resize_width = int(img.width / img.height * config["GANTRY_LENGTH"] / config["RESOLUTION_X"])

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

    # PIL coordinate system is top-left, CNC coordinate system is bottom-left, so
    # we need flip the image along the X axis to invert Y coordinates
    combined_image = np.flip(combined_image, axis=0)

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
                    segment_start = [x*config["RESOLUTION_X"], y*config["RESOLUTION_Y"]]
                else:
                    segment = [segment_start, [x*config["RESOLUTION_X"], y*config["RESOLUTION_Y"]], last_color]

                    # ignore very short lines t
                    #segment_length = abs(segment[0][1]-segment[1][1])
                    #if segment_length >= config["TOOL_DIAMETER"]:
                    #    segments_line.append(segment)

                    # if lifter2 is used an offset for the Y-axis needs to added
                    if segment[2] == 2:
                        segment[0][1] += config["OFFSET_LIFTER2"]
                        segment[1][1] += config["OFFSET_LIFTER2"]
                    
                    segments_line.append(segment)

                    if current_color == 0:
                        segment_start = None
                    else:
                        segment_start = [x*config["RESOLUTION_X"], y*config["RESOLUTION_Y"]]
                        
                last_color = current_color

        # column ends but line segment is unfinished
        if segment_start is not None:
            segment = [segment_start, [x*config["RESOLUTION_X"], y*config["RESOLUTION_Y"]], last_color]
            
            if segment[2] == 2:
                segment[0][2] += config["OFFSET_LIFTER2"]
                segment[1][2] += config["OFFSET_LIFTER2"]
                    
            segments_line.append(segment)
            segment_start = None
        
        if len(segments_line) > 0:
            segments.append(segments_line)

    segments_reversed = []
    for i in range(len(segments)):
        line = segments[i]
        if i%2==0:
            segments_reversed.append(line)
        else:
            segments_reversed.append(list(reversed([[s[1], s[0], s[2]] for s in line])))
    segments = segments_reversed

    total_distance = 0
    for segments_line in segments:
        for s in segments_line:
            total_distance += abs((s[1][1] - s[0][1]))

    workspace_dimensions = [resize_width * config["RESOLUTION_X"], resize_height * config["RESOLUTION_Y"]]

    stats = {
        "workspace_dimensions": workspace_dimensions,
        "total_distance": total_distance
    }

    gcode_str = gcode(
        segments, 
        gcode_type, 
        config, 
        triple_scrubbing,
        highspeed
    )

    # visualize: combine the distorted images from processing and resize to a 
    # resonable size to present them as a preview
    # output_img = Image.fromarray(combined_image_debug)
    # output_img = output_img.resize((
    #     int(600*img.width/img.height), 
    #     int(600)
    # ), resample=Image.Resampling.NEAREST)

    # visualize: or rather use an empty image and draw just the toolpaths
    output_img = Image.new("RGB", [int(600*img.width/img.height), int(600)])
    output_img = draw_toolpaths(
        output_img, 
        segments, 
        config["GANTRY_LENGTH"],
        config["TOOL_DIAMETER"],
        config["OFFSET_LIFTER2"])

    return {
        "image": output_img,
        "gcode": gcode_str, 
        "stats": stats
    }


def draw_toolpaths(base_img, segments, scaling, tool_diameter, offset_lifter):

    draw = ImageDraw.Draw(base_img)

    scaling_y = (1.0 / scaling) * base_img.height
    scaling_x = scaling_y

    fills = [(255, 255, 255), (255, 0, 0), (0, 255, 0)]
    line_width = int(tool_diameter*0.6)

    for segments_line in segments:
        for s in segments_line:

            offset = 0
            if s[2] == 2:
                offset = offset_lifter

            draw.line((
                    (s[0][0] + tool_diameter/2.0) * scaling_x, base_img.height - (s[0][1] - offset) * scaling_y, 
                    (s[1][0] + tool_diameter/2.0) * scaling_x, base_img.height - (s[1][1] - offset) * scaling_y
                ), fill=fills[s[2]], width=line_width)

            draw.ellipse((
                    (s[0][0] + tool_diameter/2.0 - line_width/2.0) * scaling_x,  base_img.height - (s[0][1] - offset + line_width/2.0) * scaling_y, 
                    (s[0][0] + tool_diameter/2.0 + line_width/2.0) * scaling_x,  base_img.height - (s[0][1] - offset - line_width/2.0) * scaling_y
                ), fill=fills[s[2]], width=-1)

            draw.ellipse((
                    (s[1][0] + tool_diameter/2.0 - line_width/2.0) * scaling_x,  base_img.height - (s[1][1] - offset + line_width/2.0) * scaling_y, 
                    (s[1][0] + tool_diameter/2.0 + line_width/2.0) * scaling_x,  base_img.height - (s[1][1] - offset - line_width/2.0) * scaling_y
                ), fill=fills[s[2]], width=-1)

    return base_img


def write_to_file(filename, gcode):

    with open(filename, "w") as f:
        f.write(gcode)


def _move_arm(number, pos, feedrate, acceleration, gcode_type):

    # return ""

    if gcode_type == GCODE_TYPE_KLIPPER:
        lifters = ["lifter1", "lifter2"]

        return "MANUAL_STEPPER STEPPER={lifter} MOVE={pos:.4f} SPEED={feedrate} ACCEL={accel}\n".format(
            lifter=lifters[number],
            pos=pos, 
            feedrate=feedrate, 
            accel=acceleration
        )

    if gcode_type == GCODE_TYPE_FLUIDNC:
        lifters = ["Z", "A"]

        return "G1 {lifter}{pos:.4f} F{feedrate}\n".format(
            lifter=lifters[number],
            pos=pos,
            feedrate=feedrate
        )

def _move_servo(number, pos, gcode_type):

    if gcode_type == GCODE_TYPE_KLIPPER:
        servos = ["servo_can1", "servo_can2"]

        servo_wait = SERVO_WAIT_TIME
        cmd = ""

        if pos == 0:
            servo_wait = 0

        cmd += "SET_SERVO SERVO={} ANGLE={}\n".format(servos[number], pos)

        if servo_wait > 0:
            cmd += "G4 P{}\n".format(servo_wait)

        return cmd

    if gcode_type == GCODE_TYPE_FLUIDNC:
        servos = ["Z", "A"]

        return "G1 {}{}\n".format(
            servos[number],
            pos
        )


def gcode(segments, gcode_type, config, triple_scrubbing, highspeed, params={}):
    
    FEEDRATE_X              = 1000
    FEEDRATE_Y              = 22000
    FEEDRATE_Z_RAISE        = 1000
    FEEDRATE_Z_LOWER        = 2000
    ACCELERATION_Z_RAISE    = 100
    ACCELERATION_Z_LOWER    = 800
    RAISE_DISTANCE          = 20

    if gcode_type == GCODE_TYPE_FLUIDNC:
        FEEDRATE_X          = 2000
        FEEDRATE_Y          = 22000
        FEEDRATE_Z_RAISE    = 5000
        FEEDRATE_Z_LOWER    = 10000
        RAISE_DISTANCE      = 90

    if highspeed:
        FEEDRATE_X          = 3000
        FEEDRATE_Y          = 45000
    
    MOVEX_CMD               = "G1 X{x:.4f} F{feedrate}\n"
    MOVEY_CMD               = "G1 Y{y:.4f} F{feedrate}\n"

    START_CMD               = """
G90                                 
G21                     
G28  
G92 X0 Y0 Z0

G1 F{feedrate}
"""

    START_CMD_KLIPPER_TOOL_STICK    = """
MANUAL_STEPPER STEPPER=lifter1 ENABLE=1
MANUAL_STEPPER STEPPER=lifter1 SET_POSITION=0
MANUAL_STEPPER STEPPER=lifter2 ENABLE=1
MANUAL_STEPPER STEPPER=lifter2 SET_POSITION=0
"""

    START_CMD_KLIPPER_TOOL_CAN      = """"""

    END_CMD                         = """
G1 F{feedrate}
G1 X{x:.4f} 
G1 Y0 
G92 X0 Y0 Z0
"""

    END_CMD_KLIPPER_TOOL_STICK      = """
MANUAL_STEPPER STEPPER=lifter1 ENABLE=0
MANUAL_STEPPER STEPPER=lifter2 ENABLE=0
"""

    END_CMD_KLIPPER_TOOL_CAN        = """"""

    
    f = StringIO()

    _write_gcode_comment(f, gcode_type, "ChalkRoll --- date: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    _write_gcode_comment(f, gcode_type, "gcode type: {}".format(gcode_type))
    _write_gcode_comment(f, gcode_type, "tool type: {}".format(config["TOOL_NAME"]))
    _write_gcode_comment(f, gcode_type, "gantry length: {}mm".format(config["GANTRY_LENGTH"]))
    if gcode_type == GCODE_TYPE_KLIPPER:
        _write_gcode_comment(f, gcode_type, "feedrate: X {} | Y {} | Z ^{} °{} @ {} mm/sec".format(
            FEEDRATE_X, 
            FEEDRATE_Y, 
            FEEDRATE_Z_RAISE, 
            FEEDRATE_Z_LOWER,
            ACCELERATION_Z_RAISE
        ))
    elif gcode_type == GCODE_TYPE_FLUIDNC:
        _write_gcode_comment(f, gcode_type, "feedrate: X {} | Y {} | Z ^{} °{}".format(
            FEEDRATE_X, 
            FEEDRATE_Y, 
            FEEDRATE_Z_RAISE, 
            FEEDRATE_Z_LOWER
        ))

    for key in params.keys():
        f.write("& info: {} : {}\n".format(key, params[key]))

    f.write(START_CMD.format(feedrate=FEEDRATE_X))

    if gcode_type == GCODE_TYPE_KLIPPER:
        if config == CONFIG_STICK:
            f.write(START_CMD_KLIPPER_TOOL_STICK)
            f.write(_move_arm(0, RAISE_DISTANCE, FEEDRATE_Z_RAISE, ACCELERATION_Z_RAISE, GCODE_TYPE_KLIPPER))
            f.write(_move_arm(1, RAISE_DISTANCE, FEEDRATE_Z_RAISE, ACCELERATION_Z_RAISE, GCODE_TYPE_KLIPPER))
        if config == CONFIG_CAN:
            f.write(START_CMD_KLIPPER_TOOL_CAN)

    elif gcode_type == GCODE_TYPE_FLUIDNC:
        if config == CONFIG_STICK:
            f.write(_move_arm(0, RAISE_DISTANCE, FEEDRATE_Z_RAISE, None, GCODE_TYPE_FLUIDNC))
            f.write(_move_arm(1, RAISE_DISTANCE, FEEDRATE_Z_RAISE, None, GCODE_TYPE_FLUIDNC))

    for i in range(len(segments)):

        column = segments[i]
        next_column = None
        if i < len(segments)-1:
            next_column = segments[i+1]

        if len(column) == 0:
            continue

        # move to first column
        f.write(MOVEX_CMD.format(
            x=column[0][0][0],
            feedrate=FEEDRATE_X
        ))
        
        raise_after_column = True

        for j in range(len(column)):
            s = column[j]

            # MOVE TO START

            f.write(MOVEY_CMD.format(
                y=s[0][1], feedrate=FEEDRATE_Y
            ))

            # LOWER

            if config == CONFIG_STICK:
                if raise_after_column:
                    if s[2] == 1:
                        f.write(_move_arm(0, 0, FEEDRATE_Z_LOWER, ACCELERATION_Z_LOWER, gcode_type))
                    elif s[2] == 2:
                        f.write(_move_arm(1, 0, FEEDRATE_Z_LOWER, ACCELERATION_Z_LOWER, gcode_type))
                    else:
                        raise Exception("unknown lifter head: {}".format(s[2]))
            if config == CONFIG_CAN:
                if s[2] == 1:
                    f.write(_move_servo(0, SERVO_POS_WRITE, gcode_type))
                elif s[2] == 2:
                    f.write(_move_servo(1, SERVO_POS_WRITE, gcode_type))
                else:
                    raise Exception("unknown can servo: {}".format(s[2]))

            # MOVE

            f.write(MOVEY_CMD.format(
                y=s[1][1],
                feedrate=FEEDRATE_Y
            ))

            if triple_scrubbing:
                f.write(MOVEY_CMD.format(
                    y=s[0][1], feedrate=FEEDRATE_Y
                ))
                f.write(MOVEY_CMD.format(
                    y=s[1][1], feedrate=FEEDRATE_Y
                ))

            # CHECK IF RAISE IS NECESSARY

            raise_after_column = True
            # if we're looking at the last segment of the current column
            if j == len(column)-1:
                if not next_column is None:
                    # and the next column is right next to the current one
                    diff_X = abs(s[1][0] - next_column[0][0][0])
                    # and first segment in next column starts at same Y coordinate
                    diff_Y = abs(s[1][1] - next_column[0][0][1])
                    if diff_X <= MAX_DISTANCE_X_OPTIMIZATION and diff_Y < MAX_DISTANCE_X_OPTIMIZATION:
                        # and with the same color
                        if s[2] == next_column[0][2]:
                            # do not raise
                            raise_after_column = False

            # RAISE

            if config == CONFIG_STICK:
                if raise_after_column:
                    if s[2] == 1:
                        f.write(_move_arm(0, RAISE_DISTANCE, FEEDRATE_Z_RAISE, ACCELERATION_Z_RAISE, gcode_type))
                    elif s[2] == 2:
                        f.write(_move_arm(1, RAISE_DISTANCE, FEEDRATE_Z_RAISE, ACCELERATION_Z_RAISE, gcode_type))
                    else:
                        raise Exception("unknown lifter head: {}".format(s[2]))
            if config == CONFIG_CAN:
                if s[2] == 1:
                    f.write(_move_servo(0, SERVO_POS_IDLE, gcode_type))
                elif s[2] == 2:
                    f.write(_move_servo(1, SERVO_POS_IDLE, gcode_type))
                else:
                    raise Exception("unknown can servo: {}".format(s[2]))

    last_line = segments[-1][-1]
    f.write(END_CMD.format(
        feedrate=FEEDRATE_X,
        x=last_line[1][0] + 100
    ))

    if gcode_type == GCODE_TYPE_KLIPPER:
        if config == CONFIG_STICK:
            f.write(END_CMD_KLIPPER_TOOL_STICK)
        if config == CONFIG_CAN:
            f.write(END_CMD_KLIPPER_TOOL_CAN)
    elif gcode_type == GCODE_TYPE_FLUIDNC:
        pass

    gcode_str = f.getvalue()
    f.close()
    return gcode_str


def _write_gcode_comment(f, gcode_type, msg):

    if gcode_type == GCODE_TYPE_KLIPPER:
        f.write("% ")
        f.write(msg)
        f.write("\n")
    elif gcode_type == GCODE_TYPE_FLUIDNC:
        f.write("( ")
        f.write(msg)
        f.write(" )\n")


if __name__ == "__main__":
    result = generate(Image.open(DEBUG_INPUT_IMAGE), gcode_type=GCODE_TYPE_FLUIDNC)
    print(result)
    write_to_file("output.gcode", result["gcode"])
