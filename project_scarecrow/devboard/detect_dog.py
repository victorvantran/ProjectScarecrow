# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# With credit to Justin Mitchel from codingforentrepreneurs

"""
Libraries installed:

sudo apt-get install python3-dev
sudo apt-get install bluetooth libbluetooth-dev
sudo python3 -m pip install pybluez

"""

import argparse
import collections
import common
import cv2
import numpy as np
import os
from PIL import Image
import re
import tflite_runtime.interpreter as tflite

import sys
import periphery
import time
import bluetooth


"""_____________define_____________"""

ENVIRONMENT = '/sys/firmware/devicetree/base/model'

BLUETOOTH_MODULE_ADDRESS = '00:14:03:06:21:18' # DSD TECH HM-10 Bluetooth 4.0 BLE Address
BLUETOOTH_MODULE_PORT = 1


"""_____________defaults_____________"""

DEFAULT_MODEL_DIR = 'models'
DEFAULT_MODEL = 'dog_edgetpu.tflite'
DEFAULT_LABELS = 'dog_labels.txt'
DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.6
DEFAULT_CAPTURE_MODE = 'camera' #'video' 'camera'

DEFAULT_INPUT_VIDEO_DIR = 'input_videos'
DEFAULT_INPUT_VIDEO_FILE = 'dog_park.mp4'

DEFAULT_OUTPUT_VIDEO_DIR = 'output_videos'
DEFAULT_OUTPUT_VIDEO_FILE = 'output_video_dog.avi'
DEFAULT_OUTPUT_VIDEO_FPS = 24
DEFAULT_OUTPUT_VIDEO_RES = '720p'
DEFAULT_OUTPUT_VIDEO_TIME = 30 # seconds

DEFAULT_RECORD_MODE = False
DEFAULT_MONITOR = False
DEFAULT_DEBUG_MODE = True

global RECORD_MODE
RECORD_MODE = DEFAULT_RECORD_MODE
global MONITOR
MONITOR = DEFAULT_MONITOR
global DEBUG_MODE
DEBUG_MODE = DEFAULT_DEBUG_MODE



"""_____________output video_____________"""

STD_DIMENSIONS = {
    '480p' : (640, 480),
    '720p' : (1280, 720),
    '1080p' : (1920, 1080),
    '4k' : (3840, 2160)
}

VIDEO_TYPE = {
    'avi' : cv2.VideoWriter_fourcc(*'XVID'),
    'mp4' : cv2.VideoWriter_fourcc(*'XVID')
}


def change_res(cap, width, height):
    """ changes resolution of capture """
    cap.set(3, width)
    cap.set(4, height)

def get_dims(cap, res='480p'):
    """ get dimensions of output video """
    width, height = STD_DIMENSIONS['480p']
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width, height)
    return width, height

def get_video_type(filename):
    """ return the type of written video (suffix) """
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']




"""_____________pre-load_____________"""

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def load_labels(path):
    """ Load the labels mapping object id to object name """
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}
      

def get_arguments():
    """ Returns parsed aruments. 
        Path to the tflite model.
        Path to the object labels.
        Thresholds.
        Camera or input video.
        Save to output video.
        Record mode.
        Display frames on external monitor.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help=".tflite model path",
                        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_MODEL))
    parser.add_argument('--labels', help="label file path",
                        default=os.path.join(DEFAULT_MODEL_DIR, DEFAULT_LABELS))
    parser.add_argument('--top_k', type=int, default=DEFAULT_TOP_K,
                        help="number of categories with highest score to display")
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help="classifier score threshold")
    parser.add_argument('--capture_mode', type=str, help="'video' or 'camera'",
                        default=DEFAULT_CAPTURE_MODE)
    parser.add_argument('--input_video_dir', type=str, help="input video directory",
                        default=DEFAULT_INPUT_VIDEO_DIR)
    parser.add_argument('--input_video', type=str, help="input video file",
                        default=DEFAULT_INPUT_VIDEO_FILE)
    parser.add_argument('--output_video_dir', type=str, help="output video directory",
                        default=DEFAULT_OUTPUT_VIDEO_DIR)
    parser.add_argument('--output_video', type=str, help="output video file",
                        default=DEFAULT_OUTPUT_VIDEO_FILE)
    parser.add_argument('--output_video_res', type=str, help="480p, 720p, 1080p, 4k",
                        default=DEFAULT_OUTPUT_VIDEO_RES)
    parser.add_argument('--output_video_fps', type=int, help="output video file fps",
                        default=DEFAULT_OUTPUT_VIDEO_FPS)
    parser.add_argument('--output_video_time', type=int, help="output video file time",
                        default=DEFAULT_OUTPUT_VIDEO_TIME)
    parser.add_argument('--record_mode', type=bool, help="records output file (True or False)",
                        default=DEFAULT_RECORD_MODE)
    parser.add_argument('--monitor', type=bool, help="monitor connection (True or False)",
                        default=DEFAULT_MONITOR)
    parser.add_argument('--debug_mode', type=bool, help="debug logs (True or False)",
                        default=DEFAULT_DEBUG_MODE)
    return parser.parse_args()


def debug_print(message):
    """ Print debug logs if DEBUG_MODE is true. """
    if DEBUG_MODE:
        print(message)



"""_____________video capture_____________"""

def load_cap(capture_mode, input_video):
    """ Load the capture based of video file given or camera. """
    if capture_mode == 'video':
        return cv2.VideoCapture(input_video)
    elif capture_mode == 'camera':
        return cv2.VideoCapture(0)
    else:
        return cv2.VideoCapture(0)


def load_out(cap, capture_mode, output_video, output_video_res, cap_fps, video_fps):
    """ Get output video writer. """
    dims = get_dims(cap, res=output_video_res)
    if RECORD_MODE:
        if capture_mode == 'camera':
            return cv2.VideoWriter(output_video, get_video_type(output_video), int(cap_fps), dims)
        elif capture_mode == 'video':
            if cap.isOpened():
                ret, frame = cap.read()
                fshape = frame.shape
                fheight = fshape[0]
                fwidth = fshape[1]
                debug_print("({},{})".format(fwidth, fheight))
                return cv2.VideoWriter(output_video, get_video_type(output_video), video_fps, (fwidth, fheight))
        else:
            return cv2.VideoWriter(output_video, get_video_type(output_video), int(cap_fps), dims)
    else:
        return None


def cleanup(cap, out):
    """ Cleans the capture and videowriter. """
    if cap != None:
        cap.release()
        cv2.destroyAllWindows()
    if out != None:
        out.release()
        

def display_frame(frame):
    """ Display the frame on a monitor if one is available. """
    if MONITOR:
        cv2.imshow('frame', cv2_im)



"""_____________configure detection frames_____________"""

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """ Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()



def append_objs_to_img(cv2_im, objs, labels):
    """ Given a frame/image, append bounding boxes to the image
        correlated to each object's position and return it. """
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """ Returns list of detected objects. """
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]


def detect(frame, interpreter, labels, threshold, k):
    """ Detects objects in each frame using an interpreter engine.
        Returns the editted frame with bounding boxes added and a list of objects detected. """
    cv2_im = frame
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)

    common.set_input(interpreter, pil_im)
    interpreter.invoke()
    objs = get_output(interpreter, score_threshold=threshold, top_k=k)
    cv2_im = append_objs_to_img(cv2_im, objs, labels)

    return cv2_im, objs



"""_____________hardware_compatibility_____________"""

def detectPlatform():
    try:
        model_info = open(ENVIRONMENT).read()
        if 'Raspberry Pi' in model_info:
            print("Detected Raspberry Pi.")
            return 'raspberry'
        if 'MX8MQ' in model_info:
            print("Detected EdgeTPU dev board.")
            return 'devboard'
        return 'unknown'
    except:
        print("Could not detect enviornment. Assuming generic Linux.")
        return 'unknown'


class UI(object):
    """Abstract UI class. Subclassed by specific board implementations."""
    def __init__(self):
        """ Set all pins to low """
        self._input_state = [False for _ in self._INPUTS]
        current_time = time.time()
        self._input_state_last_change = [current_time for _ in self._INPUTS]
        self._debounce_interval = 0.1 # seconds


    def setOnlyOutput(self, index):
        """ Set all pins to low except the index pin """
        for i in range(len(self._OUTPUTS)): self.setOutput(i, False)
        if index is not None: self.setOutput(index, True)


    def isInputReceived(self, index):
        """ Get if a pin is high """
        inputs = self.getInputState()
        return inputs[index]


    def setOutput(self, index, state):
        raise NotImplementedError()


    def getInputState(self):
        raise NotImplementedError()


    def getDebouncedInputState(self):
        """ Check if the state of the pins has changed """
        t = time.time()
        for i,new in enumerate(self.getInputState()):
            if not new:
                self._input_state[i] = False
                continue
            old = self._input_state[i]
            if ((t-self._input_state_last_change[i]) >
                    self._debounce_interval) and not old:
                self._input_state[i] = True
            else:
                self._input_state[i] = False
            self._input_state_last_change[i] = t
        return self._input_state


    def testInputs(self):
        """ Debug by printing the state of all pins """
        while True:
            for i in range(5):
                self.setOutput(i, self.isInputReceived(i))
            print('Inputs: ', ' '.join([str(i) for i,v in
                enumerate(self.getInputState()) if v]))
            time.sleep(0.01)


    def wiggleLEDs(self, reps=3):
        """ Wiggle test to see if pins are working """
        for i in range(reps):
            self.setOutput(0, True)
            time.sleep(0.05)
            self.setOutput(0, False)


    def sendBluetoothMessage(self):
        raise NotImplementedError()



class UI_EdgeTpuDevBoard(UI):
    def __init__(self):
        """ GPIO info: https://coral.ai/docs/dev-board/gpio/.
            Establish input and output pins of the DevBoard. Establish bluetooth connection with
            GM-10 Bluetooth Module to communicate with an Arduino 
        """
        global GPIO, PWM
        from periphery import GPIO, PWM, GPIOError

        def initPWM(pin):
            """ Sets a pin to PWM """
            pwm = PWM(pin, 0)
            pwm.frequency = 1e3
            pwm.duty_cycle = 0
            pwm.enable()
            return pwm

        def initBluetooth(bd_addr, port):
            """ Establish bluetooth socket connection with DSD TECH HM-10 Bluetooth Module """
            bluetooth_socket = None
            try:
                bluetooth_socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                bluetooth_socket.connect((bd_addr,port))
            except bluetooth.btcommon.BluetoothError as message:
                print(message)
                if bluetooth_socket:
                    bluetooth_socket.send('</CLEAN/>')
                    bluetooth_socket.close()
            finally:
                return bluetooth_socket

        try:
            self._OUTPUTS = [GPIO(140, 'out'),
                          ]
            self._INPUTS = [GPIO(141, 'in'),
                            ]
            self._bd_addr = BLUETOOTH_MODULE_ADDRESS
            self._bd_port = BLUETOOTH_MODULE_PORT
            self._bluetooth_socket = None
            self._bluetooth_socket = initBluetooth(self._bd_addr, self._bd_port)
        except GPIOError as e:
            print("Unable to access GPIO pins. Try running with sudo")
            sys.exit(1)

        super(UI_EdgeTpuDevBoard, self).__init__()


    def __del__(self):
        """ Write 0 to all output pins. Send a bluetooth message to ardunio to turn off and clean its pins too """
        if hasattr(self, '_OUTPUTS'):
            for i in range(len(self._OUTPUTS)):
                self.setOutput(i, False)
            for x in self._OUTPUTS or [] + self._INPUTS or []: x.close()
            print("cleaned up OUTPUTS")
        if self._bluetooth_socket:
            self._bluetooth_socket.send('</OFF/CLEAN/>')
            self._bluetooth_socket.close()
            print("cleaned up Bluetooth")


    def setOutput(self, index, state):
        """ Abstracts away mix of GPIO and PWM OUTPUTS. """
        if type(self._OUTPUTS[index]) is periphery.gpio.SysfsGPIO:
            self._OUTPUTS[index].write(state)
        else:
            self._OUTPUTS[index].duty_cycle = 0.0 if not state else 1.0


    def getInputState(self):
        """ Get inputs of all devboard pins """
        return [_input.read() for _input in self._INPUTS]

         
    def sendBluetoothMessage(self, message):
        """ Send bluetooth message """
        try:
            self._bluetooth_socket.send(message)
        except Exception as e:
            print("Bluetooth error:", e)
            raise Exception('Turned off')



def get_ui():
    """ Get hardware platform """
    ui = None
    platform = detectPlatform()
    if platform == 'raspberry':
        print("Raspberry-pi detected. Not implemented")
    elif platform == 'devboard':
        print("Google Coral Devboard detected. Successfully implemented")
        ui = UI_EdgeTpuDevBoard()
    else:
        print("No GPIOs detected")
    return ui


def create_bluetooth_message(list_code):
    """ Return bluetooth message is correct format to be parsed by arduino """
    message = "<>"
    if len(list_code) > 0:
        message = "</" + '/'.join(list_code) + "/>"
    return message



def relay_bluetooth_detection(ui, objs, labels, width, height):
    """ Get objects. Using dimensions of the bounding boxes of the objects, calculate
        the weighted middle point of all objects detected. From said coordinates,
        relay the proper action via bluetooth message. """
    center_x = 0
    center_y = 0
    x_pos = 0
    y_pos = 0
    w = 0
    h = 0
    detected = ""
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x, y, w, h = x0, y0, x1 - x0, y1 - y0
        x, y, w, h = int(x*width), int(y*height), int(w*width), int(h*height)
        
        center_x, center_y = (x1 - x0)/2, (y0 - y1)/2

        x_pos = x
        y_pos = y

        detected = labels.get(obj.id, obj.id)
        break # One object detected suffices for this project


    list_code = []
    if (len(objs)) > 0:
        list_code.append("ON")
        if (x_pos > (width/2) - (w/2) + (width/4)):
            list_code.append("C")
            list_code.append("8") #16
        elif (x_pos > (width/2) - (w/2) + (width/8)):
            list_code.append("C")
            list_code.append("4") #4
        elif (x_pos > (width/2) - (w/2) + (width/16)):
            list_code.append("C")
            list_code.append("2") #2
        elif (x_pos < (width/2) - (w/2) - (width/4)):
            list_code.append("CC")
            list_code.append("8") #16
        elif (x_pos < (width/2) - (w/2) - (width/8)):
            list_code.append("CC")
            list_code.append("4") #4
        elif (x_pos < (width/2) - (w/2) - (width/16)):
            list_code.append("CC")
            list_code.append("2") #2
        else:
            list_code.append("0")
            
        if (detected == "dog"):
            list_code.append("C1")
            ui.setOutput(0, True);

    else:
        print("no object detected")
        ui.setOutput(0, False)
        list_code.append("OFF")

    bluetooth_message = create_bluetooth_message(list_code)
    print(bluetooth_message)
    ui.sendBluetoothMessage(bluetooth_message)
    
        


"""_____________main_____________"""

def main():
    args = get_arguments()
    MODEL = args.model
    LABELS = args.labels
    TOP_K = args.top_k
    THRESHOLD = args.threshold
    CAPTURE_MODE = args.capture_mode
    INPUT_VIDEO_DIR = args.input_video_dir
    INPUT_VIDEO = str(os.path.join(INPUT_VIDEO_DIR, args.input_video))
    OUTPUT_VIDEO_DIR = args.output_video_dir
    OUTPUT_VIDEO = str(os.path.join(OUTPUT_VIDEO_DIR, args.output_video))
    OUTPUT_VIDEO_RES = args.output_video_res
    OUTPUT_VIDEO_FPS = args.output_video_fps
    OUTPUT_VIDEO_TIME = args.output_video_time
    
    RECORD_MODE = args.record_mode
    MONITOR = args.monitor
    DEBUG_MODE = args.debug_mode
    
    # Define the hardware
    ui = get_ui()
    
    # Create an interpreter engine based on the tflite model path given
    debug_print("Loading {} with {} labels.".format(MODEL, LABELS))
    interpreter = common.make_interpreter(MODEL)
    interpreter.allocate_tensors()

    cap = None
    out = None

    # Define camera or input video properties
    cap = load_cap(CAPTURE_MODE, INPUT_VIDEO)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    cap_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Define output file if recording
    out = load_out(cap, CAPTURE_MODE, OUTPUT_VIDEO, OUTPUT_VIDEO_RES, cap_fps, OUTPUT_VIDEO_FPS)

    labels = load_labels(LABELS)
    frame_count = 0

    # Read frame of the capture and feed it to the interpreter. Feed out an editted frame with detected objects encapuslated by bounding box.
    # Based on the object(s) captured, replay bluetooth message to the ardunio that controls the stepper motor and animatronics.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        objs = []
        cv2_im, objs = detect(frame, interpreter, labels, THRESHOLD, TOP_K)
        
        display_frame(cv2_im)

        relay_bluetooth_detection(ui, objs, labels, cap_width, cap_height)
        
        if RECORD_MODE:
            out.write(cv2_im)
        
        if DEBUG_MODE:
            frame_count += 1
            seconds = (frame_count/cap_fps)%60
            debug_print(seconds)
                                   
        if (cv2.waitKey(1) & 0xFF == ord('q')) or ((RECORD_MODE or DEBUG_MODE) and seconds > OUTPUT_VIDEO_TIME):
            debug_print("Ending capture")
            break
    
    cleanup(cap, out)

    print("Successful termination")


if __name__ == '__main__':
    main()

