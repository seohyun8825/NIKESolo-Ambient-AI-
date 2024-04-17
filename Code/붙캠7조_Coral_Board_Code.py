import argparse

import cv2
import numpy as np
import os
import time

import subprocess  ##사운드 관련 모듈

# pycoral은 coral board 내부에 위치
from pycoral.adapters import common
from pycoral.adapters import segment
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from collections import OrderedDict
from pycoral.utils.dataset import read_label_file

# 필요한 변수 정의
LABEL_TO_COLOR = OrderedDict({
    "background": [255, 255, 255],
    "high_vegetation": [40, 80, 0],
    "traversable_grass": [128, 255, 0],
    "smooth_trail": [178, 176, 153],
    "obstacle": [255, 0, 0],
    "sky": [1, 88, 255],
    "rough_trail": [156, 76, 30],
    "puddle": [255, 0, 128],
    "non_traversable_low_vegetation": [0, 160, 0]
})

PALETTE = list(LABEL_TO_COLOR.values())
palette = np.array(PALETTE, dtype=np.uint8)

previous_frame_count = 0
fps = 0

det_width = 0
det_height = 0

detection_label = "../labels/coco_labels.txt"
labels = read_label_file(detection_label)

""" Utility Functions """


# segmentation 모델의 마스크 output 이미지를 RGB 이미지로 변환
def decode_segmentation_masks(mask, colormap):
    """
    Converts mask (224, 224) into RGB image (224, 224, 3)

    Args :
        mask = numpy array of shape (224, 224)
        colormap = numpy array containing RGB for each 9 classes of mask shape (9, 3)

    Return :
        numpy array of shape (224, 224, 3)
    """
    mask = mask.astype(np.uint8)
    rgb = colormap[mask]

    return rgb

# left, center, right 영역을 나누어, 'smooth trail' pixel 계산하여 비율로 direction 판단
def calculate_direction(image, line_height_ratio, line_offset_ratio, offset, traversable_labels):
    """
    Calculates direction using three lines for norm

    Args :
        image = numpy array of shape (224, 224)
        line_height_ratio = int value for ratio of center line to image height, bigger the longer
        line_offset_ratio = float value for offset ratio between center line and right, left line, bigger the longer
        traversable_labels = list [3] includes index of 'smooth_trail'

    Return :
        direction = str value for indicating direction among 'Turn Left' , 'Turn Right' , 'Stop' , 'Hard to tell'
    """
    height, width = image.shape
    line_height = int(height * (1 - line_height_ratio))

    center_line_x = width // 2
    left_line_x = center_line_x - int(width * line_offset_ratio)
    right_line_x = center_line_x + int(width * line_offset_ratio)

    # Calculate overlaps for center, left, and right lines
    center_overlap = np.isin(image[line_height:, center_line_x - offset: center_line_x + offset + 1],
                             traversable_labels).mean()
    left_overlap = np.isin(image[line_height:, left_line_x - offset: left_line_x + offset + 1],
                           traversable_labels).mean()
    right_overlap = np.isin(image[line_height:, right_line_x - offset: right_line_x + offset + 1],
                            traversable_labels).mean()
    print("center_overlap : {} , left_overlap : {} , right_overlap : {}".format(center_overlap, left_overlap, right_overlap)) 

    if 0.1 < center_overlap <= 0.6:
        # Determine direction
        if left_overlap > right_overlap:
            direction = "Turn left"
        elif right_overlap > left_overlap:
            direction = "Turn right"
        else:
            # If there is no clear direction, do not trigger any action
            return None, None

        # Determine steps_ahead based on center_overlap
        if center_overlap > 0.4:
            steps_ahead = "six"
        elif center_overlap > 0.2:
            steps_ahead = "three"
        else:  # covers the case where 0.1 < center_overlap <= 0.2
            steps_ahead = "one"

        return direction, steps_ahead

    # For overlaps outside the specified range, do not trigger any action
    return None, None


# ######################## 사람 탐지 함수 ##########################
def detection_person(cv2_im, inference_size, objs, norm):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.id != 0:
            continue
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        # Accuracy
        percent = int(100 * obj.score)

        # Check Area of Bounding Box
        area = (x1 - x0) * (y1 - y0)
        if area < 0:
            area *= -1
        area = area / width / height

        print("Area : {}".format(area))
        # norm is number between 0 ~ 1
        # indicates the ratio between detected person image area and whole image area
        if percent > 50 and area > norm:
            return "Person Near"
        else:
            return "Safe"


################## Sound Alert 함수 구현 #######################################
def sound_alert_det(person, frame_count):
    global previous_frame_count
    global fps
    print("previous frame count : {} / frame : {} / fps : {}".format(previous_frame_count, frame_count, fps))
    if previous_frame_count != 0 and frame_count < previous_frame_count + 3 * fps:
        return

    previous_frame_count = frame_count
    audio_file_path = 'PersonAlarm.wav'
    # 'Person Near' 상태일 때 사운드 재생
    if person == "Person Near":
        try:
            # Linux 또는 Unix-like 시스템에서 'aplay' 명령어를 사용하여 사운드 파일 재생
            subprocess.Popen(['aplay', audio_file_path])
        except Exception as e:
            # 사운드 재생 중 오류 발생 시 메시지 출력
            print(f"Error playing sound: {e}")


last_played_state = ('', '')

def play_sound(file_path):
    try:
        subprocess.Popen(['aplay', file_path])
    except subprocess.CalledProcessError as e:
        print(f"Error playing sound: {e}")

def sound_alert_seg(direction, steps_ahead):
    global last_played_state  # Reference the global variable
    direction_normalized = direction.replace(' ', '').lower()

    # Construct the file name based on the direction and steps ahead
    file_name = f"{steps_ahead}_{direction_normalized}.wav"

    # Current state
    current_state = (direction, steps_ahead)

    # Check if the current state is different from the last played state
    if current_state != last_played_state:
        # Play the sound
        play_sound(file_name)

        # Update the last played state
        last_played_state = current_state


# Detection 함수
def process_det(frame, file, frame_count, interpreter_det, intermediate):
    global det_width
    global det_height
    # ------------------------- Time Record 1 시작 : 전처리 단계
    start1 = time.time()

    file.write("Start of frame {} : \n".format(frame_count + 1))
    cv2_im = frame
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    resized_img_det = cv2.resize(cv2_im_rgb, (det_width, det_height))
    common.set_input(interpreter_det, resized_img_det)

    end1 = time.time()
    file.write("Image processing time for model input : {}\n".format(end1 - start1))
    # Time Record 1 끝 -------------------------

    ############################################################################

    # ------------------------- Time Record 2 시작 : 추론 단계
    start2 = time.time()

    interpreter_det.invoke()
    # score_threshold can be modified
    objs = detect.get_objects(interpreter_det, score_threshold=0.5)

    end2 = time.time()
    file.write("Model time : {}\n".format(end2 - start2))
    # Time Record 2 끝 -------------------------

    ############################################################################

    # ------------------------- Time Record 3 시작 : output 후처리 단계
    start3 = time.time()

    # 전체 화면의 비율의 0.3 이상이면 Alert -- 0.3 은 변경 가능
    person = detection_person(cv2_im, (det_width, det_height), objs, 0.3)
    # person == 1. "Person Near"     2. "Safe"

    sound_alert_det(person, frame_count)  # 사운드 알람 함수 호출

    end3 = time.time()
    file.write("Person : {}\n".format(person))
    file.write("Alert process time : {}\n\n".format(end3 - start3))
    # Time Record 3 끝 -------------------------

    ############################################################################

    return append_objs_to_img_only_person(intermediate, (det_width, det_height), objs)

# Segmentation 함수
def process_seg(frame, file, frame_count, interpreter, width, height):
    # ------------------------- Time Record 1 시작 : 전처리 단계
    start1 = time.time()

    traversable_labels = [list(LABEL_TO_COLOR.keys()).index(key) for key in
                          ['smooth_trail']]

    file.write("Start of frame {} : \n".format(frame_count + 1))
    cv2_im = frame
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(cv2_im_rgb, (width, height))
    common.set_input(interpreter, resized_img)

    end1 = time.time()
    file.write("Image processing time for model input : {}\n".format(end1 - start1))
    # Time Record 1 끝 -------------------------

    ############################################################################

    # ------------------------- Time Record 2 시작 : 추론 단계
    start2 = time.time()

    interpreter.invoke()
    result = segment.get_output(interpreter)  # (224, 224, 3)
    result_flatten = np.argmax(result, axis=-1)  # (224, 224)
    result_final = decode_segmentation_masks(result_flatten, palette)
    result_final_resized = cv2.resize(result_final, (det_width, det_height))

    end2 = time.time()
    file.write("Model inference time : {}\n".format(end2 - start2))
    # Time Record 2 끝 -------------------------

    ############################################################################

    # ------------------------- Time Record 3 시작 : output 후처리 단계
    start3 = time.time()

    direction, steps_ahead = calculate_direction(result_flatten, 0.5, 0.2, 15, traversable_labels)

    if direction is not None:   
        sound_alert_seg(direction, steps_ahead)

    end3 = time.time()
    file.write("Direction : {} , Steps Ahead : {}\n".format(direction, steps_ahead))
    file.write("Alert process time : {}\n\n".format(end3 - start3))
    # Time Record 3 끝 -------------------------

    return result_final_resized


def append_objs_to_img_only_person(cv2_im, inference_size, objs):
    global labels

    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        if obj.id != 0:
            continue
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


""" Main Function """


def main():
    # global variable
    global fps
    global det_height
    global det_width

    # 모델 불러오기
    default_model_dir = "../all_models"
    default_model = "mobilenetv2_deeplabv3_edgetpu.tflite"

    detection_model = "../all_models/ssd_mobilenetv2_edgetpu.tflite"

    # 코멘드 읽기
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=0)
    parser.add_argument('--input', default=None)
    parser.add_argument('--output', default='./out.mp4')
    parser.add_argument('--length', type=int, default=7)
    args = parser.parse_args()

    # Bring tflite model
    # 1. Segmentation Model
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    # 2. Detection Model
    interpreter_det = make_interpreter(detection_model)
    interpreter_det.allocate_tensors()

    # Get input size of tflite model (width, height)
    width, height = common.input_size(interpreter)  # Segmentation Model
    det_width, det_height = common.input_size(interpreter_det)  # Detection Model

    # Read video or camera
    if args.input:
        # if there is input video, read it
        cap = cv2.VideoCapture(args.input)
        from_camera = False
    else:
        # if no video, record it
        cap = cv2.VideoCapture(args.camera_idx)
        # limit buffer to size of 2 for real-time
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        from_camera = True

    # Get frame width and height
    if cap.isOpened():
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        # fps
        fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        # cap이 닫혀있다면 종료
        print("camera is not ready")
        return

    # Prepare recording
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (det_width, det_height))

    # Frames
    frames = fps * args.length 
    frame_count = 0  

    # Logs
    file = open("./record.txt", 'a')

    # fps control
    frame_counter_seg = 0
    frame_counter_det = 0

    # Desired FPS allocation
    fps_seg = 2  # segmentation model get 2 fps --- approximately 0.43 sec latency
    fps_det = 1  # detection model get 1 fps --- approximately 0.017 sec latency

    intermediate = None


    # ------------------------- Time Record 0 시작 : 전체 과정이 몇 초 인지 측정을 위한 변수, 나중에 평균 fps 계산에 사용
    start0 = time.time()
    # Video / Camera Recording 시작
    while cap.isOpened() and (frames > 0 or from_camera):
        ret, frame = cap.read()
        if not ret:
            break

        frames -= 1
        frame_count += 1

        frame_counter_seg += fps_seg
        frame_counter_det += fps_det

        if frame_counter_seg >= fps:
            intermediate = process_seg(frame, file, frame_count, interpreter, width, height)
            frame_counter_seg = 0

        if frame_counter_det >= fps:
            intermediate = process_det(frame, file, frame_count, interpreter_det, intermediate)
            frame_counter_det = 0

        if intermediate is not None:
            out.write(intermediate)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end0 = time.time()
    # Time Record 0 끝 -------------------------


    file.write("Total frames taken : {}\n".format(frame_count))
    file.write("Average fps : {}\n".format(frame_count / (end0 - start0)))
    file.close()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
