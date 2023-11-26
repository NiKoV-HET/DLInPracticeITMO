import cv2
import torch

import numpy as np

from sort import Sort
import time
from TrafficSignRecognition.onnx_inferer import ONNXTrafficSignClassifier
import gradio as gr
import tempfile
import shutil
import os
import tqdm


class CustomTracker:
    def __init__(self, max_age=10, min_hits=1, iou_threshold=0.2):
        self.sort_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)

        self.max_age = 10

        self.trackers_info = {}  # Dictionary to store age and hit streak for each tracker

    def update(self, detections):
        tracked_objects = self.sort_tracker.update(detections)

        for track_id in self.trackers_info.keys():
            self.trackers_info[track_id]["time_alive"] += 1

        for *bbox, tr_id in tracked_objects:
            if int(tr_id) in self.trackers_info:
                self.trackers_info[int(tr_id)]["time_alive"] = 0
                self.trackers_info[int(tr_id)]["bbox"] = list(map(int, bbox))
            else:
                self.trackers_info[int(tr_id)] = {"time_alive": 0, "bbox": list(map(int, bbox))}

        updated_trackers_info = self.trackers_info.copy()

        for track_id in self.trackers_info.keys():
            if self.trackers_info[track_id]["time_alive"] > self.max_age:
                del updated_trackers_info[track_id]

        self.trackers_info = updated_trackers_info

        return tracked_objects

    def get_alive_trackers(self):
        return self.trackers_info


# Load model
yolo_model = torch.hub.load("ultralytics/yolov5", "custom", "models/yolov5n.onnx", device="cpu")
# classifier_model = ONNXTrafficSignClassifier(
#     r"TrafficSignRecognition\models\resnet18\traffic_sign_recognition20.onnx",
#     r"TrafficSignRecognition\models\resnet18\traffic_label_enum.json",
# )
classifier_model = ONNXTrafficSignClassifier(
    r"TrafficSignRecognition\models\resnet34\traffic_sign_recognition21.onnx",
    r"TrafficSignRecognition\models\resnet34\traffic_label_enum.json",
)


def preprocessing_image_yolo(img, size=(640, 640)):
    """Preprocessing image function for yolo. BGR->RGB, resize

    Args:
        img (np.ndarray): Image
        size (tuple, optional): Final size. Defaults to (640, 640).

    Returns:
        np.ndarray: Processed image
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Calculate the aspect ratio of the desired size and the original image
    desired_ratio = size[0] / size[1]
    original_ratio = img.shape[1] / img.shape[0]

    # Calculate padding
    if original_ratio > desired_ratio:  # Padding top and bottom
        new_width = size[0]
        new_height = int(img.shape[0] * new_width / img.shape[1])
        top_pad = (size[1] - new_height) // 2
        bottom_pad = size[1] - new_height - top_pad
        left_pad = right_pad = 0
    else:  # Padding left and right
        new_height = size[1]
        new_width = int(img.shape[1] * new_height / img.shape[0])
        left_pad = (size[0] - new_width) // 2
        right_pad = size[0] - new_width - left_pad
        top_pad = bottom_pad = 0

    # Apply padding
    padded_img = cv2.copyMakeBorder(
        img_rgb, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    return cv2.resize(img_rgb, size)


def extract_detections(results, width, height, threshold):
    """Extract detections from YOLOv5 results.

    Args:
        results: Detection result
        width (int): Full image width
        height (int): Full image height
        threshold (float): Detection threshold

    Returns:
        np.ndarray: Filtered results
    """

    detections = np.empty((0, 5))
    margin = 5

    for result in results.xyxyn[0]:
        x1, y1, x2, y2, conf, _ = np.array(result.cpu())

        if conf > threshold:
            detections = np.vstack(
                (
                    detections,
                    np.array(
                        [x1 * width - margin, y1 * height - margin, x2 * width + margin, y2 * height + margin, conf]
                    ),
                )
            )

    return detections


def draw_tracked_objects(img, tracked_objects, threshold):
    """Draw bounding boxes and ID labels on the image.
    TODO: NOT USING NOW

    Args:
        img (np.ndarray): Full image
        tracked_objects (list): All tracked signs
        threshold (float): Class threshold

    Returns:
        np.ndarray: Image with bbox and classes
    """

    for track_id, sign_data in tracked_objects.items():
        x1, y1, x2, y2 = sign_data["bbox"]
        time_alive = sign_data["time_alive"]
        sign_cls = sign_data["cls"]
        cls_conf = sign_data["cls_conf"]
        if time_alive == 0 and cls_conf > threshold and sign_cls != "background":
            img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 5)

            img = cv2.putText(
                img,
                f"CLS:{sign_cls}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return img


def get_class(image, bbox):
    """Function for classifying sign

    Args:
        image (np.ndarray): Full image
        bbox (list): Sign Bounding Box

    Returns:
        dict: Label sign with conf
    """
    x1, y1, x2, y2 = bbox
    if x2 > x1 and y2 > y1 and x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
        image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        bbox_xyxy = [x1, y1, x2, y2]
        # cv2.imshow("sign", image[y1:y2, x1:x2])
        output = classifier_model(image, bbox_xyxy)
        return {"cls": output[0], "conf": output[1]}
    return {"cls": "background", "conf": 0}


def update_signs(img, alive_signs, all_signs, threshold_cls):
    """Function for update actual signs

    Args:
        img (np.ndarray): Full image
        alive_signs (dict): Sing, which alive in tracker
        all_signs (dict): All actual signs
        threshold_cls (float): Classification threshold

    Returns:
        dict: Actual signs
    """
    die_id = []
    for sign_id in all_signs.keys():
        if sign_id not in alive_signs.keys():
            die_id.append(sign_id)

    for sign_id in die_id:
        del all_signs[sign_id]

    for sign_id, sign_data in alive_signs.items():
        if sign_data["time_alive"] == 0:
            sign_cls = get_class(img, sign_data["bbox"])
            if sign_cls["conf"] > threshold_cls:
                if sign_id not in all_signs:
                    all_signs[sign_id] = {"cls": sign_cls["cls"], "cls_conf": sign_cls["conf"]}

                elif sign_cls["conf"] > all_signs[sign_id]["cls_conf"]:
                    all_signs[sign_id] = {"cls": sign_cls["cls"], "cls_conf": sign_cls["conf"]}
        if sign_id in all_signs:
            all_signs[sign_id]["bbox"] = sign_data["bbox"]
            all_signs[sign_id]["time_alive"] = sign_data["time_alive"]

    return all_signs


def draw_signs_bar(bar_w, bar_h, signs, sign_images):
    """Function for drawing stripes with signs. The height of all signs are the same.
    If there are more signs than the length of the strip, the extra ones are not drawn.

    Args:
        bar_w (int): Bar width
        bar_h (int): Bar height
        signs (dict): Signs for drawing
        sign_images (dict): All sign images

    Returns:
        np.ndarray: Bar image
    """
    bar = np.full((bar_h, bar_w, 3), 255, dtype=np.uint8)
    last_w = 0
    for sign_id, sign_data in signs.items():
        if sign_data["cls"] != "background" and last_w < bar_w:
            sign_img = sign_images[sign_data["cls"]]
            sign_h_new, sign_w_new, _ = sign_img.shape
            bar[0:bar_h, last_w : last_w + sign_w_new] = sign_img
            last_w = last_w + sign_w_new + 1
    return bar
    # cv2.imshow("bar", bar)


def preprocessing_preview_sign_image(image_path, signs_h):
    """Function for preprocessing sign images. Remove alpha chanel and resize

    Args:
        image_path (str): Path to image
        signs_h (int): Sign height to output

    Returns:
        np.ndarray: Sign image
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 4:
        rgb_img = img[:, :, :3]
        alpha_channel = img[:, :, 3]
        white_background = np.ones_like(rgb_img, dtype=np.uint8) * 255
        alpha_3channel = cv2.cvtColor(alpha_channel, cv2.COLOR_GRAY2BGR)
        foreground = cv2.bitwise_and(rgb_img, alpha_3channel)
        background = cv2.bitwise_and(white_background, 255 - alpha_3channel)
        img_with_white_bg = cv2.add(foreground, background)
        img = img_with_white_bg

    h, w, _ = img.shape
    k = w / h
    img = cv2.resize(img, (int(signs_h * k), signs_h))

    return img


def load_preview_signs(signs_h):
    """Function for loading all sign preview

    Args:
        signs_h (int): Signs heights

    Returns:
        dict: Signs preview
    """
    sign_images = {}
    path_to_preview_signs = "data\Signs"
    for sign_image_filename in os.listdir(path_to_preview_signs):
        cls_ = sign_image_filename.split(".")[0]
        sign_images[cls_] = preprocessing_preview_sign_image(
            os.path.join(path_to_preview_signs, sign_image_filename), signs_h
        )
    return sign_images


def process_video(input_video):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        shutil.copyfile(input_video, temp_video_file.name)

    cap = cv2.VideoCapture(temp_video_file.name)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_filename = tempfile.mktemp(suffix=".mp4")

    signs = {}
    signs_bar_width = width
    signs_bar_height = 100
    signs_preview = load_preview_signs(signs_bar_height)

    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height + signs_bar_height))

    tracker = CustomTracker(max_age=5, min_hits=1, iou_threshold=0.2)

    # We will process the video at 4 FPS, regardless of how many FPS the sent video has
    processing_fps = 5
    skip_frame = fps // processing_fps

    threshold_bbox = 0.4
    threshold_cls = 0.6

    cur_frame_num = 0
    start_time = time.time()
    count_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    pbar = tqdm.tqdm(total=int(count_frames))
    while ret:
        ret, img = cap.read()
        if ret:
            if cur_frame_num % skip_frame == 0:
                results = yolo_model(preprocessing_image_yolo(img))
                detections = extract_detections(results, width, height, threshold_bbox)
                tracked_objects = tracker.update(np.array(detections))
                signs = update_signs(img, tracker.get_alive_trackers(), signs, threshold_cls)
                # vis_image = draw_tracked_objects(img, signs, threshold_bbox)
                # cv2.imshow("Image", vis_image)
                # cv2.waitKey(1)

            bar_img = draw_signs_bar(signs_bar_width, signs_bar_height, signs, signs_preview)
            stack_img = np.vstack([bar_img, img])
            out.write(stack_img)

            cur_frame_num += 1
            pbar.update(1)

    out.release()
    cap.release()
    pbar.close()
    os.unlink(temp_video_file.name)
    return output_filename


def launch_gradio_interface():
    video_input = gr.components.Video(label="Input video")
    video_output = gr.components.Video(label="Output video")
    iface = gr.Interface(
        fn=main,
        inputs=video_input,
        outputs=video_output,
        title="SignClassifier",
        examples=[["examples\example_video.mp4"]],
        cache_examples=True,
    )
    iface.launch()


def main(input_video):
    output_filename = process_video(input_video)
    return output_filename


if __name__ == "__main__":
    launch_gradio_interface()
