import multiprocessing
import numpy
import os
import sys
import tarfile
import time
import typing
import urllib

import click
import cv2
import tensorflow

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils

from . import detect

MODEL_NAME = "ssd_mobilenet_v1_coco_2017_11_17"
MODEL_FILE = MODEL_NAME + ".tar.gz"
DOWNLOAD_BASE = "http://download.tensorflow.org/models/object_detection/"

# Path to frozen detection graph. This is the actual model that is used for the
# object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + "/frozen_inference_graph.pb"

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    os.environ["LABELS_DATA"], "mscoco_label_map.pbtxt",
)

NUM_CLASSES = 90


@detect.command("stuff")
@click.option(
    "--scale-video", type=float, default=1, help="Video scaling factor",
)
@click.option(
    "--workers", type=int, default=1, help="Number of worker processes",
)
@click.option("--out-video", type=click.Path(exists=False, writable=True))
@click.argument(
    "video_file_path",
    type=click.Path(
        exists=True, file_okay=True, readable=True, resolve_path=True,
    ),
)
def stuff(
        scale_video: float,
        workers: int,
        out_video: typing.Optional[click.Path],
        video_file_path: click.Path,
) -> None:
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(multiprocessing.SUBDEBUG)

    video = cv2.VideoCapture(video_file_path)
    _, frame = video.read()
    frame = _resize_frame(frame, scale_video)
    video.release()

    input_queue = multiprocessing.Queue(maxsize=5)
    output_queue = multiprocessing.Queue(maxsize=5)
    pool = multiprocessing.Pool(
        workers, run_worker, (input_queue, output_queue, frame.shape),
    )

    video = cv2.VideoCapture(video_file_path)

    video_start_time = time.time()
    frames = 0
    out = None
    while video.isOpened():
        ret, frame = video.read()
        frames += 1
        if frame is None:
            break

        frame = _resize_frame(frame, scale_video)

        if out_video and out is None:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(
                out_video, fourcc, 20.0, (frame.shape[1], frame.shape[0]),
            )

        input_queue.put(frame)
        frame = output_queue.get()

        if out:
            out.write(frame)

        cv2.imshow("image", frame)
        sys.stdout.write(
            f"\r{frames / (time.time() - video_start_time):.6f} fps"
            f" {frames} / {time.time() - video_start_time} s"
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    if out:
        out.release()

    cv2.destroyAllWindows()


def _resize_frame(frame, scale_video):
    return cv2.resize(
        frame,
        None,
        fx=scale_video,
        fy=scale_video,
        interpolation=cv2.INTER_CUBIC,
    )


def run_worker(
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        image_shape,
) -> None:
    detection_graph = tensorflow.Graph()
    with detection_graph.as_default():
        od_graph_def = tensorflow.GraphDef()
        with tensorflow.gfile.GFile(PATH_TO_FROZEN_GRAPH, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tensorflow.import_graph_def(od_graph_def, name="")

        session = tensorflow.Session(graph=detection_graph)

    category_index = _load_category_index()
    with detection_graph.as_default():
        tensors = _get_tensors(image_shape)

    while True:
        frame = input_queue.get()
        detected = _detect_objects(
            frame, session, detection_graph, tensors,
        )
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            frame,
            detected["detection_boxes"],
            detected["detection_classes"],
            detected["detection_scores"],
            category_index,
            instance_masks=detected.get("detection_masks"),
            use_normalized_coordinates=True,
            line_thickness=8,
        )

        output_queue.put(frame)

    session.close()


def _load_category_index():
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True,
    )
    category_index = label_map_util.create_category_index(categories)
    return category_index


def _get_tensors(image_shape):
    ops = tensorflow.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensors = {}
    for key in [
        "num_detections", "detection_boxes", "detection_scores",
        "detection_classes", "detection_masks"
    ]:
        tensor_name = key + ":0"
        if tensor_name in all_tensor_names:
            tensors[key] = tensorflow.get_default_graph().get_tensor_by_name(
                tensor_name,
            )

    if "detection_masks" in tensors:
        # The following processing is only for single image
        detection_boxes = tensorflow.squeeze(
            tensors["detection_boxes"], [0],
        )
        detection_masks = tensorflow.squeeze(
            tensors["detection_masks"], [0],
        )

        # Reframe is required to translate mask from box coordinates to
        # image coordinates and fit the image size.
        real_num_detection = tensorflow.cast(
            tensors["num_detections"][0], tensorflow.int32,
        )
        detection_boxes = tensorflow.slice(
            detection_boxes, [0, 0], [real_num_detection, -1],
        )
        detection_masks = tensorflow.slice(
            detection_masks, [0, 0, 0], [real_num_detection, -1, -1],
        )
        detection_masks_reframed = (
            utils_ops.reframe_box_masks_to_image_masks(
                detection_masks,
                detection_boxes,
                image_shape[0],
                image_shape[1]
            )
        )
        detection_masks_reframed = tensorflow.cast(
            tensorflow.greater(detection_masks_reframed, 0.5),
            tensorflow.uint8,
        )

        # Follow the convention by adding back the batch dimension
        tensors["detection_masks"] = tensorflow.expand_dims(
            detection_masks_reframed, 0,
        )

    return tensors


def _detect_objects(image, session, graph, tensors) -> dict:
    with graph.as_default():
        image_tensor = tensorflow.get_default_graph().get_tensor_by_name(
            "image_tensor:0",
        )

        output = session.run(
            tensors, feed_dict={image_tensor: numpy.expand_dims(image, 0)},
        )

        output["num_detections"] = int(output["num_detections"][0])
        output["detection_classes"] = output["detection_classes"][0].astype(
            numpy.uint8,
        )
        output["detection_boxes"] = output["detection_boxes"][0]
        output["detection_scores"] = output["detection_scores"][0]

        if "detection_masks" in output:
            output["detection_masks"] = output["detection_masks"][0]

    return output


@detect.command("download-models")
def download_models():
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if "frozen_inference_graph.pb" in file_name:
            tar_file.extract(file, os.getcwd())
