import numpy
import os
import sys
import tarfile
import time
import urllib

import click
import cv2
import tensorflow

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils

from . import detect

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the
# object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.environ["LABELS_DATA"], "mscoco_label_map.pbtxt")

NUM_CLASSES = 90


@detect.command("stuff")
@click.argument(
    "video_file_path",
    type=click.Path(
        exists=True, file_okay=True, readable=True, resolve_path=True,
    ),
)
def stuff(video_file_path: click.Path):
    start_time = time.monotonic()
    detection_graph = tensorflow.Graph()
    with detection_graph.as_default():
        od_graph_def = tensorflow.GraphDef()
        with tensorflow.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tensorflow.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True,
    )
    category_index = label_map_util.create_category_index(categories)
    print(f"Warmup time: {(time.monotonic() - start_time):.6f} s")

    video = cv2.VideoCapture(video_file_path)
    video_start_time = time.monotonic()
    frames = 0
    while video.isOpened():
        frame_start_time = time.monotonic()
        ret, frame = video.read()
        frames += 1
        frame_read_time = time.monotonic() - frame_start_time
        if frame is None:
            break

        resize_start_time = time.monotonic()
        image = cv2.resize(
            frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC,
        )
        resize_time = time.monotonic() - resize_start_time
        converting_start_time = time.monotonic()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        converting_time = time.monotonic() - converting_start_time

        detection_start_time = time.monotonic()
        # Expand dimensions since the model expects images to have shape:
        # [1, None, None, 3]
        image_np_expanded = numpy.expand_dims(image, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image, detection_graph)
        detection_time = time.monotonic() - detection_start_time

        visualization_start_time = time.monotonic()
        # Visualization of the results of a detection.
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8,
        )
        visualization_time = time.monotonic() - visualization_start_time

        cv2.imshow("image", image)
        sys.stdout.write(
            "\r"
            f"{frames / (time.monotonic() - video_start_time):.6f} fps"
            f", frame: {(time.monotonic() - frame_start_time):.6f} s"
            f" (read: {frame_read_time:.6f} s"
            f", resize: {resize_time:.6f} s"
            f", convert: {converting_time:.6f} s"
            f", detection: {detection_time:.6f} s"
            f", visualisation: {visualization_time:.6f} s)",
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("")
    video.release()
    cv2.destroyAllWindows()


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tensorflow.Session() as sess:
            # Get handles to input and output tensors
            ops = tensorflow.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in
                                op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[
                        key] = tensorflow.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tensorflow.squeeze(
                    tensor_dict['detection_boxes'], [0],
                )
                detection_masks = tensorflow.squeeze(
                    tensor_dict['detection_masks'], [0],
                )

                # Reframe is required to translate mask from box coordinates to
                # image coordinates and fit the image size.
                real_num_detection = tensorflow.cast(
                    tensor_dict['num_detections'][0], tensorflow.int32,
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
                        image.shape[0],
                        image.shape[1],
                    )
                )
                detection_masks_reframed = tensorflow.cast(
                    tensorflow.greater(detection_masks_reframed, 0.5),
                    tensorflow.uint8,
                )

                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tensorflow.expand_dims(
                    detection_masks_reframed, 0,
                )

            image_tensor = tensorflow.get_default_graph().get_tensor_by_name(
                'image_tensor:0',
            )

            # Run inference
            output_dict = sess.run(
                tensor_dict,
                feed_dict={image_tensor: numpy.expand_dims(image, 0)},
            )

            # All outputs are float32 numpy arrays, so convert types as
            # appropriate.
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0],
            )
            output_dict['detection_classes'] = (
                output_dict['detection_classes'][0].astype(numpy.uint8)
            )
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = (
                output_dict['detection_scores'][0]
            )

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = (
                    output_dict['detection_masks'][0]
                )

    return output_dict


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size

    return (
        numpy
            .array(image.getdata())
            .reshape((im_height, im_width, 3))
            .astype(numpy.uint8)
    )


@detect.command("download-models")
def download_models():
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if "frozen_inference_graph.pb" in file_name:
            tar_file.extract(file, os.getcwd())
