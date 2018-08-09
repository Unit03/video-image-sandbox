import os
import pathlib

import click
import cv2

from . import detect

data_dir_path = pathlib.Path("/home/unit03/virtualenvs/video-image-sandbox/lib/python3.6/site-packages/cv2/data/")
HAAR_FACE_FILE_PATH = data_dir_path / "haarcascade_frontalface_default.xml"
HAAR_EYE_FILE_PAT = data_dir_path / "haarcascade_eye.xml"
assert os.path.exists(HAAR_FACE_FILE_PATH)
assert os.path.exists(HAAR_EYE_FILE_PAT)

face_cascade = cv2.CascadeClassifier(str(HAAR_FACE_FILE_PATH))
eye_cascade = cv2.CascadeClassifier(str(HAAR_EYE_FILE_PAT))


@detect.command("faces")
# @click.pass_context
@click.argument(
    "video_file_path",
    type=click.Path(
        exists=True, file_okay=True, readable=True, resolve_path=True,
    ),
)
def faces(video_file_path: click.Path):
    print(video_file_path)
    vc = cv2.VideoCapture(video_file_path)
    while (vc.isOpened()):
        ret, frame = vc.read()
        if frame is None:
            break

        gray = cv2.resize(frame, None, fx=0.4, fy=0.4,
                          interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = gray[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh),
                              (0, 255, 0), 2)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()
