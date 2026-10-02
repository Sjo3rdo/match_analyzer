import cv2
import numpy as np

from app.pipeline import read_frame


def test_read_frame_is_exact(tmp_path):
    # elk frame krijgt een eigen grijswaarde, zodat we kunnen zien welk frame terugkomt
    path = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(90):
        w.write(np.full((120, 160, 3), 2 * i + 10, np.uint8))
    w.release()
    cap = cv2.VideoCapture(str(path))
    seq = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        seq.append(f.astype(int))
    for t in (0.0, 0.5, 1.0, 2.0, 2.9):
        frame, ft = read_frame(path, t)
        idx = int(np.argmin([np.abs(frame.astype(int) - q).mean() for q in seq]))
        assert idx == round(t * 30), (t, idx)
        assert abs(ft - t) < 1 / 60
