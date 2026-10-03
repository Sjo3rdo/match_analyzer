import numpy as np

from app.render import SpotlightTrack, _audio_filter, draw_shapes, draw_spotlight


def test_spotlight_interpolates_between_analysed_frames():
    tr = SpotlightTrack([{"t": 0.0, "x1": 0, "y1": 0, "x2": 10, "y2": 20},
                         {"t": 0.1, "x1": 10, "y1": 0, "x2": 20, "y2": 20}])
    assert np.allclose(tr.box_at(0.05), [5, 0, 15, 20])
    assert tr.box_at(5.0) is None  # speler niet in beeld


def test_drawing_functions_change_pixels():
    img = np.zeros((200, 300, 3), np.uint8)
    draw_shapes(img, [{"type": "arrow", "from": [0.1, 0.1], "to": [0.9, 0.9]}, {"type": "text", "at": [0.2, 0.5], "text": "Hoi"}])
    draw_spotlight(img, np.array([100, 50, 130, 150]), "9 Jan")
    assert img.sum() > 0


def test_audio_filter_inserts_silence():
    _, f = _audio_filter(10.0, [(3.0, 4.0)], True)
    assert "atrim=start=0.000:end=3.000" in f and "atrim=duration=4.000" in f and "concat=n=3" in f
