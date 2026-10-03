"""Hoogtepunten in het geluid: een nagemaakte opname met achtergrondgeruis, een fluitsignaal en gejuich."""
import numpy as np

from app import audio


def _recording(seconds=60, seed=0):
    rng = np.random.default_rng(seed)
    sr = audio.SR
    t = np.arange(seconds * sr) / sr
    x = 0.02 * rng.normal(size=len(t))  # geroezemoes
    x += 0.05 * np.sin(2 * np.pi * 3 * t) * rng.normal(size=len(t)) * 0.2  # wat wind (laag)
    # fluitsignaal: 3,1 kHz met boventoon, 1 s vanaf 20 s
    w = (t >= 20) & (t < 21)
    x[w] += 0.08 * np.sin(2 * np.pi * 3100 * t[w]) + 0.02 * np.sin(2 * np.pi * 6200 * t[w])
    # gejuich: 4 s lang veel luider (ruis in het stemgebied) vanaf 40 s
    c = (t >= 40) & (t < 44)
    burst = rng.normal(size=c.sum())
    spec = np.fft.rfft(burst)
    f = np.fft.rfftfreq(len(burst), 1 / sr)
    spec[(f < 300) | (f > 4000)] = 0
    x[c] += 0.25 * np.fft.irfft(spec, n=len(burst)) / np.std(np.fft.irfft(spec, n=len(burst)))
    return x.astype(np.float32)


def test_finds_whistle_and_cheering():
    ev = audio.find_events(_recording())
    cheers = [e for e in ev if e["kind"] == "gejuich"]
    whistles = [e for e in ev if e["kind"] == "fluitsignaal"]
    assert len(cheers) == 1 and 39.5 <= cheers[0]["t"] <= 40.5 and cheers[0]["t_end"] >= 43.5, ev
    assert len(whistles) == 1 and 19.5 <= whistles[0]["t"] <= 20.5, ev


def test_quiet_recording_has_no_events():
    rng = np.random.default_rng(1)
    x = (0.02 * rng.normal(size=60 * audio.SR)).astype(np.float32)
    assert audio.find_events(x) == []


def _burst(x, start, dur, rng, gain=0.25):
    sr = audio.SR
    a, b = int(start * sr), int((start + dur) * sr)
    burst = rng.normal(size=b - a)
    spec = np.fft.rfft(burst)
    f = np.fft.rfftfreq(len(burst), 1 / sr)
    spec[(f < 300) | (f > 4000)] = 0
    y = np.fft.irfft(spec, n=len(burst))
    x[a:b] += gain * y / np.std(y)


def test_short_shout_is_not_cheering_and_one_per_half_minute():
    rng = np.random.default_rng(2)
    x = (0.02 * rng.normal(size=90 * audio.SR)).astype(np.float32)
    _burst(x, 10, 0.8, rng)  # een losse roep
    _burst(x, 40, 3.0, rng, gain=0.2)  # gejuich in twee golven vlak na elkaar
    _burst(x, 46, 3.0, rng, gain=0.3)
    cheers = [e for e in audio.find_events(x) if e["kind"] == "gejuich"]
    assert len(cheers) == 1 and 45.5 <= cheers[0]["t"] <= 46.5, cheers  # alleen de sterkste golf
