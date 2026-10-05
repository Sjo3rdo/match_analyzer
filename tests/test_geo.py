import math

import numpy as np

from app import geo


def _pitch_polygon(lat0, lon0, rot_deg, length=100.0, width=64.0):
    """Rechthoek (lat, lon) rond (lat0, lon0), gedraaid; zoals OpenStreetMap hem zou geven."""
    r = 6371000.0
    a = math.radians(rot_deg)
    u, v = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    pts = [sx * length / 2 * u + sy * width / 2 * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
    to_ll = lambda e, n: (lat0 + math.degrees(n / r), lon0 + math.degrees(e / (r * math.cos(math.radians(lat0)))))  # noqa: E731
    return [to_ll(*p) for p in pts], u, v


def test_iso6709():
    assert geo.parse_iso6709("+53.1425+006.3756+005.919/") == (53.1425, 6.3756)
    assert geo.parse_iso6709("-33.8688+151.2093/") == (-33.8688, 151.2093)
    assert geo.parse_iso6709("onzin") is None


def test_camera_position_from_pitch_polygon():
    lat0, lon0 = 53.1425, 6.3756
    for rot in (0, 30, 115, 200):
        poly, u, v = _pitch_polygon(lat0, lon0, rot)
        # camera 6 m buiten een zijlijn, 20 m van het midden richting één doel
        cam = 20 * u + (32 + 6) * v
        r = 6371000.0
        clat = lat0 + math.degrees(cam[1] / r)
        clon = lon0 + math.degrees(cam[0] / (r * math.cos(math.radians(lat0))))
        pos = geo.camera_on_pitch(clat, clon, [poly, _pitch_polygon(lat0 + 0.01, lon0, 0)[0]])
        assert abs(pos["pitch_length"] - 100) < 1 and abs(pos["pitch_width"] - 64) < 1
        # aan de 'onderste' zijlijn, 6 m (geschaald naar 68 m breed) daarbuiten
        assert abs(pos["y"] - (68 + 6 * 68 / 64)) < 0.5, (rot, pos)
        # 20 m uit het midden: links of rechts hangt af van de kijkrichting; gespiegeld blijft het 20 m
        assert abs(abs(pos["x"] - 52.5) - 20 * 105 / 100) < 0.5, (rot, pos)


def test_x_runs_left_to_right_as_seen_from_camera():
    """Camera ten zuiden van een oost-west liggend veld kijkt naar het noorden: oost = rechts = grote x."""
    lat0, lon0 = 52.0, 5.0
    poly, u, v = _pitch_polygon(lat0, lon0, 0)  # lengte oost-west
    r = 6371000.0
    clat = lat0 - math.degrees(40 / r)  # 40 m ten zuiden van het midden (8 m buiten de zijlijn)
    clon = lon0 + math.degrees(30 / (r * math.cos(math.radians(lat0))))  # 30 m naar het oosten
    pos = geo.camera_on_pitch(clat, clon, [poly])
    assert pos["x"] > 52.5 + 25 and pos["y"] > 68


def test_osm_search_falls_back_and_explains_problems(monkeypatch, tmp_path):
    """De zoekservers worden tegelijk gevraagd; antwoordt er geen, dan het kaartje rechtstreeks van
    openstreetmap.org; lukt ook dat niet, dan een melding in gewone woorden. Gevonden velden worden onthouden."""
    import io
    import json
    import ssl
    import urllib.error
    import pytest
    from app import config
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    calls = []
    answer = {"elements": [{"tags": {"sport": "soccer"},
                            "geometry": [{"lat": 53.0, "lon": 6.0}, {"lat": 53.001, "lon": 6.0},
                                         {"lat": 53.001, "lon": 6.0015}, {"lat": 53.0, "lon": 6.0015}]}]}
    osm_xml = (b'<osm><node id="1" lat="53.0" lon="6.0"/><node id="2" lat="53.001" lon="6.0"/>'
               b'<node id="3" lat="53.001" lon="6.0015"/><node id="4" lat="53.0" lon="6.0015"/>'
               b'<way id="9"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/>'
               b'<tag k="leisure" v="pitch"/><tag k="sport" v="soccer"/></way>'
               b'<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><tag k="building" v="yes"/></way></osm>')

    def fake(mode):
        def urlopen(req, timeout=None, context=None):
            calls.append(req.full_url)
            assert context is not None and req.get_header("User-agent").startswith("match-analyzer")
            osm_api = req.full_url.startswith(geo.OSM_API)
            if mode == "one-busy" and req.full_url == geo.OVERPASS_URLS[0]:
                raise urllib.error.HTTPError(req.full_url, 504, "Gateway Timeout", {}, None)
            if mode == "overpass-down" and not osm_api:
                raise urllib.error.URLError(TimeoutError("The read operation timed out"))
            if mode == "overpass-down":
                return io.BytesIO(osm_xml)
            if mode == "ssl":
                raise urllib.error.URLError(ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED"))
            if mode == "busy":
                raise urllib.error.HTTPError(req.full_url, 504, "Gateway Timeout", {}, None)
            return io.BytesIO(json.dumps(answer).encode())
        return urlopen
    monkeypatch.setattr(geo.urllib.request, "urlopen", fake("one-busy"))
    assert len(geo.query_pitches(53.0, 6.0)) == 1  # een andere zoekserver antwoordde
    calls.clear()
    assert len(geo.query_pitches(53.0, 6.0)) == 1 and calls == []  # onthouden: niet opnieuw zoeken
    monkeypatch.setattr(geo.urllib.request, "urlopen", fake("overpass-down"))
    polys = geo.query_pitches(53.2, 6.2)
    assert len(polys) == 1 and len(polys[0]) == 5 and any(c.startswith(geo.OSM_API) for c in calls)
    monkeypatch.setattr(geo.urllib.request, "urlopen", fake("ssl"))
    with pytest.raises(geo.OsmError, match="certificaten"):
        geo.query_pitches(53.3, 6.3)
    monkeypatch.setattr(geo.urllib.request, "urlopen", fake("busy"))
    with pytest.raises(geo.OsmError, match="druk"):
        geo.query_pitches(53.4, 6.4)
