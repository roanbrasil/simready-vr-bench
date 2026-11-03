from srvr.search import build_body, parse_hits, resolvable


def test_blob_to_resolve():
    u = "https://huggingface.co/datasets/frankarobotics/simready-assets/blob/89b84c2/franka_hand/simready_usd/franka_hand.usda"
    assert resolvable(u) == u.replace("/blob/", "/resolve/")
    s3 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/x.usd"
    assert resolvable(s3) == s3


def test_parse_hits_fields():
    payload = {"hits": [{"rrf_score": 0.03, "metadata": {"explanations": [
        {"matched_vectors": [{"image": "abc123"}]}]},
        "source": {"name": "Robotiq_2F_85.usda",
                   "base_key": "https://huggingface.co/datasets/R/s/blob/sha/a/simready_usd/a.usda",
                   "usd_properties": [{"name": "isaac:physics:robotJoints", "value": "/a/j1"},
                                      {"name": "simready_metadata_validation__profile", "value": "Robot-Gripper"}]}}]}
    h = parse_hits(payload)[0]
    assert "/resolve/" in h.url and h.is_robot and h.validated_profile == "Robot-Gripper"
    assert h.thumbnail_url.endswith("/images?image_key=abc123")
    assert build_body("x", 5)["limit"] == 5
