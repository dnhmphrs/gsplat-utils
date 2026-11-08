import json

from gsplat_utils.cli import build_parser, main


def test_parser_has_all_commands():
    ap = build_parser()
    sub = next(a for a in ap._actions if a.dest == "cmd")
    assert {"ingest", "sfm", "use-model", "align", "train", "export", "geometry", "poses", "web", "run", "status"} <= set(sub.choices)


def test_status_on_empty_workspace(tmp_path, capsys):
    assert main(["status", "--work", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "frames: none" in out and "sfm: none" in out and "geometry: none" in out


def test_run_parser_merges_stage_flags():
    ap = build_parser()
    a = ap.parse_args(["run", "--work", "w", "--video", "v.mp4", "--frames", "50", "--preset", "video",
                       "--camera-height", "1.6", "--cell", "0.2", "--source", "sfm", "--model", "m"])
    assert a.frames == 50 and a.preset == "video" and a.camera_height == 1.6 and a.cell == 0.2
    assert a.source == "sfm" and str(a.model) == "m"
