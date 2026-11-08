"""Video -> frames.

Two ffmpeg passes:
1. decode every frame at low resolution in grayscale and score sharpness
   (variance of Laplacian) so blurred frames are never chosen;
2. extract the chosen frames losslessly-ordered with a `select` filter.
Frame files are named by *source frame index* so they stay traceable to the
video timeline (needed for sequential matching and the camera path).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from ..common import find_ffmpeg, find_ffprobe, log, write_json
from .select import motion_score, pick_by_motion, pick_per_window, sharpness_score


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    n_frames: int


def _parse_rate(s: str) -> float:
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else 0.0
    return float(s)


def probe_video(path: os.PathLike | str) -> VideoInfo:
    path = str(path)
    ffprobe = find_ffprobe()
    if ffprobe is None:
        raise RuntimeError("ffprobe not found next to ffmpeg; install ffmpeg")
    out = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_packets,duration",
            "-show_entries", "format=duration",
            "-of", "json", path,
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    st = info["streams"][0]
    fps = _parse_rate(st.get("avg_frame_rate") or st.get("r_frame_rate") or "0")
    if fps <= 0:
        fps = _parse_rate(st.get("r_frame_rate") or "30")
    dur = float(st.get("duration") or info.get("format", {}).get("duration") or 0.0)
    n = int(st.get("nb_frames") or st.get("nb_read_packets") or 0)
    if n <= 0 and dur > 0:
        n = int(round(dur * fps))
    return VideoInfo(path, int(st["width"]), int(st["height"]), fps, dur, n)


def score_video_frames(
    video: VideoInfo,
    score_height: int = 240,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode the whole video once at low-res grayscale.

    Returns (sharpness[n], motion[n], frame_index[n]) for decoded frames in order.
    """
    ffmpeg = find_ffmpeg()
    h = int(score_height)
    w = int(round(video.width * h / video.height / 2) * 2)
    cmd = [ffmpeg, "-v", "error", "-nostdin"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", video.path]
    if end_s is not None:
        cmd += ["-t", f"{max(0.0, end_s - start_s):.3f}"]
    # passthrough: never duplicate/drop frames, so the frame counter `n` here is
    # identical to the counter the `select` filter sees in the extraction pass.
    cmd += ["-vf", f"scale={w}:{h}", "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    log.debug("$ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    nbytes = w * h
    sharp: list[float] = []
    motion: list[float] = []
    prev = None
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        g = np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
        sharp.append(sharpness_score(g))
        motion.append(motion_score(prev, g))
        prev = g
    proc.stdout.close()
    err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg scoring pass failed ({rc}): {err[-2000:]}")
    idx0 = int(round(start_s * video.fps)) if start_s > 0 else 0
    n = len(sharp)
    return np.array(sharp), np.array(motion), np.arange(idx0, idx0 + n)


def extract_video_frames(
    video_path: os.PathLike | str,
    out_dir: os.PathLike | str,
    target_frames: int = 300,
    every: int | None = None,
    max_size: int | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
    jpeg_quality: int = 2,
    min_gap_s: float = 0.0,
    drop_blur: float = 0.1,
    sampling: str = "motion",
    meta_path: os.PathLike | str | None = None,
    add: bool = False,
) -> dict:
    """Extract frames and write `frames_meta.json`.

    target_frames: number of windows; the sharpest frame of each window is kept.
    every: alternative simple mode, keep every N-th decoded frame (no scoring).
    max_size: cap the long image side (pixels); None keeps native resolution.
    jpeg_quality: ffmpeg -q:v (2 = visually lossless, 31 = worst).
    drop_blur: drop picked frames whose sharpness is below this fraction of the
        video's median sharpness (0 disables). Such frames rarely register in SfM.
    sampling: "motion" = windows of equal visual change (dense where the camera
        turns/moves fast, sparse when still); "uniform" = equal time windows.
    add: keep the existing frame set and only add frames (use with --start/--end
        and --every 1 to densify a stretch where the reconstruction broke).
    """
    video = probe_video(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_file = Path(meta_path) if meta_path else (out_dir.parent / "frames_meta.json")
    existing = {}
    if add and meta_file.exists():
        from ..common import read_json

        old_meta = read_json(meta_file)
        existing = {f["name"]: f for f in old_meta.get("frames", [])}
    else:
        for old in out_dir.glob("frame_*.jpg"):
            old.unlink()

    log.info(
        "video %dx%d @ %.3f fps, %.1fs, ~%d frames",
        video.width, video.height, video.fps, video.duration_s, video.n_frames,
    )
    sharp, motion, frame_idx = score_video_frames(video, start_s=start_s, end_s=end_s)
    n = len(sharp)
    if n == 0:
        raise RuntimeError("no frames decoded")

    if every and every > 0:
        picks = list(range(0, n, every))
    else:
        min_gap = max(1, int(round(min_gap_s * video.fps))) if min_gap_s > 0 else 1
        if sampling == "motion":
            picks = pick_by_motion(sharp, motion, target_frames, min_gap=min_gap)
        else:
            picks = pick_per_window(sharp, target_frames, min_gap=min_gap)
    picks = sorted(set(picks))
    if existing:
        picks = [p for p in picks if f"frame_{int(frame_idx[p]):06d}.jpg" not in existing]
        log.info("--add: %d new frames (%d already present)", len(picks), len(existing))
        if not picks:
            return old_meta
    if drop_blur > 0:
        thr = drop_blur * float(np.median(sharp))
        n_before = len(picks)
        picks = [p for p in picks if sharp[p] >= thr]
        if len(picks) < n_before:
            log.info("dropped %d very blurry frames (sharpness < %.1f)", n_before - len(picks), thr)
    log.info(
        "selected %d/%d frames (sharpness median %.1f -> picked median %.1f)",
        len(picks), n, float(np.median(sharp)), float(np.median(sharp[picks])),
    )

    # Pass 2: extract exactly those decoded frames with a select filter read from
    # a filter script (keeps the command line short for thousands of frames).
    ffmpeg = find_ffmpeg()
    sel = "+".join(f"eq(n\\,{int(p)})" for p in picks)
    vf = f"select='{sel}'"
    if max_size:
        vf += f",scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(vf)
        script = f.name
    tmp_pattern = str(out_dir / "_sel_%06d.jpg")
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-y"]
    if start_s > 0:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", video.path]
    if end_s is not None:
        cmd += ["-t", f"{max(0.0, end_s - start_s):.3f}"]
    cmd += ["-filter_script:v", script, "-fps_mode", "vfr", "-q:v", str(jpeg_quality), tmp_pattern]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:  # pragma: no cover
        raise RuntimeError(f"ffmpeg extract pass failed: {e.stderr.decode(errors='replace')[-2000:]}")
    finally:
        os.unlink(script)

    produced = sorted(out_dir.glob("_sel_*.jpg"))
    if len(produced) != len(picks):
        log.warning("expected %d frames, ffmpeg wrote %d; keeping the overlap", len(picks), len(produced))
    frames = []
    for src, p in zip(produced, picks):
        fi = int(frame_idx[p])
        name = f"frame_{fi:06d}.jpg"
        os.replace(src, out_dir / name)
        frames.append({
            "name": name,
            "index": fi,
            "time_s": round(fi / video.fps, 4) if video.fps else None,
            "sharpness": round(float(sharp[p]), 2),
            "motion": round(float(motion[p]), 2),
        })
    for extra in produced[len(picks):]:
        extra.unlink()

    if existing:
        meta = old_meta
        meta["frames"] = sorted(list(existing.values()) + frames, key=lambda f: f["index"])
        meta.setdefault("additions", []).append({"start_s": start_s, "end_s": end_s, "every": every,
                                                 "target_frames": target_frames, "n_added": len(frames)})
    else:
        meta = {
            "source": {"type": "video", **asdict(video)},
            "sequential": True,
            "selection": {
                "target_frames": target_frames, "every": every, "max_size": max_size,
                "start_s": start_s, "end_s": end_s, "min_gap_s": min_gap_s, "drop_blur": drop_blur,
                "sampling": sampling,
            },
            "frames": frames,
        }
    write_json(meta_file, meta)
    return meta
