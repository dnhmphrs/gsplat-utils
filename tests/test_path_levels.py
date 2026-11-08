import numpy as np

from gsplat_utils.geometry.path import segment_levels, track_inliers


def _path(profile, step=0.25):
    """profile: list of (length_m, dz_m) straight pieces along +x."""
    pts = [np.array([0.0, 0.0, 1.5])]
    for length, dz in profile:
        n = int(round(length / step))
        for k in range(1, n + 1):
            p = pts[-1].copy() if k == 1 else pts[-1].copy()
            pts.append(pts[-1] + np.array([step, 0.0, dz / n]))
    return np.array(pts)


def test_landing_inside_staircase_is_one_flight():
    # flat 6 m, flight 3 m (+1.6), landing 2 m, flight 3 m (+1.6), flat 8 m
    pts = _path([(6, 0), (3, 1.6), (2, 0), (3, 1.6), (8, 0)])
    segs, levels = segment_levels(pts, 0.25, 1.5)
    kinds = [s.kind for s in segs]
    assert kinds == ["flat", "stairs", "flat"], kinds
    assert len(levels) == 2
    assert abs((levels[1].floor_z - levels[0].floor_z) - 3.2) < 0.3


def test_up_across_down_returns_to_same_level():
    pts = _path([(5, 0), (4, 3.0), (12, 0), (4, -3.0), (5, 0.4)])  # 0.4 m of drift at the end
    segs, levels = segment_levels(pts, 0.25, 1.5)
    kinds = [s.kind for s in segs]
    assert kinds == ["flat", "stairs", "flat", "stairs", "flat"], kinds
    assert len(levels) == 2, [(l.id, l.camera_z) for l in levels]
    assert segs[0].level == segs[-1].level == 0 and segs[2].level == 1
    assert segs[1].level_from == 0 and segs[1].level_to == 1
    assert segs[3].level_from == 1 and segs[3].level_to == 0


def test_track_inliers_drops_jumps():
    rng = np.random.default_rng(0)
    pos = np.c_[np.linspace(0, 20, 200), np.zeros(200), np.full(200, 1.5)] + rng.normal(0, 0.01, (200, 3))
    pos[50] += [0, 5.0, 0]
    pos[120] += [3.0, -4.0, 2.0]
    keep = track_inliers(pos)
    assert not keep[50] and not keep[120] and keep.sum() == 198


def test_atrium_detection_and_void():
    from gsplat_utils.geometry.stairs import atrium_void, is_atrium, loopiness

    # square spiral around a 3x3 m void with 1.2 m wide flights: centreline at 0.6 m off the void
    r = 0.6
    sq = np.array([[-r, -r], [3 + r, -r], [3 + r, 3 + r], [-r, 3 + r], [-r, -r], [3 + r, -r]])
    pts = np.vstack([np.linspace(sq[i], sq[i + 1], 12, endpoint=False) for i in range(len(sq) - 1)])
    assert loopiness(pts) > 0.4
    assert is_atrium(pts)
    straight = np.c_[np.linspace(0, 8, 30), np.zeros(30)] + np.random.default_rng(0).normal(0, 0.03, (30, 2))
    assert not is_atrium(straight)
    void = atrium_void(pts, 0.0, 0.6)
    assert void is not None
    lo, hi = void.min(axis=0), void.max(axis=0)
    assert np.allclose(lo, [0.0, 0.0], atol=0.02) and np.allclose(hi, [3.0, 3.0], atol=0.02)
