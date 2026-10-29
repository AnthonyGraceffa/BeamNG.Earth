"""The sidewalks' lean collision mesh (sidewalks.Builder(collision=True), the tile's "walkcol")."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cdae  # noqa: E402
import sidewalks  # noqa: E402


def run(n=200, bend=0.0):
    s = np.linspace(0.0, 100.0, n)
    curb = np.stack([s, bend * np.sin(s / 15.0), 0.02 * s], 1)
    out = np.tile([0.0, -1.0], (n, 1))
    return curb, out


class NarrowBandTest(unittest.TestCase):
    def test_profile_never_folds(self):
        # GitHub #55 "the median eats tires": a band narrower than RAMP_RUN folded back over itself (its ramp end past
        # its outer edge), and the folded collision caught wheels
        for w in (0.3, 0.6, 1.2, 1.8, 2.5, 4.0):
            for hr in (sidewalks.CURB_H, 0.02):
                x = sidewalks.profile(sidewalks.CURB_H, w, hr)[:, 0]
                self.assertTrue((np.diff(x) > 0).all(), (w, hr, x.tolist()))
        np.testing.assert_allclose(sidewalks.profile(sidewalks.CURB_H, 2.5)[5], [sidewalks.RAMP_RUN,
                                   sidewalks.CURB_H + sidewalks.RAMP_RUN * sidewalks.CROSS_SLOPE], err_msg="wide: as before")


class SimplifyTest(unittest.TestCase):
    def test_straight_line_to_its_ends(self):
        pts = np.stack([np.linspace(0, 10, 50), np.linspace(0, 5, 50), np.zeros(50)], 1)
        self.assertEqual(sidewalks.simplify(pts).tolist(), [0, 49])

    def test_keeps_corners_and_within_tolerance(self):
        t = np.linspace(0, 2 * np.pi, 400)
        pts = np.stack([20 * np.cos(t), 20 * np.sin(t)], 1)
        keep = sidewalks.simplify(pts, 0.03)
        self.assertLess(len(keep), 120)
        self.assertEqual((keep[0], keep[-1]), (0, 399))
        # every dropped point within tol of the kept polyline
        k = pts[keep]
        for p in pts[::7]:
            a, b = k[:-1], k[1:]
            ab = b - a
            tt = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
            d = np.hypot(*(a + tt[:, None] * ab - p).T).min()
            self.assertLessEqual(d, 0.03 + 1e-9)

    def test_short(self):
        self.assertEqual(sidewalks.simplify(np.zeros((2, 3))).tolist(), [0, 1])
        self.assertEqual(sidewalks.simplify(np.zeros((1, 3))).tolist(), [0])


class LeanMeshTest(unittest.TestCase):
    def build(self, **kw):
        b = sidewalks.Builder((0.0, 0.0, 0.0), collision=True)
        curb, out = run(**kw)
        b.add_run(curb, out, 2.5, ramps=[(40.0, 42.0)])
        b.add_corner(curb[-10:], np.tile([0.0, -1.0], (10, 1)), [[100, -2.5, 2.0], [102.5, -2.5, 2.0], [102.5, 0, 2.0]])
        return b

    def test_fewer_triangles_same_surface(self):
        b = self.build(bend=0.0)
        P, T, _, _ = b.arrays()
        CP, CT, _, _ = b.col.arrays()
        self.assertLess(len(CT), len(T) / 4)
        # the slab top at its outer edge: the same height in both meshes (the lean keeps profile points 6)
        self.assertAlmostEqual(P[:, 2].max(), CP[:, 2].max(), places=6)
        self.assertAlmostEqual(P[:, 2].min(), CP[:, 2].min(), places=6)

    def test_ramp_kept(self):
        b = self.build(bend=0.0)
        CP = b.col.arrays()[0]
        # a point at the lowered curb (RAMP_H above the road) survives the simplification
        s = CP[:, 0]
        road = 0.02 * s
        low = np.abs(CP[:, 2] - road - sidewalks.RAMP_H) < 1e-6
        self.assertTrue(low.any())

    def test_written_as_collision_shape(self):
        b = self.build(bend=3.0)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "walkcol.cdae")
            b.write_collision(p)
            shape = cdae.read(p)
        self.assertEqual(shape["names"], ["Colmesh", "detail2", "Collision-1"])   # (detail2: a null mesh, cdae.encode)

    def test_without_collision_no_lean_builder(self):
        self.assertIsNone(sidewalks.Builder((0.0, 0.0, 0.0)).col)


class StripByStrip(sidewalks.Builder):
    """The Builder with its strips made one at a time as before 2026-10-25 (np.cross, a profile() per point): the
    batched Builder must give the very same arrays."""

    @staticmethod
    def _orient_ref(tris, P, want):
        a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
        flip = (np.cross(b - a, c - a) * want).sum(-1) < 0
        tris = tris.copy()
        tris[flip] = tris[flip][:, ::-1]
        return tris

    def _sections(self, curb, toward_out, w, h, hr):
        prof = np.stack([sidewalks.profile(h, w, float(x)) for x in hr])
        ring = np.empty((len(curb), prof.shape[1], 3))
        ring[:, :, :2] = curb[:, None, :2] + toward_out[:, None, :] * prof[:, :, 0:1]
        ring[:, :, 2] = curb[:, None, 2] + prof[:, :, 1]
        return ring - self.origin, prof

    def _strips(self, ring, prof, toward_out, s, strips, tactile=None, slab=1):
        k = len(ring)
        for a, b, mat, va, vb in strips:
            rep = sidewalks.CURB_REPEAT_M if mat == 0 else sidewalks.SLAB_REPEAT_M
            P = np.concatenate([ring[:, a], ring[:, b]])
            UV = np.concatenate([np.stack([s / rep, np.full(k, va)], 1), np.stack([s / rep, np.full(k, vb)], 1)])
            if mat == 1 and slab == sidewalks.BRICK:
                br = sidewalks.BRICK_REPEAT_M
                UV = np.concatenate([np.stack([s / br, prof[:, a, 0] / br], 1),
                                     np.stack([s / br, prof[:, b, 0] / br], 1)])
                mat = sidewalks.BRICK
            mats = np.full(k - 1, mat)
            if tactile is not None and a == 3:
                mats[tactile] = 2
                tu = s / sidewalks.TACTILE_REPEAT_M
                pad = np.concatenate((tactile, tactile[-1:]))
                UV[:k, 0] = np.where(pad, tu, UV[:k, 0])
                UV[k:, 0] = np.where(pad, tu, UV[k:, 0])
            base = self._add(P, UV)
            i = np.arange(k - 1)
            A, A1, B1, B = base + i, base + i + 1, base + k + i + 1, base + k + i
            tris = np.concatenate([np.stack([A, B1, A1], 1), np.stack([A, B, B1], 1)])
            dp = 0.5 * ((prof[:-1, b] - prof[:-1, a]) + (prof[1:, b] - prof[1:, a]))
            o3 = np.concatenate([toward_out[i], np.zeros((k - 1, 1))], 1)
            want = -dp[:, 1:2] * o3 + dp[:, 0:1] * np.array([0.0, 0.0, 1.0])
            self.T.append(base + self._orient_ref(tris - base, P, np.concatenate([want, want])))
            self.M.append(np.r_[mats, mats])


class BatchedTest(unittest.TestCase):
    """The batched strips, profiles and orientation (2026-10-25: ~2x faster sidewalk meshes) give the same numbers."""

    def test_profiles_point_by_point(self):
        hr = np.array([sidewalks.CURB_H, 0.1, 0.05, sidewalks.RAMP_H, 0.0, 0.12])
        for h in (sidewalks.CURB_H, 0.1, 0.0):
            for w in (0.3, 1.2, 2.5, 4.0):
                want = np.stack([sidewalks.profile(h, w, float(x)) for x in hr])
                np.testing.assert_array_equal(sidewalks.profiles(h, w, hr), want)

    def test_orient_as_np_cross(self):
        rng = np.random.default_rng(5)
        P = rng.normal(0.0, 10.0, (60, 3))
        P[5] = P[4]                                     # a zero-area triangle or two
        tris = rng.integers(0, 60, (400, 3))
        for want in (rng.normal(0.0, 1.0, (400, 3)), np.array([0.0, 0.0, 1.0])):
            np.testing.assert_array_equal(sidewalks._orient(tris, P, want), StripByStrip._orient_ref(tris, P, want))

    def test_same_arrays_as_strip_by_strip(self):
        rng = np.random.default_rng(7)
        for trial in range(40):
            col = trial % 2 == 0
            new = sidewalks.Builder((3.0, -2.0, 1.0), collision=col)
            ref = StripByStrip((3.0, -2.0, 1.0), collision=col)
            for _ in range(3):
                k = int(rng.integers(2, 30))
                ang = np.cumsum(rng.normal(0.0, 0.2, k))
                xy = np.stack([np.cumsum(2.0 * np.cos(ang)), np.cumsum(2.0 * np.sin(ang))], 1)
                curb, out = sidewalks.run_geometry(np.c_[xy, 10.0 + np.cumsum(rng.normal(0.0, 0.05, k))], 1, 2.5)
                L = float(np.hypot(*np.diff(curb[:, :2], axis=0).T).sum())
                ramps = [(L * 0.3, L * 0.3 + 1.2)] if trial % 3 else []
                slab = sidewalks.BRICK if trial % 4 == 1 else 1
                w = float(rng.choice([0.6, 2.5, 3.5]))
                for b in (new, ref):
                    b.add_run(curb, out, w, sidewalks.CURB_H, (True, trial % 5 > 0), ramps, slab)
            arc, _ = run(12, bend=2.0)
            outer = [[100, -2.5, 2.0], [102.5, -2.5, 2.0], [102.5, 0, 2.0]]
            for b in (new, ref):
                b.add_corner(arc, np.tile([0.0, -1.0], (12, 1)), outer, ramps=[(0.5, 1.7)] if trial % 2 else [])
            for got, want in zip(new.arrays(), ref.arrays()):
                np.testing.assert_array_equal(got, want)
            if col:
                for got, want in zip(new.col.arrays(), ref.col.arrays()):
                    np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
