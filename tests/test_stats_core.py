#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""stats_core 的冒烟测试：只用合成数据，不需要 MIMIC-IV、不需要数据库。

运行：  python -m unittest discover -s tests -v
"""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import stats_core as sc  # noqa: E402


def synthetic(n=600, seed=7):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-1.2 + 0.9 * z)))
    y = rng.binomial(1, p)
    return y, p, z


class TestDiscrimination(unittest.TestCase):
    def test_delong_auc_matches_sklearn(self):
        from sklearn.metrics import roc_auc_score
        y, p, _ = synthetic()
        auc, lo, hi, se = sc.delong_ci(y, p)
        self.assertAlmostEqual(auc, roc_auc_score(y, p), places=10)
        self.assertLess(lo, auc)
        self.assertGreater(hi, auc)
        self.assertGreater(se, 0)

    def test_bootstrap_contains_point_estimate(self):
        y, p, _ = synthetic()
        lo, hi = sc.bootstrap_auc(y, p, n_boot=300)
        from sklearn.metrics import roc_auc_score
        self.assertLessEqual(lo, roc_auc_score(y, p))
        self.assertGreaterEqual(hi, roc_auc_score(y, p))

    def test_paired_boot_identical_predictions(self):
        y, p, _ = synthetic()
        mean, lo, hi, pv = sc.paired_boot(y, p, p, n_boot=300)
        self.assertAlmostEqual(mean, 0.0, places=10)
        self.assertAlmostEqual(pv, 1.0, places=10)

    def test_stratified_boot_keeps_shape(self):
        y, p, _ = synthetic()
        d0, lo, hi, pv = sc.stratified_paired_boot(y, p, p, n_boot=300)
        self.assertAlmostEqual(d0, 0.0, places=10)
        self.assertLessEqual(lo, 0.0)
        self.assertGreaterEqual(hi, 0.0)
        self.assertTrue(0.0 <= pv <= 1.0)


class TestCalibration(unittest.TestCase):
    def test_slope_of_well_calibrated_model(self):
        rng = np.random.default_rng(11)
        n = 20000
        z = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(-1.0 + 0.8 * z)))
        y = rng.binomial(1, p)
        slope = sc.calib_slope(y, p)
        self.assertGreater(slope, 0.90)
        self.assertLess(slope, 1.10)

    def test_slope_not_ols_degenerate(self):
        """回归测试：OLS 会把斜率压到 ~0.1，logistic 不应如此。"""
        y, p, _ = synthetic(n=4000, seed=3)
        self.assertGreater(sc.calib_slope(y, p), 0.5)

    def test_slope_intercept_consistent(self):
        y, p, _ = synthetic()
        s, _ = sc.calib_slope_intercept(y, p)
        self.assertAlmostEqual(s, sc.calib_slope(y, p), places=8)

    def test_hosmer_lemeshow_p_in_range(self):
        y, p, _ = synthetic()
        stat, pv = sc.hosmer_lemeshow(y, p)
        self.assertGreaterEqual(stat, 0)
        self.assertTrue(0.0 <= pv <= 1.0)

    def test_oe_ratio(self):
        y, p, _ = synthetic()
        self.assertAlmostEqual(sc.oe_ratio(y, p), y.sum() / p.sum(), places=10)


class TestClinicalUtility(unittest.TestCase):
    def test_dca_none_arm_is_zero(self):
        y, p, _ = synthetic()
        tab = sc.dca(y, p, thresholds=np.array([0.1, 0.2, 0.3]))
        self.assertEqual(len(tab), 3)
        self.assertTrue((tab["nb_treat_none"] == 0).all())
        self.assertTrue((tab["threshold"].values == np.array([0.1, 0.2, 0.3])).all())

    def test_net_benefit_matches_dca(self):
        y, p, _ = synthetic()
        pt = 0.25
        self.assertAlmostEqual(sc.net_benefit(y, p, pt),
                               sc.dca(y, p, np.array([pt]))["nb_model"].iloc[0],
                               places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
