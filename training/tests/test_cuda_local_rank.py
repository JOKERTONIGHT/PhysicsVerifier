from __future__ import annotations

import unittest

from openrlhf.trainer.ray.utils import local_rank_from_ray_gpu_ids


class CudaLocalRankTests(unittest.TestCase):
    def test_physical_ids_under_shifted_cvd(self) -> None:
        self.assertEqual(local_rank_from_ray_gpu_ids([1], "1,2,3"), "0")
        self.assertEqual(local_rank_from_ray_gpu_ids([2], "1,2,3"), "1")
        self.assertEqual(local_rank_from_ray_gpu_ids([3], "1,2,3"), "2")

    def test_already_remapped_ids(self) -> None:
        self.assertEqual(local_rank_from_ray_gpu_ids([0], "1,2,3"), "0")
        self.assertEqual(local_rank_from_ray_gpu_ids([2], "0,1,2"), "2")

    def test_no_cvd_keeps_ray_id(self) -> None:
        self.assertEqual(local_rank_from_ray_gpu_ids([3], ""), "3")
        self.assertEqual(local_rank_from_ray_gpu_ids([], "1,2,3"), "0")
