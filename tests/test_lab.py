from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

from terratorch_lab.data import BANDS, MEAN, NODATA, STD, Scene, load_scene, model_input, ndvi, synthetic_scene
from terratorch_lab.encoder import patch_distances


class LabTests(TestCase):
    def test_model_input_preserves_time_and_band_order(self):
        scene = synthetic_scene()
        normalized = model_input(scene)

        self.assertEqual(normalized.shape, (1, 6, 2, 64, 64))
        np.testing.assert_array_equal(normalized[0, :, :, 0, 0], 0)
        np.testing.assert_allclose(normalized[0, :, 1, 30, 36], (scene.pixels[1, :, 30, 36] - MEAN) / STD)
        np.testing.assert_array_equal(scene.pixels, synthetic_scene().pixels)

    def test_ndvi_uses_red_and_near_infrared_and_excludes_missing(self):
        pixels = np.zeros((1, 6, 1, 3))
        pixels[0, 2, 0] = [100, NODATA, 100]
        pixels[0, 3, 0] = [300, 300, -100]

        np.testing.assert_allclose(ndvi(pixels), [[[0.5, np.nan, np.nan]]])

    def test_load_scene_rejects_mislabeled_band_order(self):
        scene = synthetic_scene()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "bands.npz"
            np.savez(path, bands=BANDS[::-1], pixels=scene.pixels, dates=[day.isoformat() for day in scene.dates])

            with self.assertRaisesRegex(ValueError, "Expected bands"):
                load_scene(path)

    def test_scene_rejects_nondivisible_pixel_dimensions(self):
        with self.assertRaisesRegex(ValueError, "positive multiples of 16"):
            Scene(np.zeros((1, 6, 16, 17)), (date(2020, 1, 1),), "test")

    def test_cosine_comparison_excludes_patches_with_no_valid_pixels(self):
        pixels = np.ones((2, 6, 16, 32), dtype=np.float32)
        pixels[-1, :, :, 16:] = NODATA
        scene = Scene(pixels, (date(2020, 1, 1), date(2020, 2, 1)), "test")
        patches = np.array([[[[1.0, 0.0], [1.0, 0.0]]], [[[0.0, 1.0], [1.0, 0.0]]]])

        np.testing.assert_allclose(patch_distances(patches, scene), [[1, np.nan]])
