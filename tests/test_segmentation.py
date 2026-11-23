import unittest

import cv2
import numpy as np

from main import watershed_segment


class SegmentationTests(unittest.TestCase):
    def test_segment_synthetic_image(self):
        image = np.full((128, 128, 3), 255, np.uint8)
        cv2.circle(image, (44, 64), 20, (0, 0, 0), -1)
        cv2.circle(image, (88, 64), 20, (0, 0, 0), -1)
        marked, mask = watershed_segment(image)
        self.assertEqual(marked.shape, image.shape)
        self.assertEqual(mask.shape, image.shape[:2])
        self.assertGreater(int(cv2.countNonZero(mask)), 0)
        self.assertTrue(np.any(np.all(marked == (0, 0, 255), axis=2)))

    def test_reject_bad_input(self):
        with self.assertRaises(ValueError):
            watershed_segment(None)
        with self.assertRaises(ValueError):
            watershed_segment(np.zeros((10, 10), np.uint8))
        with self.assertRaises(ValueError):
            watershed_segment(np.zeros((10, 10, 3), np.uint8), foreground_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
