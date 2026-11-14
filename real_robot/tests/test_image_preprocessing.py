import itertools
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from real_robot.image_preprocessing import SCHEMA,geometry,transform,validate_spec,order_quad
from real_robot.dom_dataset import export_sessions,validate_dataset
from real_robot.tests.test_dom_collection import make_session


def spec():
    return dict(schema=SCHEMA,source_size_wh=[640,480],target_size_wh=[384,384],
                padding='left_top',pad_value=0,front_serial='front',front_crop_xywh=None,
                front_quad_xy=[[140,70],[500,90],[590,420],[40,400]],front_rectified_size_wh=[400,300])


class ImagePreprocessingTests(unittest.TestCase):
    def test_selector_accepts_any_corner_order(self):
        corners=spec()['front_quad_xy']
        for perm in itertools.permutations(corners):
            self.assertEqual(order_quad(perm),corners)
        with self.assertRaises(ValueError):order_quad([[0,0],[100,0],[100,100],[75,25]])

    def test_four_corners_rectify_without_mirroring(self):
        s=spec();g=geometry(s,'front')
        p=cv2.perspectiveTransform(np.asarray([s['front_quad_xy']],np.float64),np.asarray(g['homography_raw_to_rectified']))[0]
        np.testing.assert_allclose(p,[[0,0],[399,0],[399,299],[0,299]],atol=1e-8)
        source=np.zeros((480,640,3),np.uint8)
        cv2.fillConvexPoly(source,np.array(s['front_quad_xy'],np.int32),(40,120,200))
        result=transform(source,s,'front')
        self.assertEqual(result.shape,(384,384,3))
        self.assertFalse(result[:96].any())
        np.testing.assert_array_equal(result[200,200],[40,120,200])
        # Other cameras retain their full field of view and use only resize/pad.
        self.assertIsNone(geometry(s,'wrist')['front_quad_xy'])

    def test_invalid_quad_and_nonraw_input_rejected(self):
        s=spec();s['front_quad_xy']=[[0,0],[639,479],[639,0],[0,479]]
        with self.assertRaises(ValueError):validate_spec(s)
        with self.assertRaises(ValueError):transform(np.zeros((360,480,3),np.uint8),spec(),'front')

    def test_export_uses_384_metadata_real_video_and_preserves_numeric_rows(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);session=root/'session'
            manifest,_=make_session(session)
            manifest['rig']['front_serial']='front'
            (session/'manifest.json').write_text(json.dumps(manifest))
            for p in session.rglob('*.npz'):
                with np.load(p) as raw:
                    rgb=cv2.resize(raw['bgr'],(640,480),interpolation=cv2.INTER_NEAREST)
                np.savez_compressed(p,bgr=rgb,depth_m=np.ones((480,640),np.float32))
            before=(session/'robot.jsonl').read_bytes()
            report=export_sessions([session],root/'dataset',image_preprocessing=spec())
            self.assertEqual(report['episodes'],1)
            result=validate_dataset(root/'dataset')
            self.assertEqual(result['episodes'],1)
            info=json.loads((root/'dataset/meta/info.json').read_text())
            for k,v in info['features'].items():
                if k.startswith('observation.images.'):self.assertEqual(v['shape'],[384,384,3])
            self.assertEqual((session/'robot.jsonl').read_bytes(),before)
            self.assertTrue((root/'dataset/meta/image_preprocessing.json').exists())
            rows=pq.read_table(root/'dataset/data/chunk-000/episode_000000.parquet').to_pylist()
            self.assertEqual(len(rows),info['total_frames'])
            # A metadata-only shape change cannot bypass the video decode checks.
            info['features']['observation.images.opst_cam']['shape']=[360,480,3]
            (root/'dataset/meta/info.json').write_text(json.dumps(info))
            with self.assertRaises(ValueError):validate_dataset(root/'dataset')


if __name__=='__main__':unittest.main()
