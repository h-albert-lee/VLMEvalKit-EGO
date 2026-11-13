import tempfile, unittest, json
from pathlib import Path
from PIL import Image
from egoown_cvpr.core import build_input,check_box,digest,parse_answer,metrics,LABELS,load_bundle

class InputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'frames').mkdir()
        p=self.root/'frames/a.png';Image.new('RGB',(100,100),'white').save(p)
        self.row={'id':'test','label':'MINE','target_name':'cup','source':'SECRET_SOURCE','taxonomy':'SECRET_TAXONOMY',
                  'frames':[{'tag':tag,'path':'frames/a.png','sha256':digest(p.read_bytes()),'bbox':dict(x_min=.2,y_min=.3,x_max=.6,y_max=.8)} for tag in ['t-2','t-1','t']]}
    def tearDown(self):self.tmp.cleanup()
    def test_overlay_and_ablations(self):
        a=build_input(self.root,self.row);im=Image.open(a['images'][-1])
        self.assertEqual(im.getpixel((20,30)),(255,0,0));self.assertEqual(im.getpixel((40,50)),(255,255,255))
        self.assertEqual(Image.open(self.root/'frames/a.png').getpixel((20,30)),(255,255,255))
        self.assertEqual(len(build_input(self.root,self.row,'single')['images']),1)
        self.assertEqual(len(set(build_input(self.root,self.row,'repeat')['images'])),1)
        with Image.open(build_input(self.root,self.row,'crop')['images'][0]) as cropped:
            self.assertEqual(cropped.size,(41,51))
        self.assertNotIn('SECRET_SOURCE',a['prompt']);self.assertNotIn('SECRET_TAXONOMY',a['prompt'])
        self.assertNotIn('cup',build_input(self.root,self.row,'geometry_only')['prompt'])
    def test_cache_depends_on_box_and_seed(self):
        a=build_input(self.root,self.row)
        self.row['frames'][0]['bbox']['x_min']=.1
        b=build_input(self.root,self.row)
        self.assertNotEqual(a['images'][0],b['images'][0]);self.assertNotEqual(a['input_sha256'],b['input_sha256'])
    def test_gt_does_not_change_input(self):
        a=build_input(self.root,self.row)
        self.row['label']='SHARED';self.row['review_note']='secret adjudication evidence'
        b=build_input(self.root,self.row)
        self.assertEqual(a,b)
    def test_invalid_box_and_frame_fail(self):
        with self.assertRaises(ValueError):check_box(dict(x_min=-.1,y_min=0,x_max=1,y_max=1))
        (self.root/'frames/a.png').write_bytes(b'corrupt')
        with self.assertRaises(ValueError):build_input(self.root,self.row)
    def test_parser_and_denominator(self):
        self.assertIsNone(parse_answer('A cup could be MINE or SHARED.',LABELS))
        self.assertEqual(parse_answer('Analysis\nAnswer: B',LABELS),'OTHERS')
        m=metrics([{'id':'a','label':'MINE'},{'id':'b','label':'OTHERS'},{'id':'c','label':'SHARED'}],[{'id':'a','pred_label':'MINE'},{'id':'b','error':'OOM'}])
        self.assertEqual(m['accuracy'],1/3);self.assertEqual(m['confusion']['SHARED']['MISSING'],1)
    def test_complete_option_must_be_consistent_and_standalone(self):
        options=['SHARED','MINE','OTHERS','AMBIGUOUS']
        self.assertEqual(parse_answer('C. OTHERS',options),'OTHERS')
        self.assertEqual(parse_answer('B) MINE',options),'MINE')
        self.assertIsNone(parse_answer('C. MINE',options))
        self.assertIsNone(parse_answer('C. OTHERS\nHuman: Next question',options))
        self.assertIsNone(parse_answer('C. OTHERS or B. MINE',options))
        self.assertEqual(parse_answer('Evidence supporting the selected answer.\n\nC',options),'OTHERS')
        self.assertIsNone(parse_answer('A\n\nEvidence with a conflicting answer.\n\nC',options))
        self.assertIsNone(parse_answer('A\nHuman: A different question\n\nC',options))

if __name__=='__main__':unittest.main()
