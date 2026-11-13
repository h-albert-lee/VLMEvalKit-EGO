import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import io

from .api_runner import atomic, append, read_log, infer, make_payload, pilot_report
from egoown_cvpr.core import digest


class SafetyTests(unittest.TestCase):
    def fixture(self,root):
        raw=b'fake image bytes'; (root/'im.png').write_bytes(raw)
        p=dict(id='sample',condition='single',prompt='Choose A.',images=['im.png'],
               image_sha256=[digest(raw)],version='v',options=['MINE','OTHERS','SHARED','AMBIGUOUS'],gt_label='MINE')
        p['input_sha256']=digest(json.dumps(dict(prompt=p['prompt'],image_sha256=p['image_sha256'],version='v'),sort_keys=True).encode())
        m=dict(id='model',key='m',canonical_slug='model-version',params={'model':'model'},expected_provider='Provider')
        return m,p



    def test_tampered_image_never_sent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,p=self.fixture(root);(root/'im.png').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'image changed'): make_payload(root,m,p)

    def test_tampered_prompt_never_sent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,p=self.fixture(root);p['prompt']='changed'
            with self.assertRaisesRegex(ValueError,'input changed'): make_payload(root,m,p)

    def test_refuses_image_outside_root(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,p=self.fixture(root);p['images']=['../outside']
            with self.assertRaisesRegex(ValueError,'escapes'): make_payload(root,m,p)

    def test_response_and_usage_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,p=self.fixture(root)
            response=dict(id='generation',model='model',provider='Provider',usage={'cost':.001,'prompt_tokens':123},choices=[{'message':{'content':'A'},'finish_reason':'stop'}])
            with patch('repro.cvpr2027.api.api_runner.request',return_value=response) as req: result=infer(root,'SECRET',m,p,'a')
            self.assertEqual(result['raw_response'],response);self.assertEqual(result['pred_label'],'MINE')
            self.assertEqual(result['cost_usd'],.001);self.assertEqual(req.call_count,1)
            self.assertNotIn('SECRET',json.dumps(result))

    def test_unparsed_kept_without_retry_or_relaxing_parser(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,p=self.fixture(root)
            with patch('repro.cvpr2027.api.api_runner.request',return_value={'usage':{'cost':.01},'choices':[{'message':{'content':'maybe A but also B'}}]}):
                result=infer(root,'SECRET',m,p,'a')
            self.assertIsNone(result['pred_label']);self.assertIsNone(result['error']);self.assertFalse(result['retryable'])

    def test_timeout_not_blindly_retried(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,p=self.fixture(root)
            with patch('repro.cvpr2027.api.api_runner.request',side_effect=TimeoutError('SECRET')) as req: result=infer(root,'SECRET',m,p,'a')
            self.assertEqual(req.call_count,1);self.assertTrue(result['billing_uncertain']);self.assertFalse(result['retryable'])
            self.assertNotIn('SECRET',json.dumps(result))

    def test_rate_limit_retry_is_recorded(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);m,p=self.fixture(root)
            e=urllib.error.HTTPError('https://openrouter.ai',429,'limited',{},io.BytesIO(b'SECRET rejected'))
            with patch('repro.cvpr2027.api.api_runner.request',side_effect=e): result=infer(root,'SECRET',m,p,'a')
            self.assertTrue(result['retryable']);self.assertEqual(result['cost_usd'],0);self.assertNotIn('SECRET',json.dumps(result))

    def test_torn_log_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'log.jsonl';append(p,{'event':'start'})
            with p.open('a') as f:f.write('{"incomplete":')
            with self.assertRaises(json.JSONDecodeError):read_log(p)

    def test_pilot_accuracy_and_unparsed_not_selection_gate(self):
        c={'models':[{'key':'m','id':'model','canonical_slug':'model-version','expected_provider':'Provider'}],
           'conditions':['single','sparse'],'n':2}
        pp=[dict(model_key='m',condition=cond,id='s',raw='uncertain prose',pred_label=None,cost_usd=.01,
                 provider='Provider',response_model='model',finish_reason='stop') for cond in c['conditions']]
        report=pilot_report(c,{},pp,{'s'});self.assertTrue(report['passed'])
        pp[0]['finish_reason']='length';self.assertFalse(pilot_report(c,{},pp,{'s'})['passed'])
        pp[0]['finish_reason']='stop';pp[0]['provider']='Other';self.assertFalse(pilot_report(c,{},pp,{'s'})['passed'])


if __name__=='__main__':unittest.main()
