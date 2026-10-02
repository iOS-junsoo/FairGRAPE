"""Mean-gap reporting, independent fairness denominators and paired valid sets."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('report',Path(__file__).resolve().parents[1]/'scripts/report_celeba_legacy_eo.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def row(name):
    d=dict(attribute=name,ACC=.75,BA=.75,activation_all_global_ref=.1,
           activation_positive=.2,activation_negative=.3,contribution_all=None)
    for g,counts in enumerate(((8,2,1,9),(2,8,3,7))):
        for key,value in zip(('TP','FN','FP','TN'),counts): d[f'Male{g}_{key}']=value
    refresh(d)
    return d


def refresh(d):
    for g in (0,1):
        tp,fn,fp,tn=[d[f'Male{g}_{k}'] for k in ('TP','FN','FP','TN')]
        d.update({f'Male{g}_ACC':report.divide(tp+tn,tp+fn+fp+tn),
                  f'Male{g}_TPR':report.divide(tp,tp+fn),f'Male{g}_FPR':report.divide(fp,fp+tn),
                  f'Male{g}_PPR':report.divide(tp+fp,tp+fn+fp+tn)})
    m=report.attribute_metrics(d)
    d.update(EqualizedOdds=.6,legacy_EO_mean_gap=m['EO'],
             EqualOpportunity=m['EqualOpportunity'],DemographicParity=m['DemographicParity'])


class ReportTests(unittest.TestCase):
    def test_original_mean_gap_not_larger_gap(self):
        m=report.attribute_metrics(row('example'))
        self.assertAlmostEqual(m['EO'],.4)
        self.assertAlmostEqual(m['EqualOpportunity'],.6)
        self.assertAlmostEqual(m['DemographicParity'],.2)

    def test_empty_cells_and_common_valid_sets_are_metric_specific(self):
        attrs=[row(f'Attr{i}') for i in range(39)]
        a=dict(method='dense',round=0,stage='dense',attributes=attrs,global_activation_gap=.1,seconds=1.)
        b=copy.deepcopy(a)
        b.update(method='A_activation',round=1,stage='pre')
        b['attributes'][0]['Male0_TP']=b['attributes'][0]['Male0_FN']=0
        b['attributes'][1]['Male0_FP']=b['attributes'][1]['Male0_TN']=0
        refresh(b['attributes'][0]); refresh(b['attributes'][1])
        raw=dict(status='running',metadata={},rounds=[],evaluations=[a,b])
        before=copy.deepcopy(raw)
        result=report.transform(raw)
        self.assertEqual(raw,before)
        for ev in result['evaluations']:
            for key,n in [('EO',37),('EqualOpportunity',38),('DemographicParity',39)]:
                self.assertEqual(ev['macro'][key]['n_attributes'],n)
            self.assertNotIn('EqualizedOdds',ev['macro'])
            self.assertNotIn('legacy_EO_mean_gap',ev['macro'])
        self.assertIsNone(result['evaluations'][1]['attributes'][0]['EO'])
        self.assertIsNone(result['evaluations'][1]['attributes'][0]['EqualOpportunity'])
        self.assertIsNotNone(result['evaluations'][1]['attributes'][0]['DemographicParity'])
        text=report.render(result)
        self.assertNotIn('EqualizedOdds',text)
        self.assertNotIn('legacy_EO_mean_gap',text)
        self.assertIn('EO',report.comparison(result['evaluations']))
        self.assertIn('EqualOpportunity',report.comparison(result['evaluations']))
        self.assertIn('DemographicParity',report.comparison(result['evaluations']))

    def test_final_comparison_uses_requested_22nd_round(self):
        attrs=[row(f'Attr{i}') for i in range(39)]
        dense=dict(method='dense',round=0,stage='dense',attributes=attrs,global_activation_gap=.1,seconds=1.)
        states=[dense]+[dict(copy.deepcopy(dense),method='A_activation',round=n,stage='post') for n in (3,21,22)]
        result=report.transform(dict(status='complete',metadata={'arguments':{'rounds':22}},rounds=[],evaluations=states))
        text=report.comparison(result['evaluations'],True,final_round=22)
        self.assertIn('r22 post',text)
        self.assertNotIn('r3 post',text)
        self.assertNotIn('r21 post',text)
        self.assertIn('FINAL 22-ROUND COMPARISON',report.render(result))

    def test_existing_mean_gap_mismatch_is_rejected(self):
        d=row('bad')
        d['legacy_EO_mean_gap']=.6
        attrs=[dict(d,attribute=f'Attr{i}') for i in range(39)]
        with self.assertRaises(AssertionError):
            report.transform(dict(evaluations=[dict(attributes=attrs)]))


if __name__=='__main__': unittest.main()
