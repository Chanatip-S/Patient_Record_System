"""Test classifier error and low-score behavior without downloading a model."""
import unittest
from unittest.mock import patch
from prs import AICategorizer

class AITests(unittest.TestCase):
    def make_model(self, fake):
        with patch.dict('sys.modules', {'transformers': type('FakeTransformers', (), {'pipeline': staticmethod(lambda *a, **kw: fake)})()}):
            model = AICategorizer()
        return model

    def test_good_suggestion(self):
        model = self.make_model(lambda *a, **kw: {'labels':['Respiratory','General/Other','Digestive','Musculoskeletal'], 'scores':[.82,.1,.05,.03]})
        self.assertEqual(model.suggest('Synthetic cough and wheezing'), 'Respiratory')
        self.assertIsNone(model.last_error)

    def test_low_confidence(self):
        model = self.make_model(lambda *a, **kw: {'labels':['Respiratory','General/Other','Digestive','Musculoskeletal'], 'scores':[.30,.28,.22,.20]})
        self.assertIsNone(model.suggest('Synthetic vague note'))
        self.assertIn('below', model.last_error)

    def test_prediction_exception(self):
        def broken(*a, **kw): raise RuntimeError('example model failure')
        model = self.make_model(broken)
        self.assertIsNone(model.suggest('Synthetic note'))
        self.assertIn('example model failure', model.last_error)

if __name__ == '__main__': unittest.main()
