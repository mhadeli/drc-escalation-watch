"""Run with: .venv/bin/python -m unittest discover -s tests"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate as ev  # noqa: E402
import extract_llm as el  # noqa: E402
import forecast as fc  # noqa: E402

POST = {"id": "1", "author_id": "a", "created_at": "2026-10-21T12:00:00.000Z", "lang": "fr",
        "text": "Sud-Kivu : les FARDC reprennent le contrôle de Muhungu après des affrontements avec le FRN-M23"}


class ForecastTests(unittest.TestCase):
    def test_rules_tag_place_actor_and_control_change(self):
        row = fc.classify(POST)
        self.assertEqual(row["hotspots"], ["Walungu (Kaziba, Muhungu, Kamanyola)"])
        self.assertTrue(row["control_change"] and row["event"])
        self.assertIn("M23/AFC", row["actors"])
        self.assertIn("FRN", row["actors"])

    def test_history_is_flagged(self):
        old = {**POST, "text": "Le 20 octobre 1996, des éléments de l'AFDL ont attaqué le camp de Kamanyola"}
        self.assertTrue(fc.classify(old)["historical"])

    def test_probability_rises_with_signal_and_horizon(self):
        base = fc.probability(0.5, 1.0, 30)
        self.assertGreater(fc.probability(0.5, 3.0, 30), base)
        self.assertGreater(fc.probability(0.5, 1.0, 90), base)
        # the signal boost fades: its share of the 90-day number is smaller than of the 30-day one
        lift30 = fc.probability(0.5, 3.0, 30) / base
        lift90 = fc.probability(0.5, 3.0, 90) / fc.probability(0.5, 1.0, 90)
        self.assertLess(lift90, lift30)

    def test_resolve_hotspot(self):
        self.assertEqual(fc.resolve_hotspot("Kaziba"), "Walungu (Kaziba, Muhungu, Kamanyola)")
        self.assertEqual(fc.resolve_hotspot("Bunia"), "Irumu / Bunia")
        self.assertIsNone(fc.resolve_hotspot("Paris"))


class EvaluateTests(unittest.TestCase):
    def test_score_counts_set_and_boolean_errors(self):
        gold = [{"post_id": "1", "lang": "fr", "relevant": True, "historical": False, "event": True,
                 "control_change": True, "escalation": False, "deescalation": False,
                 "hotspots": {"Walungu (Kaziba, Muhungu, Kamanyola)", "Uvira (Ruzizi plain)"},
                 "actors": {"FARDC"}, "fatalities": None}]
        report = ev.score(gold, {"1": fc.classify(POST)})
        self.assertEqual(report["n"], 1)
        self.assertEqual(report["fields"]["hotspots"]["precision"], 1.0)
        self.assertEqual(report["fields"]["hotspots"]["recall"], 0.5)
        self.assertEqual(report["fields"]["event"]["f1"], 1.0)


def fake_client(stop_reason="end_turn", parsed=True):
    output = el.PostExtraction(relevant=True, historical=False, event_types=["control_change"], escalation=False,
                               deescalation=False, hotspots=["Walungu (Kaziba, Muhungu, Kamanyola)"], other_places=[],
                               actors=["FARDC"], fatalities=None, displaced=None, source_type="party_claim")
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=kwargs["model"], stop_reason=stop_reason, parsed_output=output if parsed else None,
                               usage=SimpleNamespace(input_tokens=900, output_tokens=80, cache_read_input_tokens=800))

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(parse=parse))), calls


class ExtractorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_cache, el.CACHE = el.CACHE, Path(self.tmp.name)

    def tearDown(self):
        el.CACHE = self.old_cache
        self.tmp.cleanup()

    def test_extract_caches_and_maps_to_rule_shape(self):
        client, calls = fake_client()
        first = el.extract_one(client, POST, "claude-opus-5-5")
        second = el.extract_one(client, POST, "claude-opus-5-5")
        self.assertEqual(len(calls), 1, "second call should come from the cache")
        self.assertEqual(calls[0]["fallbacks"], "default")
        self.assertEqual(first, second)
        row = el.to_tagged(POST, first)
        self.assertTrue(row["control_change"])
        self.assertEqual(set(row) >= set(fc.classify(POST)), True)

    def test_haiku_gets_no_fallbacks(self):
        client, calls = fake_client()
        el.extract_one(client, POST, "claude-haiku-5-5")
        self.assertNotIn("fallbacks", calls[0])

    def test_refusal_becomes_an_error_row(self):
        client, _ = fake_client(stop_reason="refusal", parsed=False)
        record = el.extract_one(client, POST, "claude-opus-5-5")
        self.assertEqual(record["error"], "refusal")
        self.assertFalse(el.to_tagged(POST, record)["on_topic"])


if __name__ == "__main__":
    unittest.main()
