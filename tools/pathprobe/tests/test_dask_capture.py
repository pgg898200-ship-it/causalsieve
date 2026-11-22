import importlib.util
import json
import unittest

from examples.dask_capture import capture, demo_workflow, workflow_record
from pathprobe import DurationIndex


def event(key, intervals, status="OK"):
    return {"key": key, "status": status, "startstops": [
        {"action": action, "start": start, "stop": stop}
        for action, start, stop in intervals
    ]}


class DaskTraceTests(unittest.TestCase):
    def test_real_dependencies_and_compute_time(self):
        record = workflow_record(
            {"other": [], ("scan", 2): ["other"]},
            [event("unrelated", [], "error"),
             event("other", [("compute", 1, 1.25)]),
             event(("scan", 2), [("transfer", 2, 3), ("compute", 3.25, 3.75)])],
        )
        self.assertEqual(record["durations"], {"other": 250000, "('scan', 2)": 500000})
        self.assertEqual(record["edges"], [["other", "('scan', 2)"]])

    def test_ambiguous_key_names(self):
        with self.assertRaisesRegex(ValueError, "collide"):
            workflow_record({"('scan', 2)": [], ("scan", 2): []}, [])

    def test_reject_incomplete_or_ambiguous_run(self):
        successful = event("target", [("compute", 1, 2)])
        for events in (
            [],
            [successful, successful],
            [event("target", [], "error")],
            [event("target", [("transfer", 1, 2)])],
            [event("target", [("compute", 2, 1)])],
            [event("target", [("compute", 1, 2), ("compute", 3, 4)])],
        ):
            with self.subTest(events=events), self.assertRaises(ValueError):
                workflow_record({"target": []}, events)

    @unittest.skipUnless(importlib.util.find_spec("distributed"), "requires optional dask[distributed]")
    def test_two_worker_end_to_end(self):
        from distributed import Client

        with Client(processes=False, n_workers=2, threads_per_worker=1,
                    dashboard_address=None) as client:
            result, record = capture(client, demo_workflow())
            workflow = demo_workflow()
            cached = client.compute(workflow)
            self.assertEqual(cached.result(), 2)
            with self.assertRaisesRegex(ValueError, "missing compute event"):
                capture(client, workflow)
        self.assertEqual(result, 2)
        self.assertEqual(record["time_unit"], "microseconds")
        self.assertEqual(record["edges"], [["lexical", "merge"], ["plan", "lexical"],
                                           ["plan", "prefix"], ["prefix", "merge"]])
        record = json.loads(json.dumps(record))
        d = record["durations"]
        index = DurationIndex(d, record["edges"])
        self.assertEqual(index.predict("lexical", 0), d["plan"] + d["prefix"] + d["merge"])


if __name__ == "__main__":
    unittest.main()
