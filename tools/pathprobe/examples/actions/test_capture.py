"""Offline tests. Run with python -m unittest discover -s examples/actions -p test_capture.py."""
from copy import deepcopy
import json
import unittest

from capture import capture, unique_object


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.repo = "example/ci"
        self.base = "/repos/example/ci/actions/runs/12/attempts/2"
        self.sha = "a" * 40
        self.meta = {"id": 12, "run_attempt": 2, "status": "completed", "conclusion": "success",
                     "repository": {"full_name": self.repo}, "head_sha": self.sha}
        self.manifest = {"root": [], "unit (py310)": ["root"], "unit (py314)": ["root"]}
        self.jobs = []
        for i, (name, start, end) in enumerate([("root", 0, 5), ("unit (py310)", 9, 15), ("unit (py314)", 5, 7)], 1):
            self.jobs.append({"run_id": 12, "run_attempt": 2, "status": "completed", "conclusion": "success",
                              "head_sha": self.sha, "name": name, "id": i,
                              "started_at": f"2026-01-01T00:00:{start:02}Z",
                              "completed_at": f"2026-01-01T00:00:{end:02}Z"})

    def fake(self, path):
        if path == self.base:
            return self.meta
        paths = [f"{self.base}/jobs?per_page=100&page={i}" for i in (1, 2)]
        i = paths.index(path)
        return {"total_count": len(self.jobs), "jobs": self.jobs[100*i:100*(i+1)]}

    def run_capture(self):
        return capture(self.repo, 12, 2, self.manifest, get=self.fake, deadline=14)

    def test_full_parallel_graph(self):
        out = self.run_capture()
        self.assertEqual(out["durations"], {"root": 5, "unit (py310)": 6, "unit (py314)": 2})
        self.assertEqual(out["edges"], [["root", "unit (py310)"], ["root", "unit (py314)"]])
        self.assertEqual(out["source"]["jobs"]["unit (py310)"]["id"], 2)
        self.assertEqual(out["deadline"], 14)

    def test_pagination(self):
        for i in range(4, 113):
            self.jobs.append(dict(self.jobs[0], name=f"job-{i}", id=i))
            self.manifest[f"job-{i}"] = []
        self.assertEqual(len(self.run_capture()["durations"]), 112)

    def test_missing_or_ambiguous_jobs(self):
        original = deepcopy(self.jobs)
        for key, value in (("name", "unknown"), ("name", "root"), ("id", 1), ("run_id", 11),
                           ("run_attempt", 1), ("head_sha", "b"*40), ("conclusion", "skipped"),
                           ("conclusion", "failure"), ("status", "in_progress"),
                           ("started_at", "2026-01-01T00:00:01Z"),
                           ("completed_at", "2026-01-01T00:00:08Z")):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.jobs = deepcopy(original)
                self.jobs[1][key] = value
                self.run_capture()
        self.jobs = original[:2]
        with self.assertRaisesRegex(ValueError, "cover every reported job"):
            self.run_capture()

    def test_explicit_skip_verified_and_recorded(self):
        self.manifest["push only"] = None
        self.jobs.append(dict(self.jobs[0], name="push only", id=4, conclusion="skipped"))
        out = self.run_capture()
        self.assertNotIn("push only", out["durations"])
        self.assertEqual(out["source"]["skipped_jobs"], {"push only": 4})
        self.jobs[-1]["conclusion"] = "success"
        with self.assertRaisesRegex(ValueError, "expected a skipped job"):
            self.run_capture()

    def test_bad_manifest(self):
        for manifest in ({}, {"root": None}, {"a": ["b"]}, {"a": ["b"], "b": None}, {"a": ["a"]},
                         {"a": [], "b": ["a", "a"]}, {"a": ["b"], "b": ["a"]}):
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                self.manifest = manifest
                self.run_capture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            json.loads('{"a":[],"a":[]}', object_pairs_hook=unique_object)

    def test_run_identity_and_conclusion(self):
        for changes in ({"repository": {"full_name": "other/ci"}}, {"run_attempt": 1},
                        {"head_sha": "invalid"}, {"conclusion": "cancelled"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                original = self.meta.copy()
                self.meta.update(changes)
                try:
                    self.run_capture()
                finally:
                    self.meta = original

    def test_inputs_before_network(self):
        for repo, run, attempt in [("https://private.test", 1, 1), ("owner/..", 1, 1),
                                   ("owner/repo", 0, 1), ("owner/repo", 1, 0)]:
            with self.subTest(repo=repo, run=run, attempt=attempt), self.assertRaises(ValueError):
                capture(repo, run, attempt, self.manifest, get=lambda _: self.fail("unexpected request"))


if __name__ == "__main__":
    unittest.main()
