import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cli(*args, data=None):
    return subprocess.run([sys.executable, "-m", "pathprobe", *args], cwd=ROOT,
                          text=True, input=data, capture_output=True)


class CliTests(unittest.TestCase):
    def test_packaged_examples(self):
        self.assertEqual(json.loads(cli("predict", "examples/run.json", "search", "30").stdout)["predicted"], 80)
        report = json.loads(cli("fleet", "examples/cohort.jsonl", "search", "--duration", "30",
                                "--required", "2").stdout)
        self.assertEqual((report["impossible"], report["met"], report["limit"]), (1, 2, 40))

    def test_invalid_row_and_stdin(self):
        self.assertEqual(json.loads(cli("rank", "-", data='{"durations":{}, "edges":[]}').stdout)["tasks"], [])
        failure = cli("fleet", "-", "s", "--required", "1", data='{"durations":{"s":9},"edges":[]}')
        self.assertEqual(failure.returncode, 2)
        self.assertIn("line 1", failure.stderr)
