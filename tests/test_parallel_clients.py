"""Exercise real spawned processes without requiring CUDA or PyTorch."""

import json
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch
import warnings

import parallel_clients


_TEST_DEVICE = None


def initialize_test_worker(device_queue):
    global _TEST_DEVICE
    _TEST_DEVICE = device_queue.get()


def train_test_client(job):
    if job.get("error"):
        raise ValueError(job["error"])
    if job.get("oom_once") and job["client_id"] == 0:
        try:
            descriptor = os.open(job["oom_once"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
            raise RuntimeError("CUDA out of memory")
    started = time.monotonic()
    time.sleep(job.get("delay", 0))
    job["state"][0] += random.Random(job["seed"]).random()
    path = Path(job["directory"]) / f"client{job['client_id']}.json"
    path.write_text(json.dumps({
        "client_id": job["client_id"], "pid": os.getpid(), "device": _TEST_DEVICE,
        "start": started, "end": time.monotonic(), "state": job["state"],
    }))
    return job["client_id"], str(path)


class ParallelClientPoolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for name, replacement in [("_initialize_worker", initialize_test_worker),
                                  ("_train_client", train_test_client)]:
            mock = patch.object(parallel_clients, name, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def pool(self, devices):
        pool = parallel_clients.ParallelClientPool(devices)
        self.addCleanup(pool.close)
        return pool

    def jobs(self, count=5, **extra):
        self.state = [10.0]
        return [{"client_id": cid, "state": self.state, "seed": 42 + cid,
                 "directory": self.directory.name, **extra} for cid in range(count)]

    def records(self, paths):
        return [json.loads(Path(path).read_text()) for path in paths]

    def test_parallel_order_affinity_and_independent_inputs(self):
        pool = self.pool([0, 1])
        jobs = self.jobs(delay=0.25)
        records = self.records(pool.train(jobs[::-1]))
        self.assertEqual([r["client_id"] for r in records], list(range(5)))
        self.assertEqual(len({r["pid"] for r in records}), 2)
        self.assertTrue(any(
            a["pid"] != b["pid"] and max(a["start"], b["start"]) < min(a["end"], b["end"])
            for a in records for b in records))
        for pid in {r["pid"] for r in records}:
            self.assertEqual(len({r["device"] for r in records if r["pid"] == pid}), 1)
        self.assertEqual({r["device"] for r in records}, {0, 1})
        self.assertEqual(self.state, [10.0])
        for r in records:
            self.assertAlmostEqual(r["state"][0], 10 + random.Random(42 + r["client_id"]).random())

    def test_pool_reuse_and_restart(self):
        pool = self.pool([0])
        first = self.records(pool.train(self.jobs(count=1)))[0]
        second = self.records(pool.train(self.jobs(count=1)))[0]
        self.assertEqual(first["pid"], second["pid"])
        pool.close()
        pool.close()
        third = self.records(pool.train(self.jobs(count=1)))[0]
        self.assertNotEqual(first["pid"], third["pid"])

    def test_oom_retries_whole_batch_with_one_worker_per_gpu(self):
        pool = self.pool([0, 0])
        jobs = self.jobs(oom_once=str(Path(self.directory.name) / "oom-once"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = self.records(pool.train(jobs))
        self.assertEqual(pool.devices, [0])
        self.assertEqual(len(records), 5)
        self.assertEqual(len({r["pid"] for r in records}), 1)
        self.assertTrue(any("retrying this batch" in str(w.message) for w in caught))

    def test_worker_error_propagates_without_partial_results(self):
        pool = self.pool([0, 1])
        with self.assertRaisesRegex(ValueError, "invalid dataset"):
            pool.train(self.jobs(error="invalid dataset"))
        self.assertIsNone(pool._executor)

    def test_single_worker_oom_does_not_retry_forever(self):
        pool = self.pool([0])
        with self.assertRaisesRegex(ValueError, "CUDA out of memory"):
            pool.train(self.jobs(count=1, error="CUDA out of memory"))
        self.assertIsNone(pool._executor)


if __name__ == "__main__":
    unittest.main()
