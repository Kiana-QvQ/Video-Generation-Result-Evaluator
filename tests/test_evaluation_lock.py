from __future__ import annotations

import threading
import time
import unittest

from evaluator.modules.core.evaluation_lock import serialized_evaluation


class EvaluationLockTests(unittest.TestCase):
    def test_serialized_evaluation_allows_only_one_active_call(self) -> None:
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        @serialized_evaluation
        def critical_section() -> None:
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1

        threads = [
            threading.Thread(target=critical_section)
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(maximum_active, 1)


if __name__ == "__main__":
    unittest.main()
