"""Termination escalation must affect only the owned child process group."""
import os
import signal
import subprocess
import unittest
from unittest.mock import Mock, call, patch

with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test-device"}):
    import campaign


class CampaignTest(unittest.TestCase):
    def test_owned_child_termination_escalates_after_timeout(self):
        process = Mock(pid=1234)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("owned child", 15), 0]
        with patch.object(campaign, "active", process), patch.object(campaign.os, "killpg") as kill:
            campaign.terminate_child()
        self.assertEqual(kill.call_args_list, [call(1234, signal.SIGTERM), call(1234, signal.SIGKILL)])
        self.assertEqual(process.wait.call_args_list, [call(timeout=15), call()])


if __name__ == "__main__":
    unittest.main()
