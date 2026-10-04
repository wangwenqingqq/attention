"""Termination escalation must affect only the owned child process group."""
import os
import signal
import subprocess
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, call, patch

with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-test-device"}):
    import campaign


class CampaignTest(unittest.TestCase):
    def test_nondefault_seed_reaches_every_group_and_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            (root / "runs").mkdir()
            def fake_run(stage, command, log, *args, **kwargs):
                if stage == "official_basic":
                    log.write_text("valid/accuracy: 0.991\n")
                if stage.startswith("mqar_"):
                    self.assertEqual(command[command.index("--seed") + 1], "124")
                    output = Path(command[command.index("--output") + 1])
                    self.assertTrue(output.name.endswith("_seed124"))
                    output.mkdir()
                    (output / "result.json").write_text(json.dumps({
                        "initial_state_sha256": "matched", "best_validation": {"query_accuracy": 0.99}}))
            with patch.object(campaign, "ROOT", root), \
                 patch.object(campaign, "STATE", root / "runs" / "campaign_status.json"), \
                 patch.object(campaign, "run", side_effect=fake_run) as run, \
                 patch.object(campaign.subprocess, "check_output", return_value="mock GPU metadata"), \
                 patch.object(campaign.signal, "signal"), \
                 patch.object(campaign.sys, "argv", ["campaign.py", "--seed", "124", "--tag", "repeat"]):
                campaign.main()
            self.assertEqual(run.call_count, 7)
            self.assertEqual(json.loads((root / "runs" / "campaign_status.json").read_text())["stage"],
                             "screen_completed")

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
