"""Portable launcher safety check without allocating a GPU or launching training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LauncherTest(unittest.TestCase):
    def test_single_gpu_required_and_arguments_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy2(Path(__file__).with_name("launch.sh"), root / "launch.sh")
            commands = {
                "nvidia-smi": '#!/bin/sh\nprintf "%s\\n" "${MOCK_GPU:-GPU-test-device}"\n',
                "flock": '#!/bin/sh\n[ "$1" = -n ] || exit 3\nshift 2\nexec "$@"\n',
                "interpreter": '#!/usr/bin/env python3\nimport os,sys,json\nprint(json.dumps({"gpu":os.environ["CUDA_VISIBLE_DEVICES"],"args":sys.argv[1:]}))\n',
            }
            for name, body in commands.items():
                path = root / name
                path.write_text(body)
                path.chmod(0o755)
            env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                   "ATTENTION_GPU": "0", "ATTENTION_PYTHON": str(root / "interpreter")}
            command = ["bash", str(root / "launch.sh"), "--tag", "portable_check", "--steps", "2"]
            result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10, check=True)
            output = json.loads(result.stdout)
            self.assertEqual(output["gpu"], "GPU-test-device")
            self.assertEqual(output["args"], ["campaign.py", "--tag", "portable_check", "--steps", "2"])
            for invalid in ({"MOCK_GPU": "GPU-one\nGPU-two"}, {"ATTENTION_GPU": ""}):
                failed = subprocess.run(command, env={**env, **invalid}, capture_output=True, text=True, timeout=10)
                self.assertNotEqual(failed.returncode, 0)


if __name__ == "__main__":
    unittest.main()
