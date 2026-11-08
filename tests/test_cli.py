import unittest
import subprocess
import sys

class TestCLIEntryPoint(unittest.TestCase):

    def test_cli_help(self):
        """Test run.py help message."""
        res = subprocess.run([sys.executable, "run.py", "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("ODCNN Autopilot", res.stdout)
        self.assertIn("studio", res.stdout)
        self.assertIn("train", res.stdout)
        self.assertIn("infer", res.stdout)
        self.assertIn("synthesize", res.stdout)
        self.assertIn("export-onnx", res.stdout)
        self.assertIn("stream-sim", res.stdout)

    def test_cli_subcommand_helps(self):
        """Test run.py subcommand help flags."""
        for sub in ["studio", "train", "infer", "synthesize", "export-onnx", "stream-sim"]:
            res = subprocess.run([sys.executable, "run.py", sub, "--help"], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"Failed for subcommand {sub}")
            self.assertIn(f"usage: odcnn {sub}", res.stdout)

if __name__ == "__main__":
    unittest.main()
