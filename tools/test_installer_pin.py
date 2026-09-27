# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only checks of the installer's pinned checkout and local-change guard."""

import os
import re
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

INSTALLER = (Path(__file__).resolve().parents[1] / "uwlab.sh").read_text()


class TestInstallerPin(unittest.TestCase):
    """Exercise checkout selection with a fake Git command and no package installation."""

    def _run_checkout(self, existing=False, dirty=False):
        start = INSTALLER.index('            repo_root="${UWLAB_PATH}/_isaaclab/IsaacLab"')
        checkout = re.search(r"^\s*git .* checkout .*FETCH_HEAD.*$", INSTALLER[start:], re.MULTILINE)
        end = start + checkout.end() if checkout else INSTALLER.index("            ${pip_command}", start)
        pin = re.search(r"^\s*ISAACLAB_COMMIT=.*$", INSTALLER[:start], re.MULTILINE)
        fragment = (pin.group(0) + "\n" if pin else "") + INSTALLER[start:end]
        self.assertNotIn("${pip_command}", fragment)
        with tempfile.TemporaryDirectory(prefix="uwlab checkout ") as directory:
            root = Path(directory)
            if existing:
                (root / "_isaaclab/IsaacLab/.git").mkdir(parents=True)
            trace = root / "git.log"
            script = "\n".join([
                "set -e",
                f"UWLAB_PATH={shlex.quote(directory)}",
                f"TRACE={shlex.quote(str(trace))}",
                "git() {",
                '  printf "%s\\t" "$@" >> "$TRACE"',
                '  printf "\\n" >> "$TRACE"',
                '  if [ "${3:-}" = status ]; then printf "%s" "${TEST_DIRTY:-}"; fi',
                "  return 0",
                "}",
                fragment,
            ])
            env = os.environ.copy()
            env.pop("UWLAB_ISAACLAB_COMMIT", None)
            env["TEST_DIRTY"] = " M modified.py" if dirty else ""
            result = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, check=False)
            calls = [line.rstrip("\t").split("\t") for line in trace.read_text().splitlines()] if trace.exists() else []
        return result, calls

    def _assert_pinned_fetch(self, calls):
        fetches = [call for call in calls if "fetch" in call]
        self.assertEqual(len(fetches), 1)
        self.assertRegex(fetches[0][-1], r"^[0-9a-f]{40}$")
        self.assertTrue(any(call[-3:] == ["-q", "--detach", "FETCH_HEAD"] for call in calls))

    def test_fresh_checkout_fetches_an_exact_revision(self):
        result, calls = self._run_checkout()
        self.assertEqual(result.returncode, 0, result.stderr)
        self._assert_pinned_fetch(calls)

    def test_clean_existing_checkout_is_pinned(self):
        result, calls = self._run_checkout(existing=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self._assert_pinned_fetch(calls)

    def test_dirty_existing_checkout_is_preserved(self):
        result, calls = self._run_checkout(existing=True, dirty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any("fetch" in call or "checkout" in call for call in calls))


if __name__ == "__main__":
    unittest.main()
