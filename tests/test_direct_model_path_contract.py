#!/usr/bin/env python3
"""Regression contract for serving existing host checkpoints without copying weights."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DirectModelPathContract(unittest.TestCase):
    def test_prepare_checks_both_hosts_and_skips_target_download(self):
        source = (ROOT / "scripts/prepare.sh").read_text()

        self.assertIn('[[ -n "$MODEL_PATH" ]] || models+=("$MODEL_ID")', source)
        self.assertIn("find -L . -path './.cache' -prune", source)
        self.assertIn('[[ "$worker_manifest" == "$model_manifest" ]] || die', source)
        self.assertIn('MODEL_INFO_ARGS=(-v "$MODEL_PATH:$MODEL_CONTAINER_PATH:ro")', source)
        self.assertIn('info "$MODEL_INFO_PATH"', source)

    def test_both_ranks_mount_the_same_checkpoint_read_only(self):
        source = (ROOT / "start.sh").read_text()

        self.assertIn('MODEL_ARG="$MODEL_CONTAINER_PATH"', source)
        self.assertIn('MODEL_MOUNT=(-v "$MODEL_PATH:$MODEL_CONTAINER_PATH:ro")', source)
        self.assertRegex(source, re.compile(r'-v "\$WORKER_MOUNT" "\$\{MODEL_MOUNT\[@\]\}"'))
        self.assertRegex(source, re.compile(r'-v "\$HF_CACHE":/root/\.cache/huggingface "\$\{MODEL_MOUNT\[@\]\}"'))

    def test_prepare_validates_existing_dflash_path_and_skips_hub_download(self):
        config = (ROOT / "scripts/config.sh").read_text()
        prepare = (ROOT / "scripts/prepare.sh").read_text()

        self.assertIn('DFLASH2_PATH="${DFLASH2_PATH:-}"', config)
        self.assertIn('DFLASH2_CONTAINER_PATH="${DFLASH2_CONTAINER_PATH:-/models/dflash2}"', config)
        self.assertIn('[[ "$DRAFTER" != dflash2 || -n "$DFLASH2_PATH" ]] || models+=("$DFLASH2_ID")', prepare)
        self.assertIn('worker_dflash_manifest=$(worker "cd \'$DFLASH2_PATH\' && find -L . -path \'./.cache\' -prune -o -type f -printf \'%P %s\\\\n\' | sort")', prepare)
        self.assertIn('[[ "$worker_dflash_manifest" == "$dflash_manifest" ]] || die', prepare)
        self.assertIn('dflash2_path=$DFLASH2_PATH', config)

    def test_both_ranks_mount_existing_dflash_path_read_only(self):
        source = (ROOT / "start.sh").read_text()

        self.assertIn('DRAFTER_ARG="$DFLASH2_CONTAINER_PATH"', source)
        self.assertIn('DFLASH2_MOUNT=(-v "$DFLASH2_PATH:$DFLASH2_CONTAINER_PATH:ro")', source)
        self.assertRegex(source, re.compile(r'"\$\{MODEL_MOUNT\[@\]\}" "\$\{DFLASH2_MOUNT\[@\]\}"'))


if __name__ == "__main__":
    unittest.main()
