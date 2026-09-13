"""Deploy RunwayGuard Gradio demo to a free ZeroGPU Space.

Requires: hf auth login (personal account eligible for ZeroGPU).
Usage: python scripts/deploy_hf_space.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "deploy" / "hf_space_build"
USER = os.environ.get("HF_USER", "DarkKnight1217")
SPACE = f"{USER}/RunwayGuard-demo"


def main() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    shutil.copy(ROOT / "deploy/hf_space/README.md", STAGE / "README.md")
    shutil.copy(ROOT / "deploy/hf_space/requirements.txt", STAGE / "requirements.txt")
    shutil.copy(ROOT / "deploy/hf_space/app.py", STAGE / "app.py")
    shutil.copytree(ROOT / "app", STAGE / "app")
    shutil.copytree(ROOT / "config", STAGE / "config")
    (STAGE / "deploy").mkdir()
    shutil.copy(ROOT / "deploy/gradio_app.py", STAGE / "deploy/gradio_app.py")
    shutil.copytree(ROOT / "deploy/samples", STAGE / "deploy/samples")
    (STAGE / "deploy" / "__init__.py").write_text("", encoding="utf-8")
    (STAGE / "app" / "__init__.py").touch()

    api = HfApi()
    api.create_repo(
        repo_id=SPACE,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="zero-a10g",
        exist_ok=True,
        private=False,
    )
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=SPACE,
        repo_type="space",
        commit_message="Deploy RunwayGuard ZeroGPU demo",
    )
    for key, value in (
        ("HF_WEIGHTS_REPO", "DarkKnight1217/RunwayGuard-rtdetr-l"),
        ("RUNWAYGUARD_LLM", "0"),
    ):
        try:
            api.add_space_variable(SPACE, key, value)
        except Exception as exc:  # noqa: BLE001
            print(f"variable warning {key}: {exc}")
    # SSR-off crashes this Gradio pin on Spaces; leave default SSR and fix URLs in UI JS.
    try:
        api.delete_space_variable(SPACE, "GRADIO_SSR_MODE")
    except Exception as exc:  # noqa: BLE001
        print(f"variable warning GRADIO_SSR_MODE delete: {exc}")

    print(f"Space: https://huggingface.co/spaces/{SPACE}")
    print(f"App:   https://{USER.lower()}-runwayguard-demo.hf.space")


if __name__ == "__main__":
    main()
