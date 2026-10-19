
# All local paths are relative to the top-level clone directory.
import os as _os
import sys as _sys
from pathlib import Path as _Path
_code_root = _Path(__file__).resolve().parents[1]
_project_root = _code_root.parent.parent
_os.chdir(_project_root)
_sys.path[:0] = [str(_code_root), str(_code_root.parent / "lm-evaluation-harness")]

import argparse
from veomni.models import save_model_weights
from veomni.checkpoint import ckpt_to_state_dict
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert checkpoint to HuggingFace format")

    parser.add_argument(
        "--save_checkpoint_path",
        type=str,
        required=True,
        help="Path to the saved checkpoint directory"
    )

    parser.add_argument("--model_assets_dir", required=True,
                        help="Directory containing the saved config and tokenizer assets")
    args = parser.parse_args()
    from pathlib import Path
    import shutil
    assets = Path(args.model_assets_dir)
    if not (assets / "config.json").is_file():
        parser.error("model_assets_dir must contain config.json")

    hf_weights_path = os.path.join(args.save_checkpoint_path, "hf_ckpt")
    model_state_dict = ckpt_to_state_dict(
        save_checkpoint_path=args.save_checkpoint_path,
        ckpt_manager='dcp',
    )
    save_model_weights(hf_weights_path, model_state_dict)

    for asset in assets.iterdir():
        if asset.is_file() and asset.suffix in {".json", ".txt", ".model"}:
            shutil.copy2(asset, Path(hf_weights_path) / asset.name)
