"""Optional data-version override; use the existing official handlers and scorer."""
import os
from pathlib import Path


def configure():
    # Import before handlers/checkers bind these constants.
    from bfcl_eval.constants import eval_config
    if os.getenv("BFCL_DATA_DIR"):
        data = Path(os.environ["BFCL_DATA_DIR"]).resolve()
        if not data.is_dir():
            raise ValueError(f"Missing BFCL data directory: {data}")
        eval_config.PROMPT_PATH = data
        if (data / "multi_turn_func_doc").is_dir():
            eval_config.MULTI_TURN_FUNC_DOC_PATH = data / "multi_turn_func_doc"
        if (data / "possible_answer").is_dir():
            eval_config.POSSIBLE_ANSWER_PATH = data / "possible_answer"


if __name__ == "__main__":
    configure()
    from bfcl_eval.__main__ import cli
    cli()
