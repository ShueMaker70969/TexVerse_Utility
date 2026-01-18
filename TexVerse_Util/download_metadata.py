from pathlib import Path
from huggingface_hub import hf_hub_download

# Directory where THIS Python file lives
SCRIPT_DIR = Path(__file__).resolve().parent

hf_hub_download(
    repo_id="YiboZhang2001/TexVerse",
    repo_type="dataset",
    filename="metadata.json",
    local_dir=SCRIPT_DIR,
    local_dir_use_symlinks=False,
)