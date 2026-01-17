from huggingface_hub import hf_hub_download

def download_first_n_models(pbr_list_path, n=5):
    repo_id = "YiboZhang2001/TexVerse-1K"
    repo_type = "dataset"

    # 1) Read only the first n lines (model IDs)
    model_ids = []
    with open(pbr_list_path, "r") as f:
        for _ in range(n):
            line = f.readline()
            if not line:
                break  # end of file
            model_ids.append(line.strip())

    print(f"Processing {len(model_ids)} model IDs...")

    # 2) Try downloading each
    for model_id in model_ids:
        print(f"\n➡️ Searching model: {model_id}")
        found = False

        for i in range(89):  # buckets 000-000 → 000-088
            bucket = f"000-{i:03d}"
            path = f"glbs/glbs_1k/{bucket}/{model_id}_1024.glb"

            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    filename=path,
                    local_dir="downloaded_models",
                    # you can omit local_dir_use_symlinks now
                )
                print(f"✅ Downloaded: {local_path}")
                found = True
                break  # stop searching buckets

            except Exception:
                pass  # try next bucket

        if not found:
            print(f"❌ Not found in TexVerse-1K: {model_id}")

if __name__ == "__main__":
    download_first_n_models(
        r"C:\Users\Shuma\Desktop\Synced_folder\texverse_utility\TexVerse\metadata\TexVerse_pbr_id_list.txt",
        n=6
    )