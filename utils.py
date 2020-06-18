from pathlib import Path


def ensure_dir(path):
    path_obj = Path(path)

    if not path_obj.exists():
        path_obj.mkdir(parents=True)
