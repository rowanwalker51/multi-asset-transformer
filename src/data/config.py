import yaml
from pathlib import Path
import os


def resolve_paths(obj, base_path: Path):
    """
    Recursively resolves string paths in a nested dictionary or list relative to a base path.

    Parameters
    ----------
    obj : dict | list | str | any
        The object to resolve. Can be:
        - dict: recursively resolves all values
        - list: recursively resolves all elements
        - str: treats as a file/directory path
        - any other type: returned as-is
    base_path : Path
        The base directory to resolve relative paths against.

    Returns
    -------
    dict | list | Path | any
        Same structure as `obj`, but with all relative string paths converted to
        absolute `Path` objects. Non-string objects are returned unchanged.

    Notes
    -----
    - If a string path is already absolute, it will be resolved to an absolute Path.
    - This function preserves the structure of nested dicts and lists.
    """
    if isinstance(obj, dict):
        return {k: resolve_paths(v, base_path) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_paths(v, base_path) for v in obj]
    elif isinstance(obj, str):
        p = Path(obj)
        return (base_path / p).resolve() if not p.is_absolute() else p.resolve()
    else:
        return obj


def load_data_config(yaml_path: str = "../../configs/data.yaml"):
    """
    Loads a YAML configuration file and resolves all paths relative to the project root.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file. Defaults to "../../configs/data.yaml".
        This path is resolved relative to the current working directory.

    Returns
    -------
    dict
        The configuration dictionary with all paths converted to absolute Path objects.
        Nested paths under `paths` are resolved relative to `paths.data_root`, which
        itself is anchored to the project root.

    Notes
    -----
    - Assumes the project root is the parent directory of the `configs/` folder
      containing the YAML file.
    - Can be overridden with the environment variable `PROJECT_ROOT`.
    - All relative paths under `paths` (except `data_root`) are resolved relative
      to `data_root`.
    - Supports arbitrary nesting of dictionaries and lists under `paths`.
    """
    # Absolute path to YAML file
    yaml_path = Path(yaml_path).resolve()

    # Project root = parent of configs folder
    project_root = yaml_path.parent.parent

    # Optional: allow override via environment variable
    project_root = Path(os.getenv("PROJECT_ROOT", project_root)).resolve()

    # Load YAML
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Extract paths section
    paths_cfg = cfg.get("paths", {})

    # Resolve data_root relative to project root
    data_root = Path(paths_cfg["data_root"])
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    paths_cfg["data_root"] = data_root

    # Resolve all other nested paths under paths relative to data_root
    for key, val in paths_cfg.items():
        if key != "data_root":
            paths_cfg[key] = resolve_paths(val, data_root)

    cfg["paths"] = paths_cfg
    return cfg