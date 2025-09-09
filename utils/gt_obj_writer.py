# gt_obj_writer.py
# Robust writer that augments BlenderProc BOP output with per-object sizes.
# Keeps previous data, survives restarts, mirrors chunking logic (default 1000 frames per scene).

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

# ---------------------------- internals ----------------------------

def _split_dir(output_dir: str, dataset_name: str, split: str) -> Path:
    return Path(output_dir) / dataset_name / split

def _list_scene_dirs(split_dir: Path) -> List[Path]:
    """Return sorted list of 6-digit scene folders in the split directory."""
    if not split_dir.is_dir():
        return []
    dirs = [p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 6]
    return sorted(dirs, key=lambda p: int(p.name))

def _max_written_frame(scene_dir: Path) -> int:
    """
    Read scene_gt.json and return the maximum frame id in this scene,
    or -1 if the file is missing or empty.
    """
    gt_path = scene_dir / "scene_gt.json"
    if not gt_path.exists():
        return -1
    try:
        with gt_path.open("r") as f:
            gt = json.load(f)
    except Exception:
        return -1
    if not gt:
        return -1
    try:
        return max(int(k) for k in gt.keys())
    except ValueError:
        return -1

def _safe_load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)

def _safe_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)

def _augment_entries(entries: List[dict], obj_sizes: Dict[int, float]) -> List[dict]:
    out = []
    for e in entries:
        obj_id = int(e["obj_id"])
        size = float(obj_sizes.get(obj_id, 1.0))
        ne = dict(e)
        ne["obj_size"] = size
        out.append(ne)
    return out

def _update_chunk_for_local_ids(scene_dir: Path,
                                local_ids: List[int],
                                obj_sizes: Dict[int, float]) -> None:
    """Merge-add obj_size for the specified local frame ids in a single scene folder."""
    gt_path = scene_dir / "scene_gt.json"
    if not gt_path.exists():
        print(f"[gt_obj_writer] skip, missing {gt_path}")
        return

    scene_gt = _safe_load_json(gt_path)
    obj_path = scene_dir / "scene_gt_obj.json"
    scene_gt_obj = _safe_load_json(obj_path)

    for lid in local_ids:
        key = str(lid)
        if key in scene_gt:
            # overwrite only this key with augmented entries, preserve other keys
            scene_gt_obj[key] = _augment_entries(scene_gt[key], obj_sizes)
        else:
            # keep consistent structure even if gt is empty for this frame
            scene_gt_obj.setdefault(key, [])

    _safe_write_json(obj_path, scene_gt_obj)
    print(f"[gt_obj_writer] updated {obj_path} for {len(local_ids)} frame(s)")

def _plan_last_n_frames(split_dir: Path,
                        n: int,
                        frames_per_chunk: int) -> DefaultDict[int, List[int]]:
    """
    Plan which chunk and local ids correspond to the *last* n frames currently on disk.
    Uses the highest existing frame in the last chunk, then walks backward across chunk boundaries.
    Returns {chunk_id: [local_ids,...]} with chunk_id as int.
    """
    plan: DefaultDict[int, List[int]] = defaultdict(list)
    if n <= 0:
        return plan

    scene_dirs = _list_scene_dirs(split_dir)
    if not scene_dirs:
        return plan

    last_chunk_id = int(scene_dirs[-1].name)
    last_local_max = _max_written_frame(scene_dirs[-1])
    if last_local_max < 0:
        # last chunk empty, walk back to find a non-empty one
        idx = len(scene_dirs) - 2
        while idx >= 0 and last_local_max < 0:
            last_chunk_id = int(scene_dirs[idx].name)
            last_local_max = _max_written_frame(scene_dirs[idx])
            idx -= 1
        if last_local_max < 0:
            return plan  # nothing on disk

    # global end index (inclusive)
    global_end = last_chunk_id * frames_per_chunk + last_local_max
    global_start = max(0, global_end - n + 1)

    for g in range(global_start, global_end + 1):
        chunk = g // frames_per_chunk
        local = g % frames_per_chunk
        plan[chunk].append(local)

    return plan

def _resync_plan(split_dir: Path) -> DefaultDict[int, List[int]]:
    """
    Full resync plan, computes for each chunk the set of frames present in scene_gt.json
    that are missing or need updating in scene_gt_obj.json.
    """
    plan: DefaultDict[int, List[int]] = defaultdict(list)
    for scene_dir in _list_scene_dirs(split_dir):
        gt_path = scene_dir / "scene_gt.json"
        obj_path = scene_dir / "scene_gt_obj.json"
        if not gt_path.exists():
            continue
        gt = _safe_load_json(gt_path)
        obj = _safe_load_json(obj_path)
        # add any keys from gt, we will overwrite just those keys
        need = []
        for k in gt.keys():
            # always update to keep sizes consistent with any changes
            need.append(int(k))
        if need:
            plan[int(scene_dir.name)] = sorted(need)
    return plan

# ---------------------------- public API ----------------------------

def write_scene_gt_obj(output_dir: str,
                       dataset_name: str,
                       object_sizes: Dict[int, float],
                       num_new_frames: int | None = None,
                       split: str = "train_pbr",
                       frames_per_chunk: int = 1000,
                       mode: str = "auto") -> None:
    """
    Augment BOP annotations with per-object sizes, robust to restarts.
    Writes or extends <split>/<chunk>/scene_gt_obj.json without deleting previous frames.

    Parameters
    ----------
    output_dir : str
        Root output directory passed to BlenderProc's write_bop.
    dataset_name : str
        BOP dataset name passed as 'dataset' to write_bop.
    object_sizes : Dict[int, float]
        Mapping from BOP obj_id -> size factor (float).
    num_new_frames : Optional[int]
        If provided, only the last N frames currently on disk will be augmented,
        this is the fast common path when you call this right after a write_bop with N frames.
        If None, a full resync is performed and all frames found in scene_gt.json are updated.
    split : str
        BOP split folder, default "train_pbr".
    frames_per_chunk : int
        Must match the value used in write_bop, default 1000.
    mode : str
        "auto" chooses tail update if num_new_frames is given, otherwise resync,
        "tail" forces last-N update,
        "resync" forces updating all frames present in scene_gt.json.

    Usage
    -----
    # after rendering and write_bop(...)
    # num_frames = len(data["colors"])
    # write_scene_gt_obj(cfg['dataset']['output_dir'], cfg['dataset']['name'], object_sizes, num_new_frames=num_frames)

    Notes
    -----
    This mirrors BlenderProc's chunking, frames are stored in scene folders like 000000, 000001,
    each containing up to frames_per_chunk frames, and BlenderProc supports appending to existing output,
    so multiple runs aggregate naturally in the same split.
    """
    split_dir = _split_dir(output_dir, dataset_name, split)

    if mode not in {"auto", "tail", "resync"}:
        raise ValueError("mode must be 'auto', 'tail', or 'resync'")

    if mode == "tail" or (mode == "auto" and num_new_frames is not None):
        n = int(num_new_frames or 0)
        plan = _plan_last_n_frames(split_dir, n, frames_per_chunk)
    else:
        plan = _resync_plan(split_dir)

    if not plan:
        print("[gt_obj_writer] nothing to update, plan is empty")
        return

    # execute plan
    for chunk_id, local_ids in sorted(plan.items()):
        scene_dir = split_dir / f"{chunk_id:06d}"
        if not scene_dir.exists():
            continue
        _update_chunk_for_local_ids(scene_dir, sorted(set(local_ids)), object_sizes)
