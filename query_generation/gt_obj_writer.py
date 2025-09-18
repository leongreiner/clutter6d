from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, DefaultDict
from collections import defaultdict

def _split_dir(output_dir: str, dataset_name: str, split: str) -> Path:
    return Path(output_dir) / dataset_name / split

def _list_scene_dirs(split_dir: Path) -> List[Path]:
    if not split_dir.is_dir():
        return []
    dirs = [p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 6]
    return sorted(dirs, key=lambda p: int(p.name))

def _max_written_frame(scene_dir: Path) -> int:
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

def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)

def _write_json(path: Path, data: dict) -> None:
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
    gt_path = scene_dir / "scene_gt.json"
    if not gt_path.exists():
        print(f"[gt_obj_writer] skip, missing {gt_path}")
        return

    scene_gt = _load_json(gt_path)
    obj_path = scene_dir / "scene_gt_obj.json"
    scene_gt_obj = _load_json(obj_path)

    for lid in local_ids:
        key = str(lid)
        if key in scene_gt:
            # overwrite only this key with augmented entries, preserve other keys
            scene_gt_obj[key] = _augment_entries(scene_gt[key], obj_sizes)
        else:
            # keep consistent structure even if gt is empty for this frame
            scene_gt_obj.setdefault(key, [])

    _write_json(obj_path, scene_gt_obj)
    print(f"[gt_obj_writer] updated {obj_path} for {len(local_ids)} frame(s)")

def _plan_last_n_frames(split_dir: Path,
                        n: int,
                        frames_per_chunk: int) -> DefaultDict[int, List[int]]:
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
    plan: DefaultDict[int, List[int]] = defaultdict(list)
    for scene_dir in _list_scene_dirs(split_dir):
        gt_path = scene_dir / "scene_gt.json"
        obj_path = scene_dir / "scene_gt_obj.json"
        if not gt_path.exists():
            continue
        gt = _load_json(gt_path)
        obj = _load_json(obj_path)
        # add any keys from gt, we will overwrite just those keys
        need = []
        for k in gt.keys():
            # always update to keep sizes consistent with any changes
            need.append(int(k))
        if need:
            plan[int(scene_dir.name)] = sorted(need)
    return plan

def write_scene_gt_obj(output_dir: str,
                       dataset_name: str,
                       object_sizes: Dict[int, float],
                       num_new_frames: int | None = None,
                       split: str = "train_pbr",
                       frames_per_chunk: int = 1000,
                       mode: str = "auto") -> None:
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

    for chunk_id, local_ids in sorted(plan.items()):
        scene_dir = split_dir / f"{chunk_id:06d}"
        if not scene_dir.exists():
            continue
        _update_chunk_for_local_ids(scene_dir, sorted(set(local_ids)), object_sizes)
