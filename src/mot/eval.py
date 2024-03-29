import os
import motmetrics as mm
import numpy as np
import torch
from tqdm.auto import tqdm
import time
from torch.utils.data import DataLoader, Dataset
from mot.tracker.base import Tracker


def get_mot_accum(results: dict, seq: list[dict, int]) -> mm.MOTAccumulator:
    mot_accum = mm.MOTAccumulator(auto_id=True)

    for i in range(len(seq)):
        gt = seq.data[i]["gt"]
        gt_ids = []
        if gt:
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)

            gt_boxes = np.stack(gt_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            gt_boxes = np.stack(
                (
                    gt_boxes[:, 0],
                    gt_boxes[:, 1],
                    gt_boxes[:, 2] - gt_boxes[:, 0],
                    gt_boxes[:, 3] - gt_boxes[:, 1],
                ),
                axis=1,
            )
        else:
            gt_boxes = np.array([])

        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames = x1, y1, x2, y2, score  # skipcq: PY-W0069
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # x1, y1, x2, y2 --> x1, y1, width, height
            track_boxes = np.stack(
                (
                    track_boxes[:, 0],
                    track_boxes[:, 1],
                    track_boxes[:, 2] - track_boxes[:, 0],
                    track_boxes[:, 3] - track_boxes[:, 1],
                ),
                axis=1,
            )
        else:
            track_boxes = np.array([])

        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        mot_accum.update(gt_ids, track_ids, distance)

    return mot_accum


def evaluate_mot_accums(
    accums: mm.MOTAccumulator, names: list, generate_overall: bool = False
):
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,
    )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)
    return summary


def evaluate_obj_detect(model: torch.nn.Module, data_loader: DataLoader) -> None:
    model.eval()
    device = list(model.parameters())[0].device
    results = {}
    for imgs, targets in tqdm(data_loader):
        imgs = [img.to(device) for img in imgs]
        with torch.no_grad():
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target["image_id"].item()] = {
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
            }

    data_loader.dataset.print_eval(results)


def run_tracker(
    val_sequences: Dataset,
    tracker: Tracker,
    database: dict | None = None,
    output_dir: str | None = None,
) -> tuple:
    time_total = 0
    mot_accums = []
    results_seq = {}
    result_mot = None
    for seq in val_sequences:
        tracker.reset()
        now = time.time()

        print(f"Tracking: {seq}")

        if database is None:
            data_loader = DataLoader(seq, batch_size=1, shuffle=False)
            frame_seq_iterator = tqdm(data_loader)

        if database is not None:
            frame_seq_iterator = database[str(seq)]

        with torch.no_grad():
            for frame in frame_seq_iterator:
                tracker.step(frame)

        results = tracker.get_results()
        results_seq[str(seq)] = results

        if seq.no_gt:
            print("No GT evaluation data available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        time_total += time.time() - now

        print(f"Tracks found: {len(results)}")
        print(f"Runtime for {seq}: {time.time() - now:.1f} s.")

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            seq.write_results(results, os.path.join(output_dir))

    print(f"Runtime for all sequences: {time_total:.1f} s.")

    if mot_accums:
        result_mot = evaluate_mot_accums(
            mot_accums,
            [str(s) for s in val_sequences if not s.no_gt],
            generate_overall=True,
        )
    return result_mot, results_seq
