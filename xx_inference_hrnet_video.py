"""
Inference Video — YOLOv8 + BoT-SORT + HRNet + PoseC3D
=======================================================
Hanya mode video. Menghitung waktu komputasi detail per tahap,
khususnya HRNet, untuk perbandingan dengan YOLO11-pose.

Jalankan:
    python inference_hrnet_video.py --input data/test_video/input.mp4
    python inference_hrnet_video.py --input input.mp4 --output hasil.mp4
"""

import os
import os.path as osp
import argparse
import copy as cp
import tempfile
import time
import gc
import json

import cv2
import mmcv
import mmengine
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
from boxmot import BoTSORT
from mmaction.apis import (inference_recognizer, init_recognizer, pose_inference)
from mmaction.registry import VISUALIZERS
from mmaction.utils import frame_extract
import moviepy.editor as mpy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURASI — sesuaikan path di sini
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_CONFIG     = '01_asset_tools/ciis_7.py'
SKELETON_CHECKPOINT = '02_work_dirs/ciis_7_gab2_2s/best_acc_top1_epoch_230.pth'
LABEL_MAP_FILE      = '03_dataset/ciis_label_map.txt'

YOLO_MODEL_PATH = '01_asset_tools/models/yolov8m.pt'
DET_SCORE_THR   = 0.3

POSE_CONFIG     = ('mmaction2/demo/demo_configs/'
                   'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py')
POSE_CHECKPOINT = ('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                   'hrnet_w32_coco_256x192-c78dce93_20200708.pth')

BOTSORT_REID    = Path('01_asset_tools/models/osnet_x0_25_msmt17.pt')

ACTION_SCORE_THR  = 0.75
PREDICT_STEPSIZE  = 4
OUTPUT_FPS        = 12
DEVICE            = 'cuda:0'
VIDEO_OUT_DEFAULT = '03_dataset/test_video/output_hrnet.mp4'

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────────────
FONTFACE  = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.7
THICKNESS = 2
LINETYPE  = 1
BAHAYA    = ['melempar', 'membidik senapan', 'membidik pistol',
             'memukul', 'menendang', 'menusuk']
 
def hex2color(h):
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))
 
PLATEBLUE = [hex2color(h) for h in '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'.split('-')]
 
def load_label_map(path):
    lines = open(path).readlines()
    return {int(x[0]): x[1].strip()
            for x in [l.strip().split(': ') for l in lines if l.strip()]}
 
def abbrev(name):
    while '(' in name:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name
 
def _cal_iou(b1, b2):
    xmin1,ymin1,xmax1,ymax1 = b1
    xmin2,ymin2,xmax2,ymax2 = b2
    s1    = max(0, xmax1-xmin1) * max(0, ymax1-ymin1)
    s2    = max(0, xmax2-xmin2) * max(0, ymax2-ymin2)
    xi    = max(0, min(xmax1,xmax2) - max(xmin1,xmin2))
    yi    = max(0, min(ymax1,ymax2) - max(ymin1,ymin2))
    inter = xi * yi
    union = s1 + s2 - inter
    return inter / union if union > 0 else 0
 
def expand_bbox(bbox, h, w, ratio=1.25):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1+x2)//2, (y1+y2)//2
    sq     = max(x2-x1, y2-y1) * ratio
    return (max(0, int(cx-sq/2)), max(0, int(cy-sq/2)),
            min(int(cx+sq/2), w), min(int(cy+sq/2), h))
 
def print_timing_summary(label, times_ms, num_frame=None):
    """Cetak statistik waktu komputasi satu tahap."""
    arr = np.array(times_ms)
    print(f'\n  [{label}]')
    print(f'    Total      : {arr.sum()/1000:.3f} s')
    if num_frame:
        print(f'    Per frame  : {arr.sum()/num_frame:.2f} ms/frame')
    print(f'    Mean       : {arr.mean():.2f} ms')
    print(f'    Std        : {arr.std():.2f} ms')
    print(f'    Min        : {arr.min():.2f} ms')
    print(f'    Max        : {arr.max():.2f} ms')
    print(f'    Median     : {np.median(arr):.2f} ms')
 
# ─────────────────────────────────────────────────────────────────────────────
#  YOLOv8 + BoT-SORT
# ─────────────────────────────────────────────────────────────────────────────
def run_detection_and_tracking(frame_paths, detector, det_score_thr,
                                reid_weights, track_device, half=True):
    """
    YOLOv8 + BoT-SORT untuk setiap frame.
    Returns:
        tracked_detections : list[list[dict]]  per frame → {track_id, bbox, score}
        human_detections   : list[np.ndarray]  per frame → [N,5] untuk pose_inference
        times_yolo_ms      : list[float]        waktu per frame (ms)
    """
    tracker = BoTSORT(
        model_weights=reid_weights,
        device=track_device,
        fp16=half,
        track_high_thresh=det_score_thr,
        track_low_thresh=0.1,
        new_track_thresh=det_score_thr,
        track_buffer=50,
        match_thresh=0.8,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=True,
    )
 
    tracked_detections = []
    human_detections   = []
    times_yolo_ms      = []
 
    print('Running YOLOv8 + BoT-SORT tracking...')
    prog_bar = mmengine.ProgressBar(len(frame_paths))
 
    for frame_path in frame_paths:
        frame_bgr = cv2.imread(frame_path)
 
        t_frame = time.perf_counter()
 
        yolo_res = detector(frame_bgr, classes=[0], verbose=False)
        boxes    = yolo_res[0].boxes.xyxy.cpu().numpy()
        scores   = yolo_res[0].boxes.conf.cpu().numpy()
 
        if len(boxes) > 0:
            dets = np.hstack([boxes, scores[:, None], np.zeros((len(boxes), 1))])
        else:
            dets = np.empty((0, 6))
 
        tracks = tracker.update(dets, frame_bgr)
 
        times_yolo_ms.append((time.perf_counter() - t_frame) * 1000)
 
        frame_tracks    = []
        frame_human_det = []
 
        if len(tracks) > 0:
            for t in tracks:
                x1, y1, x2, y2 = t[0], t[1], t[2], t[3]
                tid   = int(t[4])
                score = float(t[5])
                frame_tracks.append({'track_id': tid,
                                     'bbox': [x1, y1, x2, y2],
                                     'score': score})
                frame_human_det.append([x1, y1, x2, y2, score])
 
        tracked_detections.append(frame_tracks)
        det_arr = np.array(frame_human_det) if frame_human_det else np.zeros((0, 5))
        human_detections.append(det_arr)
 
        prog_bar.update()
 
    print(f'\nSelesai. Total frame: {len(frame_paths)}')
    return tracked_detections, human_detections, times_yolo_ms
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  HRNet pose estimation dengan timing per-frame
# ─────────────────────────────────────────────────────────────────────────────
def run_hrnet_with_timing(pose_config, pose_checkpoint,
                           frame_paths, human_detections, device):
    """
    Jalankan HRNet (pose_inference dari MMAction) dan ukur waktu per frame.
 
    Karena pose_inference MMAction memproses batch sekaligus, kita pecah
    menjadi per-frame agar bisa mengukur waktu komputasi individual.
 
    Returns:
        pose_results_raw : output asli pose_inference (per frame)
        pose_datasample  : datasample untuk visualisasi
        times_hrnet_ms   : list[float] waktu HRNet per frame (ms)
        times_hrnet_per_person_ms : list[float] waktu per orang (ms),
                                    berguna jika jumlah orang bervariasi
    """
    from mmpose.apis import inference_topdown, init_model as init_pose_model
 
    print('Inisialisasi model HRNet...')
    pose_model = init_pose_model(pose_config, pose_checkpoint, device=device)
 
    times_hrnet_ms            = []
    times_hrnet_per_person_ms = []   # waktu dibagi jumlah orang di frame itu
    pose_results_all          = []
    pose_datasample_all       = []
 
    print('Running HRNet per frame...')
    prog_bar = mmengine.ProgressBar(len(frame_paths))
 
    for frame_path, human_det in zip(frame_paths, human_detections):
        frame_bgr   = cv2.imread(frame_path)
        n_person    = len(human_det)
 
        if n_person == 0:
            # Tidak ada orang di frame ini
            times_hrnet_ms.append(0.0)
            times_hrnet_per_person_ms.append(0.0)
            pose_results_all.append([])
            pose_datasample_all.append(None)
            prog_bar.update()
            continue
 
        # inference_topdown menerima list of bbox sebagai numpy array shape (4,)
        # atau (1,4) — format dict menyebabkan KeyError: None di beberapa versi MMPose
        bboxes_np = human_det[:, :4]  # shape (N, 4), xyxy
 
        # Ukur waktu hanya HRNet (inference_topdown)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
 
        pose_res_raw = inference_topdown(pose_model, frame_bgr, bboxes_np)
 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
 
        times_hrnet_ms.append(elapsed_ms)
        times_hrnet_per_person_ms.append(elapsed_ms / n_person if n_person > 0 else 0.0)
 
        # Attach dataset_meta dari model ke setiap datasample
        # (diperlukan oleh visualizer — tidak di-attach otomatis saat pakai inference_topdown langsung)
        dataset_meta = getattr(pose_model, 'dataset_meta', {})
        for ds in pose_res_raw:
            if not hasattr(ds, 'dataset_meta') or ds.dataset_meta is None:
                ds.dataset_meta = dataset_meta
 
        pose_results_all.append(pose_res_raw)
        pose_datasample_all.append(pose_res_raw[0] if pose_res_raw else None)
 
        prog_bar.update()
 
    print(f'\nSelesai. Total frame: {len(frame_paths)}')
    return pose_results_all, pose_datasample_all, times_hrnet_ms, times_hrnet_per_person_ms
 
 
def format_pose_results(pose_results_raw):
    """Format ulang output HRNet (inference_topdown) ke dict konsisten."""
    pose_results = []
    for frame_poses in pose_results_raw:
        if not frame_poses:
            pose_results.append({
                'keypoints':       np.zeros((0, 17, 2)),
                'keypoint_scores': np.zeros((0, 17)),
                'bboxes':          np.zeros((0, 4))
            })
            continue
 
        kps_list    = []
        kpconf_list = []
        bboxes_list = []
 
        for p in frame_poses:
            # keypoints: bisa shape (1,17,2) atau (17,2)
            kp = p.pred_instances.keypoints
            if hasattr(kp, 'cpu'):
                kp = kp.cpu().numpy()
            kp = np.array(kp)
            if kp.ndim == 3:
                kp = kp[0]   # (1,17,2) -> (17,2)
            kps_list.append(kp)
 
            # keypoint_scores: bisa shape (1,17) atau (17,)
            kpc = p.pred_instances.keypoint_scores
            if hasattr(kpc, 'cpu'):
                kpc = kpc.cpu().numpy()
            kpc = np.array(kpc)
            if kpc.ndim == 2:
                kpc = kpc[0]   # (1,17) -> (17,)
            kpconf_list.append(kpc)
 
            # bboxes: bisa shape (1,4) atau (4,)
            bb = p.pred_instances.bboxes
            if hasattr(bb, 'cpu'):
                bb = bb.cpu().numpy()
            bb = np.array(bb).flatten()[:4]
            bboxes_list.append(bb)
 
        pose_results.append({
            'keypoints':       np.array(kps_list),
            'keypoint_scores': np.array(kpconf_list),
            'bboxes':          np.array(bboxes_list)
        })
    return pose_results
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  BUILD TRACK POSE INDEX
# ─────────────────────────────────────────────────────────────────────────────
def build_track_pose_index(tracked_detections, pose_results):
    track_pose_index = []
    for frame_tracks, frame_poses in zip(tracked_detections, pose_results):
        id_to_pose = {}
        if not frame_tracks or len(frame_poses.get('keypoints', [])) == 0:
            track_pose_index.append(id_to_pose)
            continue
        pose_bboxes = frame_poses['bboxes']
        for track in frame_tracks:
            tid = track['track_id']
            best_iou, best_pidx = -1, -1
            for pidx, pbbox in enumerate(pose_bboxes):
                iou = _cal_iou(track['bbox'], pbbox)
                if iou > best_iou:
                    best_iou, best_pidx = iou, pidx
            if best_pidx >= 0 and best_iou > 0.1:
                id_to_pose[tid] = best_pidx
        track_pose_index.append(id_to_pose)
    return track_pose_index
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE POSEC3D
# ─────────────────────────────────────────────────────────────────────────────
def run_posec3d_on_buffer(kp_buffer, track_buffer, track_pose_index_buf,
                           model_posec3d, label_map, h, w,
                           action_score_thr, clip_len):
    center_idx    = clip_len // 2
    center_tracks = track_buffer[center_idx]
    results       = {}
 
    for track in center_tracks:
        tid            = track['track_id']
        keypoint       = np.zeros((1, clip_len, 17, 2))
        keypoint_score = np.zeros((1, clip_len, 17))
 
        for j in range(clip_len):
            pose_idx = track_pose_index_buf[j].get(tid, None)
            if pose_idx is not None:
                keypoint[0, j]       = kp_buffer[j]['keypoints'][pose_idx]
                keypoint_score[0, j] = kp_buffer[j]['keypoint_scores'][pose_idx]
 
        fake_anno = dict(
            frame_dir='', label=-1,
            img_shape=(h, w), original_shape=(h, w),
            num_clips=1, total_frames=clip_len,
            keypoint=keypoint, keypoint_score=keypoint_score
        )
 
        output  = inference_recognizer(model_posec3d, fake_anno)
        score   = output.pred_score.tolist()
        best_k  = int(np.argmax(score))
        best_sc = score[best_k]
 
        if best_sc > action_score_thr:
            results[tid] = (label_map[best_k], best_sc)
        else:
            results[tid] = (None, 0.0)
 
    return results
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISASI
# ─────────────────────────────────────────────────────────────────────────────
def draw_label_overlay(frame, trk, label_str, score):
    x1, y1, x2, y2 = [int(v) for v in trk['bbox']]
    tid = trk['track_id']
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,200), 2)
    cv2.putText(frame, f'ID:{tid}',
                (x1, max(0,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,200), 1)
    if label_str:
        text   = f'{abbrev(label_str)}: {score*100:.1f}%'
        tw, th = cv2.getTextSize(text, FONTFACE, FONTSCALE, THICKNESS)[0]
        loc    = (x1, y1+22)
        cv2.rectangle(frame,
                      (loc[0],      loc[1]-th-2),
                      (loc[0]+tw+4, loc[1]+2),
                      PLATEBLUE[1], -1)
        fc = (255,0,0) if label_str in BAHAYA else (255,255,255)
        cv2.putText(frame, text, loc, FONTFACE, FONTSCALE, fc, THICKNESS, LINETYPE)
    return frame
 
 
def visualize_video(frames, stdet_results, pose_datasample,
                    tracked_detections, output_timestamps, all_timestamps):
    frames_ = cp.deepcopy(frames)
    frames_ = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
    anchor_set = set(all_timestamps.tolist())
    h, w, _    = frames[0].shape
 
    if pose_datasample and any(d is not None for d in pose_datasample):
        pose_cfg   = mmengine.Config.fromfile(POSE_CONFIG)
        visualizer = VISUALIZERS.build(
            pose_cfg.visualizer | {'line_width': 1,
                                    'bbox_color': (101,193,255),
                                    'radius': 2})
        valid_ds = next(d for d in pose_datasample if d is not None)
 
        # Ambil dataset_meta dari datasample; fallback ke config pose jika tidak ada
        ds_meta = getattr(valid_ds, 'dataset_meta', None)
        if not ds_meta:
            # Fallback: buat dataset_meta minimal dari config
            try:
                from mmpose.datasets.datasets.utils import parse_pose_metainfo
                ds_meta = parse_pose_metainfo(
                    dict(from_file='configs/_base_/datasets/coco.py'))
            except Exception:
                ds_meta = {}
        visualizer.set_dataset_meta(ds_meta)
 
        for i, (d, f) in enumerate(zip(pose_datasample, frames_)):
            if d is None:
                continue
            try:
                visualizer.add_datasample(
                    'result', f, data_sample=d,
                    draw_gt=False, draw_heatmap=False,
                    draw_bbox=False, draw_pred=True,
                    show=False, wait_time=0, out_file=None, kpt_thr=0.3)
                frames_[i] = visualizer.get_image()
            except Exception:
                # Jika visualisasi skeleton gagal, lanjut tanpa skeleton di frame ini
                pass
 
    for i, frame in enumerate(frames_):
        actual_ts = int(output_timestamps[i])
        fi        = min(actual_ts-1, len(tracked_detections)-1)
        is_anchor = actual_ts in anchor_set
 
        if tracked_detections and fi >= 0:
            for trk in tracked_detections[fi]:
                tid    = trk['track_id']
                lb, sc = None, 0.0
                if is_anchor:
                    ann_idx = list(all_timestamps).index(actual_ts) \
                              if actual_ts in all_timestamps else -1
                    if ann_idx >= 0 and ann_idx < len(stdet_results) \
                            and stdet_results[ann_idx]:
                        for ann in stdet_results[ann_idx]:
                            if ann is None:
                                continue
                            lb_list = ann[1]
                            sc_list = ann[2]
                            if lb_list:
                                lb, sc = lb_list[0], sc_list[0]
                                break
                frame = draw_label_overlay(frame, trk, lb, sc)
 
        if is_anchor:
            cv2.putText(frame, f'Frame:{actual_ts}',
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
 
        frames_[i] = frame
 
    return frames_
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  SAVE TIMING KE JSON (untuk perbandingan dengan YOLO11-pose)
# ─────────────────────────────────────────────────────────────────────────────
def save_timing_json(out_path, timing_data):
    """
    Simpan hasil timing ke JSON agar mudah dibandingkan dengan YOLO11-pose.
    Format output JSON konsisten — bisa langsung di-load di notebook perbandingan.
    """
    with open(out_path, 'w') as f:
        json.dump(timing_data, f, indent=2)
    print(f'\n  Timing data disimpan ke: {out_path}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MODE VIDEO (satu-satunya mode)
# ─────────────────────────────────────────────────────────────────────────────
def run_video(model_posec3d, label_map, video_in, video_out, save_timing=True):
    print(f'\n{"="*60}')
    print(f'  MODE VIDEO — YOLOv8 + HRNet + PoseC3D')
    print(f'{"="*60}')
    print(f'  Input : {video_in}')
    print(f'  Output: {video_out}')
 
    t_total_start = time.perf_counter()
 
    detector = YOLO(YOLO_MODEL_PATH)
 
    # ── Step 1: Ekstrak frame ────────────────────────────────────────────────
    print('\n[1/5] Ekstrak frame...')
    t0      = time.perf_counter()
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames = frame_extract(video_in, 720, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _   = original_frames[0].shape
    del original_frames; gc.collect()
    t_extract = (time.perf_counter() - t0) * 1000
    print(f'      {num_frame} frame | {w}x{h} | {t_extract/1000:.2f}s')
 
    # ── Step 2: YOLOv8 + BoT-SORT ───────────────────────────────────────────
    print('\n[2/5] YOLOv8 + BoT-SORT...')
    tracked_detections, human_detections, times_yolo_ms = run_detection_and_tracking(
        frame_paths, detector, DET_SCORE_THR,
        reid_weights=BOTSORT_REID,
        track_device=DEVICE, half=True
    )
    torch.cuda.empty_cache()
 
    # Hitung jumlah orang per frame (untuk normalisasi)
    n_persons_per_frame = [len(td) for td in tracked_detections]
 
    # ── Step 3: HRNet pose estimation (dengan timing detail) ─────────────────
    print('\n[3/5] HRNet pose estimation...')
    (pose_results_raw,
     pose_datasample,
     times_hrnet_ms,
     times_hrnet_per_person_ms) = run_hrnet_with_timing(
        POSE_CONFIG, POSE_CHECKPOINT,
        frame_paths, human_detections, DEVICE
    )
    pose_results = format_pose_results(pose_results_raw)
 
    # Filter hanya frame yang ada orangnya (untuk statistik yang lebih representatif)
    hrnet_active = [t for t, n in zip(times_hrnet_ms, n_persons_per_frame) if n > 0]
    hrnet_pp     = [t for t in times_hrnet_per_person_ms if t > 0]
    torch.cuda.empty_cache()
 
    # ── Step 4: PoseC3D inference ────────────────────────────────────────────
    print('\n[4/5] PoseC3D inference...')
    clip_len   = PREDICT_STEPSIZE
    timestamps = np.arange(clip_len//2, num_frame+1-clip_len//2, PREDICT_STEPSIZE)
 
    track_pose_index = build_track_pose_index(tracked_detections, pose_results)
 
    stdet_preds      = []
    times_posec3d_ms = []
    t0               = time.perf_counter()
    prog_bar         = mmengine.ProgressBar(len(timestamps))
 
    for timestamp in timestamps:
        start  = timestamp - (clip_len//2 - 1)
        fi     = [max(0, min(int(start+j-1), num_frame-1)) for j in range(clip_len)]
        kp_buf = [pose_results[i]       for i in fi]
        tr_buf = [tracked_detections[i] for i in fi]
        tp_buf = [track_pose_index[i]   for i in fi]
 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_clip = time.perf_counter()
 
        pred = run_posec3d_on_buffer(
            kp_buf, tr_buf, tp_buf, model_posec3d,
            label_map, h, w, ACTION_SCORE_THR, clip_len)
 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_posec3d_ms.append((time.perf_counter() - t_clip) * 1000)
 
        center_tracks = tracked_detections[int(timestamp)-1]
        if center_tracks:
            frame_result = []
            for trk in center_tracks:
                tid   = trk['track_id']
                lb, sc = pred.get(tid, (None, 0.0))
                x1,y1,x2,y2 = trk['bbox']
                bbox_norm = np.array([x1/w, y1/h, x2/w, y2/h])
                frame_result.append((
                    bbox_norm,
                    [lb] if lb else [],
                    [sc] if lb else []
                ))
            stdet_preds.append(frame_result)
        else:
            stdet_preds.append(None)
 
        prog_bar.update()
 
    torch.cuda.empty_cache()
 
    # ── Step 5: Render video ─────────────────────────────────────────────────
    print('\n[5/5] Render video output...')
    t0                = time.perf_counter()
    output_timestamps = np.arange(1, num_frame+1, dtype=np.int64)
    out_frames        = [cv2.imread(frame_paths[min(ts-1, num_frame-1)])
                         for ts in output_timestamps]
    pose_ds_out       = [pose_datasample[min(ts-1, len(pose_datasample)-1)]
                         for ts in output_timestamps]
 
    vis_frames = visualize_video(
        out_frames, stdet_preds, pose_ds_out,
        tracked_detections, output_timestamps, timestamps)
 
    os.makedirs(osp.dirname(video_out) if osp.dirname(video_out) else '.', exist_ok=True)
    vid = mpy.ImageSequenceClip(vis_frames, fps=OUTPUT_FPS)
    vid.write_videofile(video_out, logger=None)
    t_render = (time.perf_counter() - t0) * 1000
 
    t_total = (time.perf_counter() - t_total_start)
 
    tmp_dir.cleanup()
    del vis_frames, pose_results, tracked_detections, human_detections
    gc.collect(); torch.cuda.empty_cache()
 
    # ── RINGKASAN WAKTU KOMPUTASI ────────────────────────────────────────────
    t_yolo_total    = sum(times_yolo_ms)
    t_hrnet_total   = sum(times_hrnet_ms)
    t_posec3d_total = sum(times_posec3d_ms)
    t_pipeline      = t_yolo_total + t_hrnet_total + t_posec3d_total
 
    print(f'\n\n{"="*60}')
    print(f'  STATISTIK WAKTU KOMPUTASI — YOLOv8 + HRNet + PoseC3D')
    print(f'{"="*60}')
    print(f'  Video            : {video_in}')
    print(f'  Resolusi         : {w}x{h}')
    print(f'  Total frame      : {num_frame}')
    print(f'  Device           : {DEVICE}')
    print(f'  Rata2 orang/frame: {np.mean(n_persons_per_frame):.2f}')
 
    print(f'\n  {"─"*56}')
    print(f'  TAHAP 1 — YOLOv8 + BoT-SORT')
    print_timing_summary('Per frame (semua frame)', times_yolo_ms, num_frame)
 
    print(f'\n  {"─"*56}')
    print(f'  TAHAP 2 — HRNet (pose estimation)  ← FOKUS PERBANDINGAN')
    print_timing_summary('Per frame (semua frame, incl. frame kosong)', times_hrnet_ms, num_frame)
    if hrnet_active:
        print_timing_summary('Per frame (hanya frame ada orang)', hrnet_active)
    if hrnet_pp:
        print_timing_summary('Per orang (normalisasi per person)', hrnet_pp)
 
    print(f'\n  {"─"*56}')
    print(f'  TAHAP 3 — PoseC3D (action recognition)')
    print_timing_summary('Per clip ({} frame)'.format(clip_len), times_posec3d_ms)
 
    print(f'\n  {"="*56}')
    print(f'  TOTAL PIPELINE (tanpa render)')
    print(f'    YOLOv8+BoTSORT : {t_yolo_total/1000:7.3f} s  '
          f'({t_yolo_total/num_frame:6.2f} ms/frame | '
          f'{t_yolo_total/t_pipeline*100:.1f}%)')
    print(f'    HRNet          : {t_hrnet_total/1000:7.3f} s  '
          f'({t_hrnet_total/num_frame:6.2f} ms/frame | '
          f'{t_hrnet_total/t_pipeline*100:.1f}%)')
    print(f'    PoseC3D        : {t_posec3d_total/1000:7.3f} s  '
          f'({t_posec3d_total/len(times_posec3d_ms):6.2f} ms/clip  | '
          f'{t_posec3d_total/t_pipeline*100:.1f}%)')
    print(f'    ─────────────────────────────────────────────────────')
    print(f'    Pipeline total : {t_pipeline/1000:7.3f} s')
    print(f'    Effective FPS  : {num_frame/(t_pipeline/1000):7.2f} fps')
    print(f'    Total (incl.render): {t_total:.2f} s')
    print(f'  {"="*56}')
 
    # ── Simpan timing ke JSON ────────────────────────────────────────────────
    if save_timing:
        timing_data = {
            'meta': {
                'model': 'HRNet-W32',
                'pipeline': 'YOLOv8m + BoTSORT + HRNet-W32 + PoseC3D',
                'video': video_in,
                'resolution': f'{w}x{h}',
                'num_frame': num_frame,
                'device': DEVICE,
                'clip_len': clip_len,
                'avg_persons_per_frame': float(np.mean(n_persons_per_frame)),
            },
            'pose_estimator': 'hrnet',  # key untuk perbandingan
            'times_ms': {
                # Data per-frame mentah untuk plotting distribusi
                'yolo_per_frame':           times_yolo_ms,
                'hrnet_per_frame':          times_hrnet_ms,
                'hrnet_per_frame_active':   hrnet_active,
                'hrnet_per_person':         hrnet_pp,
                'posec3d_per_clip':         times_posec3d_ms,
                'n_persons_per_frame':      n_persons_per_frame,
            },
            'summary_ms': {
                # Ringkasan agregat — ini yang dipasangkan dengan YOLO11-pose
                'yolo_mean':              float(np.mean(times_yolo_ms)),
                'yolo_std':               float(np.std(times_yolo_ms)),
                'yolo_total':             float(t_yolo_total),
 
                'hrnet_mean_all':         float(np.mean(times_hrnet_ms)),
                'hrnet_std_all':          float(np.std(times_hrnet_ms)),
                'hrnet_mean_active':      float(np.mean(hrnet_active)) if hrnet_active else 0.0,
                'hrnet_std_active':       float(np.std(hrnet_active))  if hrnet_active else 0.0,
                'hrnet_mean_per_person':  float(np.mean(hrnet_pp))     if hrnet_pp     else 0.0,
                'hrnet_total':            float(t_hrnet_total),
 
                'posec3d_mean':           float(np.mean(times_posec3d_ms)),
                'posec3d_std':            float(np.std(times_posec3d_ms)),
                'posec3d_total':          float(t_posec3d_total),
 
                'pipeline_total':         float(t_pipeline),
                'effective_fps':          float(num_frame / (t_pipeline / 1000)),
            }
        }
 
        json_out = osp.splitext(video_out)[0] + '_timing_hrnet.json'
        save_timing_json(json_out, timing_data)
 
    print(f'\n✓ Video output: {video_out}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Inference video YOLOv8 + HRNet + PoseC3D dengan timing detail')
    parser.add_argument('--input',  type=str, required=True,
                        help='Path video input (.mp4, .avi, dst.)')
    parser.add_argument('--output', type=str, default=VIDEO_OUT_DEFAULT,
                        help=f'Path video output (default: {VIDEO_OUT_DEFAULT})')
    parser.add_argument('--no-save-timing', action='store_true',
                        help='Jangan simpan timing ke JSON')
    args = parser.parse_args()
 
    print('Loading PoseC3D model...')
    label_map     = load_label_map(LABEL_MAP_FILE)
    num_classes   = max(label_map.keys()) + 1
    skeleton_cfg  = mmengine.Config.fromfile(SKELETON_CONFIG)
    skeleton_cfg.model.cls_head.num_classes = num_classes
    model_posec3d = init_recognizer(skeleton_cfg, SKELETON_CHECKPOINT, DEVICE)
    print(f'✓ PoseC3D   : {SKELETON_CHECKPOINT}')
    print(f'✓ Kelas ({num_classes}): {list(label_map.values())}')
 
    run_video(
        model_posec3d, label_map,
        args.input, args.output,
        save_timing=not args.no_save_timing
    )
 
 
if __name__ == '__main__':
    main()