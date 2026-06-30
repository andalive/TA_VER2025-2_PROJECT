"""
Inference Video — YOLO11-pose + BoT-SORT + PoseC3D
====================================================
Pipeline identik dengan inference_hrnet_video.py, tapi pose estimator
diganti YOLO11-pose (all-in-one: deteksi + pose dalam 1 model).

Perbedaan utama vs HRNet:
  - HRNet  : YOLOv8 (deteksi) → HRNet (pose)   → 2 model terpisah
  - YOLO11 : YOLO11-pose (deteksi + pose sekaligus) → 1 model

Output JSON timing-nya format sama dengan inference_hrnet_video.py
sehingga bisa langsung dibandingkan.

Jalankan:
    python inference_yolo11_video.py --input data/test_video/input.mp4
    python inference_yolo11_video.py --input input.mp4 --output hasil.mp4
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
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.utils import frame_extract
import moviepy.editor as mpy

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURASI — sesuaikan path di sini
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_CONFIG     = '01_asset_tools/ciis_7.py'
SKELETON_CHECKPOINT = '02_work_dirs/ciis_7_gab2_2s/best_acc_top1_epoch_230.pth'
LABEL_MAP_FILE      = '03_dataset/ciis_label_map.txt'

# YOLO11-pose — deteksi manusia + pose sekaligus dalam 1 model
YOLO11_POSE_PATH = '01_asset_tools/models/yolo11m-pose.pt'  # sesuaikan nama file
DET_SCORE_THR    = 0.3
KP_CONF_THR      = 0.3   # threshold confidence keypoint

# BoT-SORT (tetap sama)
BOTSORT_REID = Path('01_asset_tools/models/osnet_x0_25_msmt17.pt')

ACTION_SCORE_THR  = 0.75
PREDICT_STEPSIZE  = 4
OUTPUT_FPS        = 12
DEVICE            = 'cuda:0'
VIDEO_OUT_DEFAULT = '03_dataset/test_video/output_yolo11pose.mp4'

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
 
def save_timing_json(out_path, timing_data):
    """Simpan hasil timing ke JSON — format sama dengan HRNet untuk perbandingan."""
    with open(out_path, 'w') as f:
        json.dump(timing_data, f, indent=2)
    print(f'\n  Timing data disimpan ke: {out_path}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  YOLO11-pose — deteksi + pose dalam 1 langkah per frame
# ─────────────────────────────────────────────────────────────────────────────
def run_yolo11pose_with_timing(frame_paths, model_path, det_score_thr,
                                reid_weights, device, half=True):
    """
    YOLO11-pose: deteksi orang + estimasi pose sekaligus per frame,
    lalu BoT-SORT untuk tracking.
 
    Berbeda dengan HRNet yang butuh 2 langkah (YOLOv8 → HRNet),
    YOLO11-pose melakukan keduanya dalam 1 forward pass.
 
    Returns:
        tracked_detections       : list[list[dict]]  per frame
        pose_results             : list[dict]         per frame (format sama dengan HRNet)
        times_yolo11_ms          : list[float]        waktu YOLO11-pose per frame (ms)
        times_yolo11_per_person  : list[float]        waktu per orang per frame (ms)
        n_persons_per_frame      : list[int]          jumlah orang per frame
    """
    print(f'Loading YOLO11-pose dari {model_path}...')
    pose_model = YOLO(model_path)
 
    tracker = BoTSORT(
        model_weights=reid_weights,
        device=device,
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
 
    tracked_detections      = []
    pose_results            = []
    times_yolo11_ms         = []
    times_yolo11_per_person = []
    times_botsort_ms        = []
    n_persons_per_frame     = []
 
    print('Running YOLO11-pose + BoT-SORT per frame...')
    prog_bar = mmengine.ProgressBar(len(frame_paths))
 
    for frame_path in frame_paths:
        frame_bgr = cv2.imread(frame_path)
 
        # ── YOLO11-pose: deteksi + pose dalam 1 forward pass ────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
 
        results = pose_model(frame_bgr, classes=[0],
                             conf=det_score_thr, verbose=False)
 
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
 
        # ── Parse output YOLO11-pose ─────────────────────────────────────────
        res       = results[0]
        boxes     = res.boxes.xyxy.cpu().numpy()    # (N, 4)
        scores    = res.boxes.conf.cpu().numpy()    # (N,)
        keypoints = res.keypoints                    # Keypoints object
 
        n_person = len(boxes)
        times_yolo11_ms.append(elapsed_ms)
        times_yolo11_per_person.append(
            elapsed_ms / n_person if n_person > 0 else 0.0)
        n_persons_per_frame.append(n_person)
 
        # ── BoT-SORT tracking (diukur terpisah) ──────────────────────────────
        if n_person > 0:
            dets = np.hstack([boxes, scores[:, None], np.zeros((n_person, 1))])
        else:
            dets = np.empty((0, 6))
 
        t_sort = time.perf_counter()
        tracks = tracker.update(dets, frame_bgr)
        times_botsort_ms.append((time.perf_counter() - t_sort) * 1000)
 
        frame_tracks = []
        if len(tracks) > 0:
            for t in tracks:
                frame_tracks.append({
                    'track_id': int(t[4]),
                    'bbox':     [t[0], t[1], t[2], t[3]],
                    'score':    float(t[5])
                })
        tracked_detections.append(frame_tracks)
 
        # ── Format keypoint ke dict konsisten (sama dengan HRNet output) ────
        if n_person > 0 and keypoints is not None:
            kp_xy  = keypoints.xy.cpu().numpy()    # (N, 17, 2)
            kp_conf = keypoints.conf               # bisa None kalau model tidak output conf
            if kp_conf is not None:
                kp_conf = kp_conf.cpu().numpy()    # (N, 17)
            else:
                # Fallback: isi confidence 1.0 untuk semua keypoint terdeteksi
                kp_conf = np.ones((n_person, 17), dtype=np.float32)
 
            pose_results.append({
                'keypoints':       kp_xy,    # (N, 17, 2)
                'keypoint_scores': kp_conf,  # (N, 17)
                'bboxes':          boxes     # (N, 4)
            })
        else:
            pose_results.append({
                'keypoints':       np.zeros((0, 17, 2)),
                'keypoint_scores': np.zeros((0, 17)),
                'bboxes':          np.zeros((0, 4))
            })
 
        prog_bar.update()
 
    print(f'\nSelesai. Total frame: {len(frame_paths)}')
    return (tracked_detections, pose_results,
            times_yolo11_ms, times_yolo11_per_person,
            times_botsort_ms, n_persons_per_frame)
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  BUILD TRACK POSE INDEX — identik dengan HRNet
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
#  INFERENCE POSEC3D — identik dengan HRNet
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
#  VISUALISASI — gambar skeleton manual dari keypoint YOLO11-pose
# ─────────────────────────────────────────────────────────────────────────────
 
# COCO 17-keypoint skeleton connections
COCO_SKELETON = [
    (0,1),(0,2),(1,3),(2,4),          # kepala
    (5,6),                             # bahu
    (5,7),(7,9),(6,8),(8,10),         # lengan
    (5,11),(6,12),(11,12),            # torso
    (11,13),(13,15),(12,14),(14,16)   # kaki
]
KP_COLOR  = (0, 255, 128)
LIMB_COLOR = (255, 165, 0)
 
def draw_skeleton(frame, keypoints, keypoint_scores, kp_thr=0.3):
    """Gambar skeleton COCO 17-keypoint pada frame."""
    for kp, kpc in zip(keypoints, keypoint_scores):
        # Titik keypoint
        for i, (x, y) in enumerate(kp):
            if kpc[i] >= kp_thr and x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 3, KP_COLOR, -1)
        # Garis limb
        for i, j in COCO_SKELETON:
            if (kpc[i] >= kp_thr and kpc[j] >= kp_thr and
                    kp[i][0] > 0 and kp[j][0] > 0):
                cv2.line(frame,
                         (int(kp[i][0]), int(kp[i][1])),
                         (int(kp[j][0]), int(kp[j][1])),
                         LIMB_COLOR, 2)
    return frame
 
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
 
def visualize_video(frames, stdet_results, pose_results_all,
                    tracked_detections, output_timestamps, all_timestamps,
                    kp_thr=0.3):
    """
    Visualisasi video — skeleton digambar manual dari keypoint YOLO11-pose
    (tidak butuh MMPose visualizer / dataset_meta).
    """
    frames_    = cp.deepcopy(frames)
    frames_    = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
    anchor_set = set(all_timestamps.tolist())
 
    for i, frame in enumerate(frames_):
        actual_ts = int(output_timestamps[i])
        fi        = min(actual_ts - 1, len(tracked_detections) - 1)
        is_anchor = actual_ts in anchor_set
 
        # Gambar skeleton dari pose_results
        if fi >= 0 and fi < len(pose_results_all):
            pr = pose_results_all[fi]
            if len(pr['keypoints']) > 0:
                frame = draw_skeleton(frame,
                                      pr['keypoints'],
                                      pr['keypoint_scores'],
                                      kp_thr=kp_thr)
 
        # Gambar bbox + label aksi
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
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
 
        frames_[i] = frame
 
    return frames_
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MODE VIDEO
# ─────────────────────────────────────────────────────────────────────────────
def run_video(model_posec3d, label_map, video_in, video_out, save_timing=True):
    print(f'\n{"="*60}')
    print(f'  MODE VIDEO — YOLO11-pose + PoseC3D')
    print(f'{"="*60}')
    print(f'  Input : {video_in}')
    print(f'  Output: {video_out}')
 
    t_total_start = time.perf_counter()
 
    # ── Step 1: Ekstrak frame ────────────────────────────────────────────────
    print('\n[1/4] Ekstrak frame...')
    t0      = time.perf_counter()
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames = frame_extract(video_in, 720, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _   = original_frames[0].shape
    del original_frames; gc.collect()
    t_extract = (time.perf_counter() - t0) * 1000
    print(f'      {num_frame} frame | {w}x{h} | {t_extract/1000:.2f}s')
 
    # ── Step 2: YOLO11-pose + BoT-SORT ──────────────────────────────────────
    # Ini menggantikan 2 tahap terpisah di HRNet (YOLOv8 + HRNet)
    print('\n[2/4] YOLO11-pose + BoT-SORT (deteksi + pose sekaligus)...')
    (tracked_detections,
     pose_results,
     times_yolo11_ms,
     times_yolo11_per_person,
     times_botsort_ms,
     n_persons_per_frame) = run_yolo11pose_with_timing(
        frame_paths, YOLO11_POSE_PATH, DET_SCORE_THR,
        reid_weights=BOTSORT_REID,
        device=DEVICE, half=True
    )
    torch.cuda.empty_cache()
 
    # Filter frame aktif (ada orangnya) untuk statistik representatif
    yolo11_active  = [t for t, n in zip(times_yolo11_ms, n_persons_per_frame) if n > 0]
    yolo11_pp      = [t for t in times_yolo11_per_person if t > 0]
    botsort_active = [t for t, n in zip(times_botsort_ms, n_persons_per_frame) if n > 0]
 
    # ── Step 3: PoseC3D inference ────────────────────────────────────────────
    print('\n[3/4] PoseC3D inference...')
    clip_len   = PREDICT_STEPSIZE
    timestamps = np.arange(clip_len//2, num_frame+1-clip_len//2, PREDICT_STEPSIZE)
 
    track_pose_index = build_track_pose_index(tracked_detections, pose_results)
 
    stdet_preds      = []
    times_posec3d_ms = []
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
                tid    = trk['track_id']
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
 
    # ── Step 4: Render video ─────────────────────────────────────────────────
    print('\n[4/4] Render video output...')
    output_timestamps = np.arange(1, num_frame+1, dtype=np.int64)
    out_frames        = [cv2.imread(frame_paths[min(ts-1, num_frame-1)])
                         for ts in output_timestamps]
 
    vis_frames = visualize_video(
        out_frames, stdet_preds, pose_results,
        tracked_detections, output_timestamps, timestamps,
        kp_thr=KP_CONF_THR)
 
    os.makedirs(osp.dirname(video_out) if osp.dirname(video_out) else '.', exist_ok=True)
    vid = mpy.ImageSequenceClip(vis_frames, fps=OUTPUT_FPS)
    vid.write_videofile(video_out, logger=None)
 
    t_total = time.perf_counter() - t_total_start
 
    tmp_dir.cleanup()
    del vis_frames, pose_results, tracked_detections
    gc.collect(); torch.cuda.empty_cache()
 
    # ── RINGKASAN WAKTU KOMPUTASI ────────────────────────────────────────────
    t_yolo11_total  = sum(times_yolo11_ms)
    t_botsort_total = sum(times_botsort_ms)
    t_posec3d_total = sum(times_posec3d_ms)
    t_pipeline      = t_yolo11_total + t_botsort_total + t_posec3d_total
 
    print(f'\n\n{"="*60}')
    print(f'  STATISTIK WAKTU KOMPUTASI — YOLO11-pose + BoTSORT + PoseC3D')
    print(f'{"="*60}')
    print(f'  Video            : {video_in}')
    print(f'  Resolusi         : {w}x{h}')
    print(f'  Total frame      : {num_frame}')
    print(f'  Device           : {DEVICE}')
    print(f'  Rata2 orang/frame: {np.mean(n_persons_per_frame):.2f}')
 
    print(f'\n  {"─"*56}')
    print(f'  TAHAP 1 — YOLO11-pose (deteksi + pose sekaligus)  ← FOKUS')
    print_timing_summary('Per frame (semua frame)', times_yolo11_ms, num_frame)
    if yolo11_active:
        print_timing_summary('Per frame (hanya frame ada orang)', yolo11_active)
    if yolo11_pp:
        print_timing_summary('Per orang (normalisasi per person)', yolo11_pp)
 
    print(f'\n  {"─"*56}')
    print(f'  TAHAP 2 — BoT-SORT (tracking)')
    print_timing_summary('Per frame (semua frame)', times_botsort_ms, num_frame)
    if botsort_active:
        print_timing_summary('Per frame (hanya frame ada orang)', botsort_active)
 
    print(f'\n  {"─"*56}')
    print(f'  TAHAP 3 — PoseC3D (action recognition)')
    print_timing_summary('Per clip ({} frame)'.format(clip_len), times_posec3d_ms)
 
    print(f'\n  {"="*56}')
    print(f'  TOTAL PIPELINE (tanpa render)')
    print(f'    YOLO11-pose    : {t_yolo11_total/1000:7.3f} s  '
          f'({t_yolo11_total/num_frame:6.2f} ms/frame | '
          f'{t_yolo11_total/t_pipeline*100:.1f}%)')
    print(f'    BoT-SORT       : {t_botsort_total/1000:7.3f} s  '
          f'({t_botsort_total/num_frame:6.2f} ms/frame | '
          f'{t_botsort_total/t_pipeline*100:.1f}%)')
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
                'model': 'YOLO11m-pose',
                'pipeline': 'YOLO11m-pose + BoTSORT + PoseC3D',
                'video': video_in,
                'resolution': f'{w}x{h}',
                'num_frame': num_frame,
                'device': DEVICE,
                'clip_len': clip_len,
                'avg_persons_per_frame': float(np.mean(n_persons_per_frame)),
            },
            'pose_estimator': 'yolo11',   # key untuk perbandingan
            'times_ms': {
                # Format sama persis dengan HRNet untuk perbandingan langsung
                'yolo11_per_frame':          times_yolo11_ms,
                'yolo11_per_frame_active':   yolo11_active,
                'yolo11_per_person':         yolo11_pp,
                'botsort_per_frame':         times_botsort_ms,
                'botsort_per_frame_active':  botsort_active,
                'posec3d_per_clip':          times_posec3d_ms,
                'n_persons_per_frame':       n_persons_per_frame,
            },
            'summary_ms': {
                # Key untuk cross-compare dengan HRNet JSON
                'yolo11_mean_all':          float(np.mean(times_yolo11_ms)),
                'yolo11_std_all':           float(np.std(times_yolo11_ms)),
                'yolo11_mean_active':       float(np.mean(yolo11_active)) if yolo11_active else 0.0,
                'yolo11_std_active':        float(np.std(yolo11_active))  if yolo11_active else 0.0,
                'yolo11_mean_per_person':   float(np.mean(yolo11_pp))     if yolo11_pp     else 0.0,
                'yolo11_total':             float(t_yolo11_total),
 
                'botsort_mean_all':         float(np.mean(times_botsort_ms)),
                'botsort_std_all':          float(np.std(times_botsort_ms)),
                'botsort_mean_active':      float(np.mean(botsort_active)) if botsort_active else 0.0,
                'botsort_total':            float(t_botsort_total),
 
                'posec3d_mean':             float(np.mean(times_posec3d_ms)),
                'posec3d_std':              float(np.std(times_posec3d_ms)),
                'posec3d_total':            float(t_posec3d_total),
 
                'pipeline_total':           float(t_pipeline),
                'effective_fps':            float(num_frame / (t_pipeline / 1000)),
            }
        }
 
        json_out = osp.splitext(video_out)[0] + '_timing_yolo11pose.json'
        save_timing_json(json_out, timing_data)
 
    print(f'\n✓ Video output: {video_out}')
 
 
# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Inference video YOLO11-pose + PoseC3D dengan timing detail')
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
    print(f'✓ PoseC3D    : {SKELETON_CHECKPOINT}')
    print(f'✓ Kelas ({num_classes}) : {list(label_map.values())}')
    print(f'✓ YOLO11-pose: {YOLO11_POSE_PATH}')
 
    run_video(
        model_posec3d, label_map,
        args.input, args.output,
        save_timing=not args.no_save_timing
    )
 
 
if __name__ == '__main__':
    main()