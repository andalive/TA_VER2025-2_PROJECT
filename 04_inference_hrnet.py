"""
Inference Semi Real-Time — YOLOv8 + BoT-SORT + HRNet + PoseC3D
===============================================================
Jalankan dari terminal:
    python 04_inference_hrnet.py --mode video --input data/test_video/input.mp4 --> contoh (sesuaikan path input)
    python 04_inference_hrnet.py --mode camera

Tekan Q untuk keluar (mode camera).
"""

import os
import os.path as osp
import argparse
import copy as cp
import tempfile
import time
import gc

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
SKELETON_CHECKPOINT = '02_work_dirs/<sesuaikan_folder>/<sesuaikan_checkpoint>.pth'
LABEL_MAP_FILE      = '03_dataset/ciis_label_map.txt'

# YOLOv8 — deteksi manusia
YOLO_MODEL_PATH = os.path.abspath('01_asset_tools/models/yolov8m.pt')
DET_SCORE_THR   = 0.3

# HRNet — pose estimation
POSE_CONFIG     = ('mmaction2/demo/demo_configs/'
                   'td-hm_hrnet-w32_8xb64-210e_coco-256x192_infer.py')
POSE_CHECKPOINT = ('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                   'hrnet_w32_coco_256x192-c78dce93_20200708.pth')

# BoT-SORT
BOTSORT_REID    = Path('01_asset_tools/models/osnet_x0_25_msmt17.pt')

ACTION_SCORE_THR  = 0.75
PREDICT_STEPSIZE  = 4
OUTPUT_FPS        = 12
DEVICE            = 'cuda:0'
CAMERA_IDX        = 0
CAMERA_W          = 640
CAMERA_H          = 480
VIDEO_OUT_DEFAULT = '03_dataset/test_video/test_out/output1_hrnet.mp4'
CAMERA_OUT        = '04_lampiran/<sesuaikan>.mp4'

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────────────────────────────────────────
FONTFACE  = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.7
THICKNESS = 2
LINETYPE  = 1
BAHAYA    = ['membidik senapan', 'membidik pistol',
             'memukul', 'menendang']

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

# ─────────────────────────────────────────────────────────────────────────────
#  YOLOv8 + BoT-SORT — sama persis dengan clip_extraction.ipynb
# ─────────────────────────────────────────────────────────────────────────────
def run_detection_and_tracking(frame_paths, detector, det_score_thr,
                                reid_weights, track_device, half=True):
    """
    YOLOv8 + BoT-SORT untuk setiap frame.
    Sama persis dengan clip_extraction.ipynb Section 3.

    Returns:
        tracked_detections : list[list[dict]]  per frame → {track_id, bbox, score}
        human_detections   : list[np.ndarray]  per frame → [N,5] untuk pose_inference
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

    print('Running YOLOv8 + BoT-SORT tracking...')
    prog_bar = mmengine.ProgressBar(len(frame_paths))

    for frame_path in frame_paths:
        frame_bgr = cv2.imread(frame_path)

        yolo_res = detector(frame_bgr, classes=[0], verbose=False)
        boxes    = yolo_res[0].boxes.xyxy.cpu().numpy()
        scores   = yolo_res[0].boxes.conf.cpu().numpy()

        if len(boxes) > 0:
            dets = np.hstack([boxes, scores[:, None], np.zeros((len(boxes), 1))])
        else:
            dets = np.empty((0, 6))

        tracks = tracker.update(dets, frame_bgr)

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
    return tracked_detections, human_detections


def format_pose_results(pose_results_raw):
    """
    Format ulang output pose_inference MMPose ke format dict konsisten.
    Sama seperti yang dipakai di build_track_pose_index clip_extraction.
    """
    pose_results = []
    for frame_poses in pose_results_raw:
        if not frame_poses:
            pose_results.append({
                'keypoints':       np.zeros((0, 17, 2)),
                'keypoint_scores': np.zeros((0, 17)),
                'bboxes':          np.zeros((0, 4))
            })
            continue
        kps    = np.array([p['keypoints']       for p in frame_poses])
        kpconf = np.array([p['keypoint_scores'] for p in frame_poses])
        # bboxes dari pose_inference formatnya bisa berbeda — ambil dengan aman
        bboxes = []
        for p in frame_poses:
            bb = p.get('bboxes', p.get('bbox', np.zeros(4)))
            if hasattr(bb, 'flatten'):
                bb = bb.flatten()[:4]
            bboxes.append(np.array(bb).flatten()[:4])
        bboxes = np.array(bboxes) if bboxes else np.zeros((0, 4))
        pose_results.append({
            'keypoints':       kps,
            'keypoint_scores': kpconf,
            'bboxes':          bboxes
        })
    return pose_results

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD TRACK POSE INDEX — sama persis dengan clip_extraction.ipynb
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
#  INFERENCE POSEC3D — pakai track_pose_index seperti clip_extraction
# ─────────────────────────────────────────────────────────────────────────────
def run_posec3d_on_buffer(kp_buffer, track_buffer, track_pose_index_buf,
                           model_posec3d, label_map, h, w,
                           action_score_thr, clip_len):
    """
    Jalankan PoseC3D inference pada buffer clip_len frame.
    Menggunakan track_pose_index untuk matching — konsisten dengan clip_extraction.
    Returns: {track_id: (label, score)}
    """
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
#  VISUALISASI — sama persis dengan clip_extraction.ipynb Section 5
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
                    tracked_detections, output_timestamps,
                    all_timestamps):
    """
    Visualisasi untuk mode video — skeleton via MMPose VISUALIZERS.
    Sama dengan visualize() di clip_extraction.ipynb.
    """
    frames_ = cp.deepcopy(frames)
    frames_ = [mmcv.imconvert(f, 'bgr', 'rgb') for f in frames_]
    anchor_set = set(all_timestamps.tolist())
    h, w, _    = frames[0].shape

    # Skeleton via MMPose VISUALIZERS
    if pose_datasample and any(d is not None for d in pose_datasample):
        pose_cfg   = mmengine.Config.fromfile(POSE_CONFIG)
        visualizer = VISUALIZERS.build(
            pose_cfg.visualizer | {'line_width': 1,
                                    'bbox_color': (101,193,255),
                                    'radius': 2})
        visualizer.set_dataset_meta(
            next(d for d in pose_datasample if d is not None).dataset_meta)
        for i, (d, f) in enumerate(zip(pose_datasample, frames_)):
            if d is None:
                continue
            visualizer.add_datasample('result', f, data_sample=d,
                                      draw_gt=False, draw_heatmap=False,
                                      draw_bbox=False, draw_pred=True,
                                      show=False, wait_time=0, 
                                      out_file=None, kpt_thr=0.3)
            frames_[i] = visualizer.get_image()

    for i, frame in enumerate(frames_):
        actual_ts = int(output_timestamps[i])
        fi        = min(actual_ts-1, len(tracked_detections)-1)
        is_anchor = actual_ts in anchor_set

        # Bbox + label
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
#  MODE VIDEO
# ─────────────────────────────────────────────────────────────────────────────
def run_video(model_posec3d, label_map, video_in, video_out):
    print(f'\n[MODE VIDEO — YOLOv8+HRNet]')
    print(f'  Input : {video_in}')
    print(f'  Output: {video_out}')

    detector = YOLO(YOLO_MODEL_PATH)

    # Step 1: Ekstrak frame
    print('\n[1/5] Ekstrak frame...')
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames = frame_extract(video_in, 720, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _   = original_frames[0].shape
    del original_frames; gc.collect()
    print(f'      {num_frame} frame | {w}x{h}')

    # Step 2: YOLOv8 + BoT-SORT
    print('\n[2/5] YOLOv8 + BoT-SORT...')
    t0 = time.time()
    tracked_detections, human_detections = run_detection_and_tracking(
        frame_paths, detector, DET_SCORE_THR,
        reid_weights=BOTSORT_REID,
        track_device=DEVICE, half=True
    )
    t_det = time.time() - t0
    torch.cuda.empty_cache()
    print(f'      {t_det:.1f}s ({t_det/num_frame*1000:.1f}ms/frame)')

    # Step 3: HRNet pose estimation
    print('\n[3/5] HRNet pose estimation...')
    t0 = time.time()
    pose_results_raw, pose_datasample = pose_inference(
        POSE_CONFIG, POSE_CHECKPOINT,
        frame_paths, human_detections, device=DEVICE)
    pose_results = format_pose_results(pose_results_raw)
    t_pose = time.time() - t0
    torch.cuda.empty_cache()
    print(f'      {t_pose:.1f}s ({t_pose/num_frame*1000:.1f}ms/frame)')

    # Step 4: PoseC3D inference
    print('\n[4/5] PoseC3D inference...')
    clip_len   = PREDICT_STEPSIZE
    timestamps = np.arange(clip_len//2, num_frame+1-clip_len//2, PREDICT_STEPSIZE)

    track_pose_index = build_track_pose_index(tracked_detections, pose_results)

    stdet_preds     = []
    inference_times = []
    t0              = time.time()
    prog_bar        = mmengine.ProgressBar(len(timestamps))

    for timestamp in timestamps:
        start  = timestamp - (clip_len//2 - 1)
        fi     = [max(0, min(int(start+j-1), num_frame-1)) for j in range(clip_len)]
        kp_buf = [pose_results[i]       for i in fi]
        tr_buf = [tracked_detections[i] for i in fi]
        tp_buf = [track_pose_index[i]   for i in fi]

        t_inf = time.time()
        pred  = run_posec3d_on_buffer(
            kp_buf, tr_buf, tp_buf, model_posec3d,
            label_map, h, w, ACTION_SCORE_THR, clip_len)
        inference_times.append((time.time()-t_inf)*1000)

        # Format stdet_results seperti pack_result
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

    t_inf_total = time.time() - t0
    torch.cuda.empty_cache()

    total_time = t_det + t_pose + t_inf_total
    print(f'\n\n{"="*50}')
    print(f'  STATISTIK (YOLOv8+HRNet+VIDEO)')
    print(f'{"="*50}')
    print(f'  Frame              : {num_frame}')
    print(f'  YOLOv8+BoTSORT     : {t_det:.2f}s ({t_det/num_frame*1000:.1f}ms/frame)')
    print(f'  HRNet pose         : {t_pose:.2f}s ({t_pose/num_frame*1000:.1f}ms/frame)')
    print(f'  PoseC3D inference  : {t_inf_total:.2f}s ({np.mean(inference_times):.1f}ms/clip avg)')
    print(f'  Total              : {total_time:.2f}s')
    print(f'  Effective FPS      : {num_frame/total_time:.2f}')
    print(f'{"="*50}')

    # Step 5: Render video
    print('\n[5/5] Render video output...')
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

    tmp_dir.cleanup()
    del vis_frames, pose_results, tracked_detections, human_detections
    gc.collect(); torch.cuda.empty_cache()
    print(f'\n✓ Selesai: {video_out}')

# ─────────────────────────────────────────────────────────────────────────────
#  MODE KAMERA
# ─────────────────────────────────────────────────────────────────────────────
def run_camera(model_posec3d, label_map):
    print(f'\n[MODE KAMERA — YOLOv8+HRNet] index={CAMERA_IDX} | Q=keluar')

    from mmpose.apis import inference_topdown, init_model as init_pose_model
    detector   = YOLO(YOLO_MODEL_PATH)
    pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device=DEVICE)

    # Visualizer skeleton
    pose_cfg = mmengine.Config.fromfile(POSE_CONFIG)

    visualizer = VISUALIZERS.build(
        pose_cfg.visualizer | {
            'line_width': 2,
            'radius': 3,
            'bbox_color': (101, 193, 255)
        }
    )

    visualizer.set_dataset_meta(
        pose_model.dataset_meta
    )

    tracker = BoTSORT(
        model_weights=BOTSORT_REID, device=DEVICE, fp16=True,
        track_high_thresh=DET_SCORE_THR, track_low_thresh=0.1,
        new_track_thresh=DET_SCORE_THR, track_buffer=50,
        match_thresh=0.8, proximity_thresh=0.5,
        appearance_thresh=0.25, with_reid=True,
    )

    cap = cv2.VideoCapture(CAMERA_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

    if not cap.isOpened():
        print(f'ERROR: Kamera {CAMERA_IDX} tidak bisa dibuka.')
        return

    # Warm up
    for _ in range(30):
        cap.read()

    # Window
    cv2.namedWindow('CIIS — YOLOv8+HRNet (Q=keluar)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CIIS — YOLOv8+HRNet (Q=keluar)', 1280, 720)

    # VideoWriter
    cam_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(osp.dirname(CAMERA_OUT) if osp.dirname(CAMERA_OUT) else '.', exist_ok=True)
    out_writer = cv2.VideoWriter(CAMERA_OUT, fourcc, cam_fps, (CAMERA_W, CAMERA_H))
    print(f'Rekaman disimpan ke: {CAMERA_OUT}')

    # Sliding window buffer
    kp_buffer      = []
    track_buffer   = []
    tpi_buffer     = []   # track_pose_index per frame
    current_labels = {}

    fps_counter = 0
    t_fps       = time.time()
    fps_display = 0.0

    print('Kamera aktif. Tekan Q untuk keluar.')

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        h, w = frame_bgr.shape[:2]

        # ── YOLOv8 deteksi ────────────────────────────────────────────────
        yolo_res = detector(frame_bgr, classes=[0], verbose=False)
        boxes    = yolo_res[0].boxes.xyxy.cpu().numpy()
        scores   = yolo_res[0].boxes.conf.cpu().numpy()

        if len(boxes) > 0:
            dets = np.hstack([boxes, scores[:,None], np.zeros((len(boxes),1))])
        else:
            dets = np.empty((0,6))

        tracks = tracker.update(dets, frame_bgr)

        frame_tracks = []
        frame_human_det = []
        if len(tracks) > 0:
            for t in tracks:
                frame_tracks.append({'track_id': int(t[4]),
                                     'bbox': [t[0],t[1],t[2],t[3]],
                                     'score': float(t[5])})
                frame_human_det.append([t[0],t[1],t[2],t[3],float(t[5])])

        # ── HRNet pose estimation ─────────────────────────────────────────
        pose_frame = {
            'keypoints': np.zeros((0,17,2)),
            'keypoint_scores': np.zeros((0,17)),
            'bboxes': np.zeros((0,4))
        }

        pose_datasample = None

        if frame_human_det:

            det_arr = np.array(frame_human_det, dtype=np.float32)

            pose_res_raw = inference_topdown(
                pose_model,
                frame_bgr,
                det_arr[:, :4]
            )

            if len(pose_res_raw) > 0:

                kps = np.array([
                    r.pred_instances.keypoints[0]
                    for r in pose_res_raw
                ])

                kpconf = np.array([
                    r.pred_instances.keypoint_scores[0]
                    for r in pose_res_raw
                ])

                bboxes = det_arr[:, :4]

                pose_frame = {
                    'keypoints': kps,
                    'keypoint_scores': kpconf,
                    'bboxes': bboxes
                }

                pose_datasample = pose_res_raw

        # ── Buffer ────────────────────────────────────────────────────────
        kp_buffer.append(pose_frame)
        track_buffer.append(frame_tracks)

        # Build track_pose_index untuk frame ini
        id_to_pose = {}
        if frame_tracks and len(pose_frame['keypoints']) > 0:
            for track in frame_tracks:
                tid = track['track_id']
                best_iou, best_pidx = -1, -1
                for pidx, pbbox in enumerate(pose_frame['bboxes']):
                    iou = _cal_iou(track['bbox'], pbbox)
                    if iou > best_iou:
                        best_iou, best_pidx = iou, pidx
                if best_pidx >= 0 and best_iou > 0.1:
                    id_to_pose[tid] = best_pidx
        tpi_buffer.append(id_to_pose)

        if len(kp_buffer) > PREDICT_STEPSIZE:
            kp_buffer.pop(0)
            track_buffer.pop(0)
            tpi_buffer.pop(0)

        # ── Inference kalau buffer penuh ──────────────────────────────────
        if len(kp_buffer) == PREDICT_STEPSIZE:
            pred = run_posec3d_on_buffer(
                kp_buffer, track_buffer, tpi_buffer,
                model_posec3d, label_map, h, w,
                ACTION_SCORE_THR, PREDICT_STEPSIZE)
            for tid, val in pred.items():
                current_labels[tid] = val

        # ── Overlay ───────────────────────────────────────────────────────
        display = frame_bgr.copy()

        if pose_datasample is not None:

            for ds in pose_datasample:

                visualizer.add_datasample(
                    'result',
                    display,
                    data_sample=ds,
                    draw_gt=False,
                    draw_heatmap=False,
                    draw_bbox=False,
                    draw_pred=True,
                    show=False,
                    wait_time=0,
                    out_file=None,
                    kpt_thr=0.3
                )

        display = visualizer.get_image()

        for trk in frame_tracks:
            tid    = trk['track_id']
            lb, sc = current_labels.get(tid, (None, 0.0))
            display = draw_label_overlay(display, trk, lb, sc)

        fps_counter += 1
        if fps_counter % 15 == 0:
            fps_display = 15 / (time.time() - t_fps)
            t_fps       = time.time()

        cv2.putText(display, f'FPS: {fps_display:.1f}',
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(display, f'Buffer: {len(kp_buffer)}/{PREDICT_STEPSIZE}',
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(display, 'YOLOv8 + HRNet + PoseC3D',
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        out_writer.write(display)
        cv2.imshow('CIIS — YOLOv8+HRNet (Q=keluar)', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_writer.release()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()
    print(f'\nSelesai. Rekaman: {CAMERA_OUT}')

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Inference YOLOv8+HRNet+PoseC3D')
    parser.add_argument('--mode',   type=str, default='video',
                        choices=['camera', 'video'],
                        help='Mode input: camera atau video')
    parser.add_argument('--input',  type=str, default=None,
                        help='Path video input (mode video)')
    parser.add_argument('--output', type=str, default=VIDEO_OUT_DEFAULT,
                        help='Path video output (mode video)')
    args = parser.parse_args()

    print('Loading PoseC3D...')
    label_map     = load_label_map(LABEL_MAP_FILE)
    num_classes   = max(label_map.keys()) + 1
    skeleton_cfg  = mmengine.Config.fromfile(SKELETON_CONFIG)
    skeleton_cfg.model.cls_head.num_classes = num_classes
    model_posec3d = init_recognizer(skeleton_cfg, SKELETON_CHECKPOINT, DEVICE)
    print(f'✓ PoseC3D   : {SKELETON_CHECKPOINT}')
    print(f'✓ Kelas ({num_classes}): {list(label_map.values())}')

    if args.mode == 'camera':
        run_camera(model_posec3d, label_map)
    else:
        if args.input is None:
            print('ERROR: --input harus diisi untuk mode video.')
            print('Contoh: python inference_hrnet.py --mode video --input input.mp4')
            return
        run_video(model_posec3d, label_map, args.input, args.output)


if __name__ == '__main__':
    main()
