"""
Inference Semi Real-Time — YOLO11-pose + PoseC3D
=================================================
Jalankan dari terminal:
    python inference_yolo11.py --mode camera
    python inference_yolo11.py --mode video --input data/test_video/input.mp4

Tekan Q untuk keluar (mode camera).
"""

import os
import os.path as osp
import argparse
import time
import copy as cp

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
import tempfile
import gc

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─────────────────────────────────────────────────────────────────────────────
#  KONFIGURASI — sesuaikan path di sini
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_CONFIG     = '01_asset_tools/ciis_7.py'
SKELETON_CHECKPOINT = '02_work_dirs/ciis_7_gab2_2s/best_acc_top1_epoch_230.pth'
LABEL_MAP_FILE      = '03_dataset/ciis_label_map.txt'
MODEL_POSE_PATH     = '01_asset_tools/models/yolo11m-pose.pt'
BOTSORT_REID        = Path('01_asset_tools/models/osnet_x0_25_msmt17.pt')
DEVICE              = 'cuda:0'
ACTION_SCORE_THR    = 0.75
PREDICT_STEPSIZE    = 4
DET_SCORE_THR       = 0.3
OUTPUT_FPS          = 12        # fps video output (mode video)
CAMERA_IDX          = 0        # index webcam
CAMERA_W            = 640
CAMERA_H            = 480
VIDEO_OUT_DEFAULT   = '03_dataset/test_video/test_out/output_yolo11.mp4'

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
    return (int(h[:2],16), int(h[2:4],16), int(h[4:],16))

PLATEBLUE = [hex2color(h) for h in '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'.split('-')]

def load_label_map(path):
    lines = open(path).readlines()
    return {int(x[0]): x[1].strip()
            for x in [l.strip().split(': ') for l in lines if l.strip()]}

def abbrev(name):
    while '(' in name:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed+1:]
    return name

def cal_iou(b1, b2):
    xmin1,ymin1,xmax1,ymax1 = b1
    xmin2,ymin2,xmax2,ymax2 = b2
    s1    = max(0,xmax1-xmin1)*max(0,ymax1-ymin1)
    s2    = max(0,xmax2-xmin2)*max(0,ymax2-ymin2)
    xi    = max(0,min(xmax1,xmax2)-max(xmin1,xmin2))
    yi    = max(0,min(ymax1,ymax2)-max(ymin1,ymin2))
    inter = xi*yi
    union = s1+s2-inter
    return inter/union if union > 0 else 0

# ─────────────────────────────────────────────────────────────────────────────
#  VISUALISASI SKELETON
# ─────────────────────────────────────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    (0,  1,  (255,128,  0)),
    (0,  2,  (  0,128,255)),
    (1,  3,  (255,128,  0)),
    (2,  4,  (  0,128,255)),
    (5,  6,  (  0,255,  0)),
    (5,  7,  (255,128,  0)),
    (7,  9,  (255,128,  0)),
    (6,  8,  (  0,128,255)),
    (8, 10,  (  0,128,255)),
    (5, 11,  (  0,255,128)),
    (6, 12,  (  0,255,128)),
    (11,12,  (  0,255,  0)),
    (11,13,  (255,128,  0)),
    (13,15,  (255,128,  0)),
    (12,14,  (  0,128,255)),
    (14,16,  (  0,128,255)),
]

def draw_skeleton(frame, keypoints, keypoint_scores, kpt_thr=0.3):
    for pid in range(len(keypoints)):
        kps   = keypoints[pid]
        score = keypoint_scores[pid]
        for (i, j, color) in SKELETON_CONNECTIONS:
            if score[i] >= kpt_thr and score[j] >= kpt_thr:
                x1,y1 = int(kps[i][0]), int(kps[i][1])
                x2,y2 = int(kps[j][0]), int(kps[j][1])
                if x1>0 and y1>0 and x2>0 and y2>0:
                    cv2.line(frame, (x1,y1), (x2,y2), color, 2, cv2.LINE_AA)
        for k in range(17):
            if score[k] >= kpt_thr:
                x,y = int(kps[k][0]), int(kps[k][1])
                if x>0 and y>0:
                    cv2.circle(frame, (x,y), 3, (255,255,255), -1, cv2.LINE_AA)
    return frame

def draw_label(frame, trk, label_str, score):
    x1,y1,x2,y2 = [int(v) for v in trk['bbox']]
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
        fc = (0,0,255) if label_str in BAHAYA else (255,255,255)
        cv2.putText(frame, text, loc, FONTFACE, FONTSCALE, fc, THICKNESS, LINETYPE)
    return frame

# ─────────────────────────────────────────────────────────────────────────────
#  FUNGSI INTI
# ─────────────────────────────────────────────────────────────────────────────
def process_frame(frame_bgr, model_pose, tracker, det_score_thr):
    """Proses satu frame: YOLO11-pose + BoT-SORT."""
    res = model_pose(frame_bgr, classes=[0], verbose=False)[0]

    if res.boxes is not None and len(res.boxes):
        boxes  = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        kps    = res.keypoints.xy.cpu().numpy()   if res.keypoints and res.keypoints.xy   is not None else np.zeros((0,17,2))
        kpconf = res.keypoints.conf.cpu().numpy() if res.keypoints and res.keypoints.conf is not None else np.zeros((0,17))
        mask   = scores >= det_score_thr
        boxes  = boxes[mask]; scores = scores[mask]
        kps    = kps[mask];   kpconf = kpconf[mask]
    else:
        boxes  = np.zeros((0,4)); scores = np.zeros((0,))
        kps    = np.zeros((0,17,2)); kpconf = np.zeros((0,17))

    dets   = np.hstack([boxes, scores[:,None], np.zeros((len(boxes),1))]) \
             if len(boxes) > 0 else np.empty((0,6))
    tracks = tracker.update(dets, frame_bgr)

    frame_tracks = []
    if len(tracks) > 0:
        for t in tracks:
            frame_tracks.append({'track_id': int(t[4]),
                                 'bbox':     [t[0],t[1],t[2],t[3]],
                                 'score':    float(t[5])})
    pose_frame = {'keypoints': kps, 'keypoint_scores': kpconf, 'bboxes': boxes}
    return frame_tracks, pose_frame


def run_inference(kp_buffer, track_buffer, model_posec3d,
                  label_map, h, w, action_score_thr, clip_len):
    """Jalankan PoseC3D pada buffer clip_len frame."""
    center_idx    = clip_len // 2
    center_tracks = track_buffer[center_idx]
    results       = {}

    for trk in center_tracks:
        tid            = trk['track_id']
        keypoint       = np.zeros((1, clip_len, 17, 2))
        keypoint_score = np.zeros((1, clip_len, 17))

        for j in range(clip_len):
            pr = kp_buffer[j]
            if len(pr['keypoints']) == 0:
                continue
            best_iou, best_idx = -1, -1
            for k, bbox in enumerate(pr['bboxes']):
                iou = cal_iou(trk['bbox'], bbox)
                if iou > best_iou:
                    best_iou, best_idx = iou, k
            if best_idx >= 0 and best_iou > 0.1:
                keypoint[0,j]       = pr['keypoints'][best_idx]
                keypoint_score[0,j] = pr['keypoint_scores'][best_idx]

        fake_anno = dict(
            frame_dir='', label=-1,
            img_shape=(h,w), original_shape=(h,w),
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
#  MODE KAMERA
# ─────────────────────────────────────────────────────────────────────────────
def run_camera(model_posec3d, model_pose, label_map):
    print(f'\n[MODE KAMERA] index={CAMERA_IDX} | Q=keluar')

    cap = cv2.VideoCapture(CAMERA_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

    if not cap.isOpened():
        print(f'ERROR: Kamera index {CAMERA_IDX} tidak bisa dibuka.')
        return
    
    # warm up kamera
    for _ in range(30):
        cap.read()

    tracker = BoTSORT(
        model_weights=BOTSORT_REID, device=DEVICE, fp16=True,
        track_high_thresh=DET_SCORE_THR, track_low_thresh=0.1,
        new_track_thresh=DET_SCORE_THR, track_buffer=50,
        match_thresh=0.8, proximity_thresh=0.5,
        appearance_thresh=0.25, with_reid=True,
    )

    kp_buffer      = []
    track_buffer   = []
    current_labels = {}
    fps_counter    = 0
    t_fps          = time.time()
    fps_display    = 0.0

    cv2.namedWindow('CIIS — YOLO11-pose (Q=keluar)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('CIIS — YOLO11-pose (Q=keluar)', 1280, 720)

    print('Kamera aktif. Tekan Q untuk keluar.')

    CAMERA_SAVE_OUT = '03_dataset/test_video/test_out/camera_out2_yolo11.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(CAMERA_SAVE_OUT, fourcc, 30.0, (CAMERA_W, CAMERA_H))
    print(f'Output kamera akan disimpan di: {CAMERA_SAVE_OUT}')

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print('Frame tidak terbaca.')
            break

        h, w = frame_bgr.shape[:2]

        frame_tracks, pose_frame = process_frame(
            frame_bgr, model_pose, tracker, DET_SCORE_THR)

        kp_buffer.append(pose_frame)
        track_buffer.append(frame_tracks)
        if len(kp_buffer) > PREDICT_STEPSIZE:
            kp_buffer.pop(0)
            track_buffer.pop(0)

        if len(kp_buffer) == PREDICT_STEPSIZE:
            pred = run_inference(
                kp_buffer, track_buffer, model_posec3d,
                label_map, h, w, ACTION_SCORE_THR, PREDICT_STEPSIZE)
            for tid, val in pred.items():
                current_labels[tid] = val

        # Overlay
        display = frame_bgr.copy()
        if len(pose_frame['keypoints']) > 0:
            display = draw_skeleton(
                display, pose_frame['keypoints'], pose_frame['keypoint_scores'])
        for trk in frame_tracks:
            tid      = trk['track_id']
            lb, sc   = current_labels.get(tid, (None, 0.0))
            display  = draw_label(display, trk, lb, sc)

        fps_counter += 1
        if fps_counter % 15 == 0:
            fps_display = 15 / (time.time() - t_fps)
            t_fps       = time.time()

        cv2.putText(display, f'FPS: {fps_display:.1f}',
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        # cv2.putText(display, f'Buffer: {len(kp_buffer)}/{PREDICT_STEPSIZE}',
        #             (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        # cv2.putText(display, 'YOLO11-pose + PoseC3D',
        #             (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        print(f'Frame shape: {display.shape}')
        print(f'Frame min/max: {display.min()}/{display.max()}')
        print(f'Tracks: {len(frame_tracks)} | Keypoints: {len(pose_frame["keypoints"])}')

        out_vid.write(display)

        cv2.imshow('CIIS — YOLO11-pose (Q=keluar)', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_vid.release()
    cv2.destroyAllWindows()
    torch.cuda.empty_cache()
    print('Kamera selesai.')

# ─────────────────────────────────────────────────────────────────────────────
#  MODE VIDEO
# ─────────────────────────────────────────────────────────────────────────────
def run_video(model_posec3d, model_pose, label_map, video_in, video_out):
    print(f'\n[MODE VIDEO]')
    print(f'  Input : {video_in}')
    print(f'  Output: {video_out}')

    # Step 1: Ekstrak frame
    print('\n[1/4] Ekstrak frame...')
    tmp_dir = tempfile.TemporaryDirectory()
    frame_paths, original_frames = frame_extract(video_in, 720, out_dir=tmp_dir.name)
    num_frame = len(frame_paths)
    h, w, _   = original_frames[0].shape
    del original_frames; gc.collect()
    print(f'      {num_frame} frame | {w}x{h}')

    # Step 2: YOLO11-pose + BoT-SORT semua frame
    print('\n[2/4] YOLO11-pose + BoT-SORT...')
    tracker = BoTSORT(
        model_weights=BOTSORT_REID, device=DEVICE, fp16=True,
        track_high_thresh=DET_SCORE_THR, track_low_thresh=0.1,
        new_track_thresh=DET_SCORE_THR, track_buffer=50,
        match_thresh=0.8, proximity_thresh=0.5,
        appearance_thresh=0.25, with_reid=True,
    )

    tracked_detections = []
    pose_results       = []
    t0                 = time.time()
    prog_bar           = mmengine.ProgressBar(num_frame)

    for frame_path in frame_paths:
        frame_bgr = cv2.imread(frame_path)
        ft, pf    = process_frame(frame_bgr, model_pose, tracker, DET_SCORE_THR)
        tracked_detections.append(ft)
        pose_results.append(pf)
        prog_bar.update()

    t_pose = time.time() - t0
    torch.cuda.empty_cache()
    print(f'\n      {t_pose:.1f}s ({t_pose/num_frame*1000:.1f}ms/frame)')

    # Step 3: PoseC3D inference
    print('\n[3/4] PoseC3D inference...')
    clip_len   = PREDICT_STEPSIZE
    timestamps = np.arange(clip_len//2, num_frame+1-clip_len//2, PREDICT_STEPSIZE)
    stdet_preds     = []
    inference_times = []
    t0              = time.time()
    prog_bar        = mmengine.ProgressBar(len(timestamps))

    for timestamp in timestamps:
        start  = timestamp - (clip_len//2 - 1)
        fi     = [max(0, min(int(start+j-1), num_frame-1)) for j in range(clip_len)]
        kp_buf = [pose_results[i]       for i in fi]
        tr_buf = [tracked_detections[i] for i in fi]

        t_inf = time.time()
        pred  = run_inference(kp_buf, tr_buf, model_posec3d,
                               label_map, h, w, ACTION_SCORE_THR, clip_len)
        inference_times.append((time.time()-t_inf)*1000)
        stdet_preds.append(pred)
        prog_bar.update()

    t_inf_total = time.time() - t0
    torch.cuda.empty_cache()

    total_time = t_pose + t_inf_total
    print(f'\n\n{"="*50}')
    print(f'  STATISTIK')
    print(f'{"="*50}')
    print(f'  Frame           : {num_frame}')
    print(f'  Pose estimation : {t_pose:.2f}s ({t_pose/num_frame*1000:.1f}ms/frame)')
    print(f'  PoseC3D         : {t_inf_total:.2f}s ({np.mean(inference_times):.1f}ms/clip avg)')
    print(f'  Total           : {total_time:.2f}s')
    print(f'  Effective FPS   : {num_frame/total_time:.2f}')
    print(f'{"="*50}')

    # Step 4: Render video output
    print('\n[4/4] Render video output...')
    anchor_anno_map = {int(ts): pred for ts, pred in zip(timestamps, stdet_preds)}

    vis_frames = []
    for ts in range(1, num_frame+1):
        frame_bgr = cv2.imread(frame_paths[min(ts-1, num_frame-1)])
        fi        = min(ts-1, len(pose_results)-1)

        # Skeleton
        pr = pose_results[fi]
        if len(pr.get('keypoints', [])) > 0:
            frame_bgr = draw_skeleton(
                frame_bgr, pr['keypoints'], pr['keypoint_scores'])

        # Bbox + label
        for trk in tracked_detections[fi]:
            tid      = trk['track_id']
            lb, sc   = None, 0.0
            for anchor_ts in sorted(anchor_anno_map.keys(), reverse=True):
                if anchor_ts <= ts and tid in anchor_anno_map[anchor_ts]:
                    lb, sc = anchor_anno_map[anchor_ts][tid]
                    if lb is not None:
                        break
            frame_bgr = draw_label(frame_bgr, trk, lb, sc)

        cv2.putText(frame_bgr, f'Frame:{ts}/{num_frame}',
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame_bgr, 'YOLO11-pose + PoseC3D',
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        vis_frames.append(mmcv.imconvert(frame_bgr, 'bgr', 'rgb'))

    os.makedirs(osp.dirname(video_out) if osp.dirname(video_out) else '.', exist_ok=True)
    vid = mpy.ImageSequenceClip(vis_frames, fps=OUTPUT_FPS)
    vid.write_videofile(video_out, logger=None)

    tmp_dir.cleanup()
    del vis_frames, pose_results, tracked_detections
    gc.collect(); torch.cuda.empty_cache()
    print(f'\n✓ Selesai: {video_out}')

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Inference YOLO11-pose + PoseC3D')
    parser.add_argument('--mode',   type=str, default='camera',
                        choices=['camera', 'video'],
                        help='Mode input: camera atau video')
    parser.add_argument('--input',  type=str, default=None,
                        help='Path video input (mode video)')
    parser.add_argument('--output', type=str, default=VIDEO_OUT_DEFAULT,
                        help='Path video output (mode video)')
    args = parser.parse_args()

    # Load model
    print('Loading model...')
    label_map     = load_label_map(LABEL_MAP_FILE)
    num_classes   = max(label_map.keys()) + 1
    skeleton_cfg  = mmengine.Config.fromfile(SKELETON_CONFIG)
    skeleton_cfg.model.cls_head.num_classes = num_classes
    model_posec3d = init_recognizer(skeleton_cfg, SKELETON_CHECKPOINT, DEVICE)
    model_pose    = YOLO(MODEL_POSE_PATH)
    print(f'✓ PoseC3D   : {SKELETON_CHECKPOINT}')
    print(f'✓ YOLO11    : {MODEL_POSE_PATH}')
    print(f'✓ Kelas ({num_classes}): {list(label_map.values())}')

    if args.mode == 'camera':
        run_camera(model_posec3d, model_pose, label_map)
    else:
        video_in = args.input
        if video_in is None:
            print('ERROR: --input harus diisi untuk mode video.')
            print('Contoh: python inference_yolo11.py --mode video --input input.mp4')
            return
        run_video(model_posec3d, model_pose, label_map, video_in, args.output)


if __name__ == '__main__':
    main()
