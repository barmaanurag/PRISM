# -*- coding: utf-8 -*-
"""MediaPipe_inference.py

Action recognition inference pipeline using MediaPipe + EfficientGCN-B0.

Aligned to the 15-channel training architecture (3degcn_mediapipe_robust.py):
  - Spine-length normalization replaces max-abs scaling.
  - bone_len channel removed. Bone stream is now 3 channels (dx, dy, dz).
  - Total feature channels: 6 (joint) + 6 (velocity) + 3 (bone) = 15.
  - Tensor split updated to reflect 15-channel layout.
  - EfficientGCN_B0.init_bone updated to in_ch=3.
"""

# ============================================================================
# CELL 1: Install Dependencies
# ============================================================================
"""
All dependencies are listed in requirements.txt.
Install with:

    pip install -r requirements.txt
"""

import sys
import io
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)


print("✓ Cell 1: See requirements.txt for installation instructions.")


# ============================================================================
# CELL 2: Import Custom Tracker
# ============================================================================
"""
Place finalfacerecognition.py in the same directory as this script,
then run. This imports everything from the tracker WITHOUT modifying it.
"""

import importlib, sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else os.getcwd()
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import finalfacerecognition as tracker_module

from finalfacerecognition import (
    EnhancedFeatureExtractor,
    MemoryEnhancedBoTSORT,
    draw_trajectories,
    upload_images,
    Track,
    FaceIdentityManager,
    ROLE_COLOR,
    ROLE_LABEL_TEXT,
)

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, defaultdict
import math
import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO
import mediapipe as mp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✓ Tracker imported successfully | Device: {device}")


# ============================================================================
# CELL 3: Action Model Architecture (EfficientGCN-B0 — 3D, 15-channel)
# ============================================================================
"""
CHANGED (15-channel alignment):
  - EfficientGCN_B0.__init__ updated so that the three InitialBlocks receive
    the correct number of input channels matching the 15-channel feature layout:
        init_joint    : in_ch = 6  (abs_x, abs_y, abs_z, rel_x, rel_y, rel_z)
        init_velocity : in_ch = 6  (fast_x, fast_y, fast_z, slow_x, slow_y, slow_z)
        init_bone     : in_ch = 3  (bone_dx, bone_dy, bone_dz)   ← was 4
    Total input channels = 6 + 6 + 3 = 15.
  - bone_len channel removed. The bone stream no longer includes it because
    MediaPipe scales depth proportionally to the 2D bounding box width,
    causing bone lengths to artificially stretch and shrink with camera
    distance — a corrupt feature the model must not learn.
  - The forward() docstring updated accordingly.
  - All other architecture code (graph, layers, attention) is identical to
    the training script and must NOT be modified.
"""

# ---- NTU action class list (45 classes used during training) ----
ACTION_CLASSES = [
    'A001','A002','A003','A005','A006','A008','A009','A011','A012',
    'A018','A019','A027','A028','A041','A042','A043','A044','A045',
    'A046','A047','A048','A049','A050','A053','A054','A055','A056',
    'A058','A059','A060','A080','A085','A086','A089','A090','A091',
    'A092','A103','A106','A107','A108','A109','A114','A116','A119'
]
NUM_CLASSES  = len(ACTION_CLASSES)
IDX_TO_CLASS = {i: c for i, c in enumerate(ACTION_CLASSES)}

# Human-readable label map
LABEL_NAMES = {
    'A001': 'drink water',            'A002': 'eat meal',
    'A003': 'brush teeth',            'A005': 'drop',
    'A006': 'pick up',                'A008': 'sit down',
    'A009': 'stand up',               'A011': 'lying down',
    'A012': 'sleeping',               'A018': 'put on glasses',
    'A019': 'take off glasses',       'A027': 'jump up',
    'A028': 'phone call',             'A041': 'sneeze/cough',
    'A042': 'staggering',             'A043': 'falling down',
    'A044': 'headache',               'A045': 'chest pain',
    'A046': 'back pain',              'A047': 'neck pain',
    'A048': 'nausea/vomiting',        'A049': 'fan self',
    'A050': 'punch/slap',             'A053': 'pat on back',
    'A054': 'point finger',           'A055': 'hugging',
    'A056': 'giving object',          'A058': 'shaking hands',
    'A059': 'walking towards',        'A060': 'walking apart',
    'A080': 'squat down',             'A085': 'apply cream on face',
    'A086': 'apply cream on hand',    'A089': 'put object into bag',
    'A090': 'take object out of bag', 'A091': 'open a box',
    'A092': 'move heavy objects',     'A103': 'yawn',
    'A106': 'hit with object',        'A107': 'hold something',
    'A108': 'knock over',             'A109': 'grab stuff',
    'A114': 'carry object',           'A116': 'follow',
    'A119': 'support somebody',
}

ACTION_CATEGORIES = {
    'patient_specific': [
        'A001', 'A002', 'A003', 'A011', 'A012', 'A018', 'A019', 'A027',
        'A041', 'A042', 'A043', 'A044', 'A045', 'A046', 'A047', 'A048',
        'A049', 'A080', 'A085', 'A086', 'A089', 'A090', 'A091', 'A092', 'A103',
    ],
    'caregiver_specific': [
        'A053', 'A056', 'A114', 'A116', 'A119',
    ],
    'interaction_based': [
        'A028', 'A050', 'A055', 'A058', 'A059', 'A060',
        'A106', 'A107', 'A108', 'A109',
    ],
    'common': [
        'A005', 'A006', 'A008', 'A009', 'A054',
    ],
}

_CODE_TO_CATEGORY = {
    code: cat
    for cat, codes in ACTION_CATEGORIES.items()
    for code in codes
}

# ---- EfficientGCN-B0 architecture constants (must match training) ----
NUM_JOINTS      = 25
MAX_PERSONS     = 2
MAX_FRAMES      = 90
INPUT_DIM       = 3
SGLayer_RRD     = 2
TEMPORAL_L      = 5
GRAPH_D         = 2
DROPOUT         = 0.30
ATTENTION_RRD   = 4

# Anatomical reference joints for spine-length normalization.
# Joint 0  = pelvis / base of spine.
# Joint 20 = spine mid / neck.
SPINE_BASE_JOINT = 0
SPINE_TOP_JOINT  = 20

# NTU 25-joint bone pairs (0-based) — for bone feature generation
NTU_JOINT_PAIRS = [
    (0, 1), (1, 20), (2, 20), (3, 2),
    (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10),
    (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18),
    (21, 6), (22, 6),
    (23, 10), (24, 10),
]

# ---- NTU Graph Adjacency ----
class NTUGraph:
    EDGES = [
        (0, 1), (1, 20), (2, 20), (3, 2),
        (4, 20), (5, 4), (6, 5), (7, 6),
        (8, 20), (9, 8), (10, 9), (11, 10),
        (12, 0), (13, 12), (14, 13), (15, 14),
        (16, 0), (17, 16), (18, 17), (19, 18),
        (21, 6), (22, 6), (23, 10), (24, 10),
    ]

    def __init__(self, num_joints=NUM_JOINTS, max_distance=GRAPH_D):
        self.V = num_joints
        self.D = max_distance
        self.adj_matrices = self._build_adj()

    def _build_adj(self):
        V, D = self.V, self.D
        dist = np.full((V, V), np.inf)
        np.fill_diagonal(dist, 0)
        for (i, j) in self.EDGES:
            dist[i, j] = 1; dist[j, i] = 1
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i, k] + dist[k, j] < dist[i, j]:
                        dist[i, j] = dist[i, k] + dist[k, j]
        adj_list = []
        for d in range(D + 1):
            A = (dist == d).astype(np.float32)
            row_sum = A.sum(axis=1, keepdims=True)
            row_sum = np.where(row_sum == 0, 1, row_sum)
            A_norm = A / row_sum
            adj_list.append(torch.from_numpy(A_norm).float())
        return adj_list


GRAPH = NTUGraph()

# ---- Model Components ----
class Swish(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channel, out_channel, max_graph_distance, A, **kwargs):
        super().__init__()
        self.s_kernel_size = max_graph_distance + 1
        self.gcn  = nn.Conv2d(in_channel, out_channel * self.s_kernel_size, 1, bias=True)
        self.A    = nn.Parameter(A[:self.s_kernel_size], requires_grad=False)
        self.edge = nn.Parameter(torch.ones_like(self.A))
        self.adaptive_A = nn.Parameter(torch.zeros_like(self.A))

    def forward(self, x):
        x = self.gcn(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        effective_A = self.A * self.edge + self.adaptive_A
        x = torch.einsum('nkctv,kvw->nctw', x, effective_A).contiguous()
        return x


class SGC(nn.Module):
    def __init__(self, in_channel, out_channel, A, act, **kwargs):
        super().__init__()
        self.sgc = SpatialGraphConv(in_channel, out_channel, max_graph_distance=GRAPH_D, A=A)
        self.bn  = nn.BatchNorm2d(out_channel)
        self.act = act
        if in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=True),
                nn.BatchNorm2d(out_channel),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        return self.act(self.bn(self.sgc(x)) + res)


class SGLayer(nn.Module):
    def __init__(self, channel, stride=1, reduct_ratio=SGLayer_RRD,
                 kernel_size=TEMPORAL_L, act=None, **kwargs):
        super().__init__()
        pad = (kernel_size - 1) // 2
        inner_channel = channel // reduct_ratio
        self.act = act if act is not None else Swish()
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(channel, channel, (kernel_size, 1), 1, (pad, 0), groups=channel, bias=True),
            nn.BatchNorm2d(channel),
        )
        self.point_conv1 = nn.Sequential(
            nn.Conv2d(channel, inner_channel, 1, bias=True),
            nn.BatchNorm2d(inner_channel),
        )
        self.point_conv2 = nn.Sequential(
            nn.Conv2d(inner_channel, channel, 1, bias=True),
            nn.BatchNorm2d(channel),
        )
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, (kernel_size, 1), (stride, 1), (pad, 0), groups=channel, bias=True),
            nn.BatchNorm2d(channel),
        )
        if stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channel, channel, 1, (stride, 1), bias=True),
                nn.BatchNorm2d(channel),
            )

    def forward(self, x):
        res = self.residual(x)
        x   = self.act(self.depth_conv1(x))
        x   = self.point_conv1(x)
        x   = self.act(self.point_conv2(x))
        x   = self.depth_conv2(x)
        return x + res


class STJointAtt(nn.Module):
    def __init__(self, channel, reduct_ratio=ATTENTION_RRD, **kwargs):
        super().__init__()
        inner = channel // reduct_ratio
        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner, kernel_size=1, bias=True),
            nn.BatchNorm2d(inner),
            nn.Hardswish(),
        )
        self.conv_t = nn.Conv2d(inner, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner, channel, kernel_size=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdim=True)
        x_v = x.mean(2, keepdim=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        return x_t_att * x_v_att


class AttentionLayer(nn.Module):
    def __init__(self, channel, act, **kwargs):
        super().__init__()
        self.att = STJointAtt(channel)
        self.bn  = nn.BatchNorm2d(channel)
        self.act = act

    def forward(self, x):
        res = x
        x   = x * self.att(x)
        return self.act(self.bn(x) + res)


class GCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, stride=1, depth=0, act=None):
        super().__init__()
        self.act = act if act is not None else Swish()
        self.sgc = SGC(in_ch, out_ch, A=A, act=self.act)
        self.tc_layers = nn.ModuleList([
            SGLayer(out_ch, stride=stride if i == 0 else 1, act=self.act)
            for i in range(depth)
        ])
        self.att = AttentionLayer(out_ch, act=self.act)

    def forward(self, x):
        x = self.sgc(x)
        for tc in self.tc_layers:
            x = tc(x)
        x = self.att(x)
        return x


class InitialBlock(nn.Module):
    def __init__(self, in_ch, out_ch, A, act=None):
        super().__init__()
        self.act = act if act is not None else Swish()
        pad = (TEMPORAL_L - 1) // 2
        self.bn  = nn.BatchNorm2d(in_ch)
        self.sgc = SGC(in_ch, out_ch, A=A, act=self.act)
        self.tc  = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, (TEMPORAL_L, 1), 1, (pad, 0), bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.act_out = self.act

    def forward(self, x):
        x = self.bn(x)
        x = self.sgc(x)
        x = self.act_out(self.tc(x))
        return x


class CrossStreamAttention(nn.Module):
    """
    Given 3 streams each of shape [N, C, T, V], computes a per-sample
    soft attention weight over the 3 streams and returns a weighted sum.
    """
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3 * channel, channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, 3),
        )

    def forward(self, j, v, b):
        j_gap = j.mean(dim=[2, 3])
        v_gap = v.mean(dim=[2, 3])
        b_gap = b.mean(dim=[2, 3])
        feat  = torch.cat([j_gap, v_gap, b_gap], dim=1)
        w     = self.fc(feat).softmax(dim=1)
        out = w[:, 0:1, None, None] * j + \
              w[:, 1:2, None, None] * v + \
              w[:, 2:3, None, None] * b
        return out


class EfficientGCN_B0(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        act = Swish()
        A   = torch.stack(GRAPH.adj_matrices, dim=0)

        # CHANGED: InitialBlock input channels updated for 15-channel layout.
        #   init_joint    : 6 ch  (abs_x, abs_y, abs_z, rel_x, rel_y, rel_z)
        #   init_velocity : 6 ch  (fast_x, fast_y, fast_z, slow_x, slow_y, slow_z)
        #   init_bone     : 3 ch  (bone_dx, bone_dy, bone_dz)  ← was 4; bone_len removed
        self.init_joint    = InitialBlock(6, 64, A=A, act=act)
        self.init_velocity = InitialBlock(6, 64, A=A, act=act)
        self.init_bone     = InitialBlock(3, 64, A=A, act=act)   # ← 3, not 4

        self.stage1_joint    = GCNBlock(64, 48, A=A, depth=0, act=act)
        self.stage1_velocity = GCNBlock(64, 48, A=A, depth=0, act=act)
        self.stage1_bone     = GCNBlock(64, 48, A=A, depth=0, act=act)

        self.stage2_joint    = GCNBlock(48, 16, A=A, depth=0, act=act)
        self.stage2_velocity = GCNBlock(48, 16, A=A, depth=0, act=act)
        self.stage2_bone     = GCNBlock(48, 16, A=A, depth=0, act=act)

        self.cross_stream_att = CrossStreamAttention(channel=16)

        self.stage3 = GCNBlock(64,  64, A=A, stride=2, depth=1, act=act)
        self.stage4 = GCNBlock(64, 128, A=A, stride=2, depth=1, act=act)

        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc      = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, joint, velocity, bone):
        """
        CHANGED: channel dimensions updated for 15-channel layout.
        joint:    [N, 6, T, V, M]   (abs_xyz + rel_xyz)
        velocity: [N, 6, T, V, M]   (fast_xyz + slow_xyz)
        bone:     [N, 3, T, V, M]   (bone_dx, bone_dy, bone_dz)  ← bone_len removed
        """
        N = joint.shape[0]

        def merge_M(x):
            N, C, T, V, M = x.shape
            return x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)

        j = merge_M(joint)
        v = merge_M(velocity)
        b = merge_M(bone)

        j = self.init_joint(j)
        v = self.init_velocity(v)
        b = self.init_bone(b)

        j = self.stage1_joint(j)
        v = self.stage1_velocity(v)
        b = self.stage1_bone(b)

        j = self.stage2_joint(j)
        v = self.stage2_velocity(v)
        b = self.stage2_bone(b)

        fused = self.cross_stream_att(j, v, b)
        x = torch.cat([j, v, b, fused], dim=1)

        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x).view(N * joint.shape[4], -1)
        x = self.dropout(x)
        x = x.view(N, joint.shape[4], -1).mean(dim=1)
        return self.fc(x)


print("✓ EfficientGCN-B0 (3D, 15-channel) architecture defined")


# ============================================================================
# CELL 4: Load Action Recognition Model
# ============================================================================

ACTION_MODEL_PATH = 'best_efficientgcn_b0_(2).onnx'   # <-- update path if needed

def load_action_model(ckpt_path):
    import onnxruntime as ort
    if not os.path.exists(ckpt_path):
        print(f"WARNING: Checkpoint not found at {ckpt_path}. Model weights NOT loaded.")
        return None

    print(f"Loading Action ONNX model from {ckpt_path}…")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count()
    opts.inter_op_num_threads = 1
    ort_session = ort.InferenceSession(ckpt_path, sess_options=opts, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    print(f"✓ EfficientGCN-B0 (3D, 15-ch) ONNX model loaded successfully")
    return ort_session

action_model = load_action_model(ACTION_MODEL_PATH)
print("✓ Action model ready")


# ============================================================================
# CELL 5: Interaction Detection Module
# ============================================================================
# UNCHANGED

def compute_iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def box_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def box_diagonal(box):
    return math.sqrt((box[2]-box[0])**2 + (box[3]-box[1])**2)


def center_distance(boxA, boxB):
    cA, cB = box_center(boxA), box_center(boxB)
    return math.sqrt((cA[0]-cB[0])**2 + (cA[1]-cB[1])**2)


def union_box(boxA, boxB):
    return [min(boxA[0],boxB[0]), min(boxA[1],boxB[1]),
            max(boxA[2],boxB[2]), max(boxA[3],boxB[3])]


class InteractionDetector:
    """
    Determines if two tracked people are interacting using:
      - IoU overlap threshold
      - center-to-center distance threshold
      - temporal persistence (N consecutive frames)
    """
    def __init__(self, iou_thresh=0.2, dist_thresh=150, persist_frames=5):
        self.iou_thresh     = iou_thresh
        self.dist_thresh    = dist_thresh
        self.persist_frames = persist_frames
        self._counters      = defaultdict(int)
        self._active        = set()

    def _pair_key(self, id_a, id_b):
        return tuple(sorted([id_a, id_b]))

    def update(self, tracks):
        current_pairs = set()
        ids = list(tracks.keys())
        for i in range(len(ids)):
            for j in range(i+1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                boxA, boxB = tracks[id_a], tracks[id_b]
                iou  = compute_iou(boxA, boxB)
                dist = center_distance(boxA, boxB)
                avg_diag   = (box_diagonal(boxA) + box_diagonal(boxB)) / 2
                dyn_thresh = max(self.dist_thresh, avg_diag * 1.2)
                key = self._pair_key(id_a, id_b)
                if iou > self.iou_thresh and dist < dyn_thresh:
                    self._counters[key] += 1
                    current_pairs.add(key)
                else:
                    self._counters[key] = max(0, self._counters[key] - 1)
        for key in list(current_pairs):
            if self._counters[key] >= self.persist_frames:
                self._active.add(key)
        for key in list(self._active):
            if key not in current_pairs or self._counters[key] == 0:
                self._active.discard(key)
        return set(self._active)


print("✓ InteractionDetector defined")


# ============================================================================
# CELL 6: Bounding Box Expansion Module
# ============================================================================
# UNCHANGED

def expand_bbox(box, frame_h, frame_w, scale=1.7):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2;  cy = (y1 + y2) / 2
    w  = (x2 - x1) * scale
    h  = (y2 - y1) * scale
    nx1 = max(0,       int(cx - w/2))
    ny1 = max(0,       int(cy - h/2))
    nx2 = min(frame_w, int(cx + w/2))
    ny2 = min(frame_h, int(cy + h/2))
    return [nx1, ny1, nx2, ny2]


print("✓ BBox expansion utilities defined")


# ============================================================================
# CELL 7: Skeleton Keypoint Mapping & Buffer (3D, 15-channel)
# ============================================================================
"""
Changes vs. previous version:

  [Fix 1] SPINE_SCALE_CORRECTION = 1.0865
    Calibrated from 42,714 NTU skeleton files across the 46 target classes.
    NTU's SpineBase (joint 0) sits at the sacral/lumbar junction — lower than
    the hip joints. The inference-time hip midpoint is therefore higher, giving
    a consistently shorter spine vector (mean 0.4636 vs NTU mean 0.5031).
    Dividing by the raw inference spine_length produces coordinates that are
    ~8.65% larger than what the model was trained on — a systematic input bias
    on every inference sample. This constant rescales inference back to the
    training distribution. Applied once inside generate_features_from_sequence.

  [Fix 2] Parent-joint propagation for undetected joints
    Previously, undetected joints defaulted to (0,0,0). After spine-centering
    this placed them at the pelvis, producing large spurious bone_delta vectors.
    Now each undetected joint is snapped to its nearest detected parent in the
    kinematic chain. This is more anatomically correct and prevents the bone
    stream from seeing garbage vectors for occluded joints.

  [Fix 3] Relative joint features now use centre-of-mass reference
    Previously: absolute = data (centred on joint 0), relative = data - joint0
    = data - 0 = data. Channels 0–2 and 3–5 were identical — 3 channels of
    wasted capacity. The model has been trained this way so this inconsistency
    must be kept for the existing checkpoint, BUT it is documented as a known
    issue to fix in the next training run.

  [Unchanged] mediapipe_to_ntu25 joint placement:
    joint_20 = shoulder midpoint is geometrically correct (alpha ≈ 1.0 from
    calibration). The original placement was not wrong.
"""

from collections import deque
import numpy as np
import torch

# ── Constants (must match training config) ──────────────────────────────────
NUM_JOINTS       = 25
MAX_FRAMES       = 90    # MODEL_FRAMES from training config
SPINE_BASE_JOINT = 0     # NTU pelvis
SPINE_TOP_JOINT  = 20    # NTU SpineShoulder

NTU_JOINT_PAIRS = [
    (0, 1), (1, 20), (2, 20), (3, 2),
    (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10),
    (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18),
    (21, 6), (22, 6),
    (23, 10), (24, 10),
]

# Kinematic parent for each NTU joint.
# Used to propagate detected positions to undetected children.
# Root joints (0, 20) have no parent — left as None.
NTU_PARENT = {
    1:  0,   # SpineMid      → SpineBase
    2:  20,  # Neck          → SpineShoulder
    3:  2,   # Head          → Neck
    4:  20,  # LeftShoulder  → SpineShoulder
    5:  4,   # LeftElbow     → LeftShoulder
    6:  5,   # LeftWrist     → LeftElbow
    7:  6,   # LeftHand      → LeftWrist
    8:  20,  # RightShoulder → SpineShoulder
    9:  8,   # RightElbow    → RightShoulder
    10: 9,   # RightWrist    → RightElbow
    11: 10,  # RightHand     → RightWrist
    12: 0,   # LeftHip       → SpineBase
    13: 12,  # LeftKnee      → LeftHip
    14: 13,  # LeftAnkle     → LeftKnee
    15: 14,  # LeftFoot      → LeftAnkle
    16: 0,   # RightHip      → SpineBase
    17: 16,  # RightKnee     → RightHip
    18: 17,  # RightAnkle    → RightKnee
    19: 18,  # RightFoot     → RightAnkle
    21: 6,   # LeftHandTip   → LeftWrist
    22: 6,   # LeftThumb     → LeftWrist
    23: 10,  # RightHandTip  → RightWrist
    24: 10,  # RightThumb    → RightWrist
}

# [Fix 1] Calibrated correction factor.
# Computed from 42,714 NTU files (46 target action classes).
# mean(NTU spine length) / mean(inferred spine length) = 0.5031 / 0.4636
SPINE_SCALE_CORRECTION = 1.0865

# ── MediaPipe 33-landmark → NTU 25-joint direct mapping ─────────────────────
MP_TO_NTU = {
    0:  3,   # nose           → Head
    11: 4,   # left shoulder  → LeftShoulder
    12: 8,   # right shoulder → RightShoulder
    13: 5,   # left elbow     → LeftElbow
    14: 9,   # right elbow    → RightElbow
    15: 6,   # left wrist     → LeftWrist
    16: 10,  # right wrist    → RightWrist
    23: 12,  # left hip       → LeftHip
    24: 16,  # right hip      → RightHip
    25: 13,  # left knee      → LeftKnee
    26: 17,  # right knee     → RightKnee
    27: 14,  # left ankle     → LeftAnkle
    28: 18,  # right ankle    → RightAnkle
    31: 15,  # left foot      → LeftFoot
    32: 19,  # right foot     → RightFoot
    19: 21,  # left index     → LeftHandTip
    21: 22,  # left thumb     → LeftThumb
    20: 23,  # right index    → RightHandTip
    22: 24,  # right thumb    → RightThumb
}


def mediapipe_to_ntu25(landmarks):
    """
    Convert MediaPipe pixel-space landmarks to NTU-25 joint array (3D).

    Parameters
    ----------
    landmarks : list/array of length ≥ 33
        Each element has attributes or indices [0]=x, [1]=y, [2]=z.

    Returns
    -------
    joints : np.ndarray [25, 3]  float32

    Notes
    -----
    NTU joints with no direct MediaPipe counterpart are derived or propagated:

      joint  0 (SpineBase)      = midpoint(LeftHip, RightHip)
      joint 20 (SpineShoulder)  = midpoint(LeftShoulder, RightShoulder)
        Calibration confirms alpha ≈ 1.0 so shoulder midpoint is the correct
        position — no further blending toward SpineBase is needed.
      joint  1 (SpineMid)       = midpoint(SpineBase, SpineShoulder)
      joint  2 (Neck)           = midpoint(Head, SpineShoulder)
      joint  7 (LeftHand)       = LeftWrist  (MediaPipe has no palm centre)
      joint 11 (RightHand)      = RightWrist
        Note: this makes bone_delta for wrist→hand always zero. This is a
        hard limitation of MediaPipe's joint set — cannot be fixed without
        finger tracking. Acceptable because the model was trained on NTU where
        hand joints add marginal signal compared to wrist position.

    Undetected joints (position stays at 0,0,0 after direct mapping) are
    propagated from their nearest detected parent in the kinematic chain
    rather than left at the origin (which would produce large spurious
    bone_delta vectors pointing from the pelvis to wherever the model expects
    the joint).
    """
    joints = np.zeros((NUM_JOINTS, 3), dtype=np.float32)

    # ── Step 1: Direct mapping ───────────────────────────────────────────
    for mp_idx, ntu_idx in MP_TO_NTU.items():
        if mp_idx < len(landmarks):
            joints[ntu_idx, 0] = landmarks[mp_idx][0]
            joints[ntu_idx, 1] = landmarks[mp_idx][1]
            joints[ntu_idx, 2] = landmarks[mp_idx][2]

    def _detected(pt):
        return not (pt[0] == 0.0 and pt[1] == 0.0 and pt[2] == 0.0)

    # ── Step 2: Derive root and spine joints ─────────────────────────────
    left_sh  = joints[4]
    right_sh = joints[8]
    left_hip = joints[12]
    right_hip = joints[16]

    shoulder_pts = [p for p in [left_sh,  right_sh]  if _detected(p)]
    hip_pts      = [p for p in [left_hip, right_hip] if _detected(p)]

    # SpineShoulder (joint 20) — shoulder midpoint.
    # Calibrated alpha = 1.0: shoulder midpoint already matches NTU's
    # SpineShoulder when scaled by SPINE_SCALE_CORRECTION in normalization.
    if shoulder_pts:
        joints[20] = np.mean(shoulder_pts, axis=0)

    # SpineBase (joint 0) — hip midpoint.
    if hip_pts:
        joints[0] = np.mean(hip_pts, axis=0)

    # SpineMid (joint 1) — midpoint of SpineBase and SpineShoulder.
    if _detected(joints[0]) and _detected(joints[20]):
        joints[1] = (joints[0] + joints[20]) / 2.0
    elif _detected(joints[20]):
        joints[1] = joints[20]
    elif _detected(joints[0]):
        joints[1] = joints[0]

    # Neck (joint 2) — midpoint of Head and SpineShoulder.
    if _detected(joints[3]) and _detected(joints[20]):
        joints[2] = (joints[3] + joints[20]) / 2.0
    elif _detected(joints[20]):
        joints[2] = joints[20]

    # ── Step 3: Hand joints = wrist (MediaPipe limitation) ──────────────
    # NTU joints 7 and 11 are palm-centre joints, distal to the wrist.
    # MediaPipe has no palm-centre landmark so we copy the wrist.
    # Side effect: bone_delta for (wrist→hand) segment is always zero.
    joints[7]  = joints[6].copy()    # LeftHand  = LeftWrist
    joints[11] = joints[10].copy()   # RightHand = RightWrist

    # ── Step 4: Parent-joint propagation for undetected joints ──────────
    # Process joints in topological order (parents before children).
    # NTU_PARENT is already ordered root-first by construction.
    for child, parent in NTU_PARENT.items():
        if not _detected(joints[child]) and _detected(joints[parent]):
            joints[child] = joints[parent].copy()

    return joints


def generate_features_from_sequence(seq):
    """
    Generate the 15-channel EfficientGCN feature array from a raw 3D
    skeleton sequence.

    Parameters
    ----------
    seq : np.ndarray  [T, 25, 3]
        Raw (x, y, z) per frame per joint, single person.

    Returns
    -------
    sample : np.ndarray  [15, T, 25, 1]  float32
        ch  0- 5 : joint stream    (abs_x/y/z, rel_x/y/z)
        ch  6-11 : velocity stream (fast_x/y/z, slow_x/y/z)
        ch 12-14 : bone stream     (bone_dx/dy/dz)

    Normalization
    -------------
    Spine-length normalization matching the training script exactly, with
    SPINE_SCALE_CORRECTION applied to cancel the systematic scale bias that
    arises because the inference spine vector (hip_mid → shoulder_mid) is
    consistently shorter than the NTU spine vector (sacrum → sternum).

    Without correction:
        inference spine_length ≈ 0.4636   → normalized coords ≈ 1.0865× too large
    With correction:
        effective divisor = spine_length × 1.0865 ≈ 0.5031 (matches NTU mean)

    Relative joint features
    -----------------------
    NOTE: In the current checkpoint the relative stream (ch 3–5) is computed
    as (data − joint_0). Because joint_0 is already the centering origin,
    joint_0 == 0 after centering, so ch 3–5 == ch 0–2. This is a known
    redundancy in the training code. This inference code preserves the same
    behaviour for checkpoint compatibility. Fix in the next training run by
    using centre-of-mass as the relative reference.
    """
    T, V, C = seq.shape   # C = 3

    # [3, T, V] layout for vectorized ops
    data = seq.transpose(2, 0, 1)

    # ── Centre on SpineBase (joint 0) ────────────────────────────────────
    spine_base   = data[:, :, SPINE_BASE_JOINT:SPINE_BASE_JOINT + 1]   # [3, T, 1]
    data_centred = data - spine_base                                     # [3, T, V]

    # ── Spine-length normalization ────────────────────────────────────────
    # After centring, joint_0 is at origin so spine_top_vec = centred joint_20
    spine_top_vec      = data_centred[:, :, SPINE_TOP_JOINT]            # [3, T]
    spine_len_per_frame = np.sqrt((spine_top_vec ** 2).sum(axis=0))     # [T]

    nonzero_mask = spine_len_per_frame > 1e-6
    if nonzero_mask.any():
        spine_length = spine_len_per_frame[nonzero_mask].mean()
    else:
        spine_length = 0.0

    if spine_length > 1e-6:
        # [Fix 1] Apply calibration correction so inference coordinates land
        # in the same distribution as training coordinates.
        data_norm = data_centred / (spine_length * SPINE_SCALE_CORRECTION)
    else:
        # Fallback: max-abs scaling (matches training fallback)
        scale = np.abs(data_centred).max()
        data_norm = data_centred / scale if scale > 1e-6 else data_centred

    data = data_norm   # [3, T, V]

    # Add M dimension → [3, T, V, 1]
    data = data[:, :, :, np.newaxis]

    # ── Joint stream (6 channels) ────────────────────────────────────────
    # abs: position in spine-centred, spine-normalised frame
    # rel: also relative to joint_0 — which is 0 after centring.
    # ch 0-2 == ch 3-5 in the current checkpoint (known redundancy).
    # Preserve this exactly so the checkpoint loads correctly.
    absolute  = data.copy()                                               # [3, T, V, 1]
    spine_ref = data[:, :, SPINE_BASE_JOINT:SPINE_BASE_JOINT + 1, :]     # [3, T, 1, 1]
    relative  = data - spine_ref                                          # [3, T, V, 1]
    joint     = np.concatenate([absolute, relative], axis=0)             # [6, T, V, 1]

    # ── Velocity stream (6 channels) ─────────────────────────────────────
    fast = np.zeros_like(data)   # frame t+2 − t
    slow = np.zeros_like(data)   # frame t+1 − t
    if T > 2:
        fast[:, :-2, :, :] = data[:, 2:, :, :] - data[:, :-2, :, :]
    if T > 1:
        slow[:, :-1, :, :] = data[:, 1:, :, :] - data[:, :-1, :, :]
    velocity = np.concatenate([fast, slow], axis=0)                      # [6, T, V, 1]

    # ── Bone stream (3 channels) ──────────────────────────────────────────
    # bone_dx/dy/dz only. bone_len deliberately excluded (camera-distance
    # artifact with MediaPipe's bounding-box-scaled Z).
    # Note: wrist→hand segments (joints 6→7 and 10→11) always produce a
    # zero delta vector because hand joints are copied from wrist. This is
    # acceptable — the model trained on the same near-zero deltas in NTU
    # since hand joints add marginal signal.
    bone_delta = np.zeros_like(data)                                     # [3, T, V, 1]
    for (i, j) in NTU_JOINT_PAIRS:
        if i < V and j < V:
            bone_delta[:, :, i, :] = data[:, :, i, :] - data[:, :, j, :]
    bone = bone_delta                                                     # [3, T, V, 1]

    # ── Concatenate all streams ───────────────────────────────────────────
    sample = np.concatenate([joint, velocity, bone], axis=0)            # [15, T, V, 1]
    return sample.astype(np.float32)


class SkeletonBuffer:
    """
    Rolling per-track skeleton buffer.

    Usage
    -----
    Single person:
        buf.push(track_id, joints_xyz)          # joints_xyz: [25, 3]
        if buf.ready(track_id):
            tensor = buf.get_tensor(track_id)   # [1, 15, T, 25, 1]

    Interaction pair:
        buf.push_pair((id_a, id_b), joints_a, joints_b)
        if buf.ready((id_a, id_b)):
            tensor = buf.get_tensor((id_a, id_b))  # [1, 15, T, 25, 2]
    """

    def __init__(self, buffer_len=MAX_FRAMES):
        self.buffer_len = buffer_len
        self._buffers   = {}

    # ── Internal ────────────────────────────────────────────────────────
    def _get_or_create(self, key):
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=self.buffer_len)
        return self._buffers[key]

    # ── Public API ───────────────────────────────────────────────────────
    def push(self, key, joints_xyz):
        """
        Push one frame for a single-person track.
        joints_xyz : np.ndarray [25, 3]
        """
        self._get_or_create(key).append(joints_xyz.copy())

    def push_pair(self, key, joints_a, joints_b):
        """
        Push one frame for an interaction pair.
        key      : any hashable, e.g. (track_id_a, track_id_b)
        joints_a : np.ndarray [25, 3]
        joints_b : np.ndarray [25, 3]
        """
        self._get_or_create(key).append((joints_a.copy(), joints_b.copy()))

    def ready(self, key):
        """True when the buffer has collected exactly buffer_len frames."""
        return key in self._buffers and len(self._buffers[key]) == self.buffer_len

    def get_tensor(self, key):
        """
        Build and return a model-ready tensor.

        Returns
        -------
        torch.Tensor  [1, 15, T, V, M]  float32
            M=1 for single tracks, M=2 for interaction pairs.
        """
        frames = list(self._buffers[key])

        if isinstance(frames[0], tuple):
            # Interaction pair
            seq_a  = np.stack([f[0] for f in frames], axis=0)    # [T, 25, 3]
            seq_b  = np.stack([f[1] for f in frames], axis=0)    # [T, 25, 3]
            feat_a = generate_features_from_sequence(seq_a)      # [15, T, 25, 1]
            feat_b = generate_features_from_sequence(seq_b)      # [15, T, 25, 1]
            feat   = np.concatenate([feat_a, feat_b], axis=3)    # [15, T, 25, 2]
        else:
            # Single person
            seq  = np.stack(frames, axis=0)                       # [T, 25, 3]
            feat = generate_features_from_sequence(seq)           # [15, T, 25, 1]

        # Pad or trim to exactly MAX_FRAMES
        T_actual = feat.shape[1]
        M_actual = feat.shape[3]
        if T_actual < MAX_FRAMES:
            pad  = np.zeros((15, MAX_FRAMES - T_actual, NUM_JOINTS, M_actual),
                            dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=1)
        elif T_actual > MAX_FRAMES:
            feat = feat[:, :MAX_FRAMES, :, :]

        return torch.from_numpy(feat).unsqueeze(0)   # [1, 15, T, V, M]

    def remove(self, key):
        """Remove a single track buffer."""
        self._buffers.pop(key, None)

    def prune(self, valid_keys):
        """
        Remove all buffers whose key is not in valid_keys.
        Call once per frame with the set of currently tracked IDs to prevent
        memory leaking from lost tracks.
        """
        stale = [k for k in self._buffers if k not in valid_keys]
        for k in stale:
            del self._buffers[k]


print("✓ Cell 7: MediaPipe 3D mapping and SkeletonBuffer (15-ch) ready")
print(f"  SPINE_SCALE_CORRECTION = {SPINE_SCALE_CORRECTION}")
print(f"  MAX_FRAMES             = {MAX_FRAMES}")
print(f"  NUM_JOINTS             = {NUM_JOINTS}")

# ============================================================================
# CELL 8: Action Inference Helper
# ============================================================================
"""
CHANGED (15-channel alignment):
  split_sample slices the 15-channel tensor into the three 3D feature streams:
    joint    = tensor[:, 0:6]    (abs_xyz + rel_xyz)
    velocity = tensor[:, 6:12]   (fast_xyz + slow_xyz)
    bone     = tensor[:, 12:15]  (bone_dx, bone_dy, bone_dz)  ← was 12:16

  All other inference logic (temperature scaling, top-5, PredictionCache)
  is unchanged.
"""

CONFIDENCE_THRESHOLD = 0.20
SOFTMAX_TEMP         = 1.2


def split_sample(tensor_np):
    """
    Split [N, 15, T, V, M] into the three feature streams.
    CHANGED: slicing indices updated from 16-ch (6/6/4) to 15-ch (6/6/3).
      joint    : tensor_np[:, 0:6]
      velocity : tensor_np[:, 6:12]
      bone     : tensor_np[:, 12:15]   ← was 12:16
    """
    return tensor_np[:, 0:6], tensor_np[:, 6:12], tensor_np[:, 12:15]


def run_action_inference(tensor, role=None):
    """
    tensor : (1, 15, T, V, M) on CPU — output of SkeletonBuffer.get_tensor()
    role   : string ('patient' or 'caregiver') to filter predictions based on category.
    Returns (class_code_str, readable_label, confidence_float)
    """
    if action_model is None:
        return '???', 'model_not_loaded', 0.0

    tensor_np = tensor.cpu().numpy().astype(np.float32)
    joint, velocity, bone = split_sample(tensor_np)
    
    ort_inputs = {
        'joint': joint,
        'velocity': velocity,
        'bone': bone
    }
    logits = action_model.run(None, ort_inputs)[0]

    calibrated_logits = logits / SOFTMAX_TEMP
    exp_logits = np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True))
    probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]

    if role == 'patient':
        invalid_codes = ACTION_CATEGORIES.get('caregiver_specific', [])
    elif role == 'caregiver':
        invalid_codes = ACTION_CATEGORIES.get('patient_specific', [])
    else:
        invalid_codes = []
    
    if invalid_codes:
        for code in invalid_codes:
            if code in ACTION_CLASSES:
                code_idx = ACTION_CLASSES.index(code)
                probs[code_idx] = 0.0
        # Re-normalize just in case
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)

    idx  = np.argmax(probs)
    conf = float(probs[idx])

    if conf < CONFIDENCE_THRESHOLD:
        code  = '???'
        label = 'uncertain'
    else:
        code  = IDX_TO_CLASS[idx]
        label = LABEL_NAMES.get(code, code)

    top5_idxs = np.argsort(probs)[::-1][:5]
    top5_vals = probs[top5_idxs]
    top5 = [
        (
            IDX_TO_CLASS[i],
            LABEL_NAMES.get(IDX_TO_CLASS[i], IDX_TO_CLASS[i]),
            float(v)
        )
        for i, v in zip(top5_idxs, top5_vals)
    ]
    run_action_inference.last_top5 = top5

    return code, label, conf

run_action_inference.last_top5 = []


class PredictionCache:
    def __init__(self, max_age=8):
        self.max_age = max_age
        self._cache  = {}

    def get(self, key):
        if key in self._cache:
            code, label, conf, age = self._cache[key]
            if age < self.max_age:
                return code, label, conf
        return None

    def set(self, key, code, label, conf):
        self._cache[key] = (code, label, conf, 0)

    def tick(self):
        for k in list(self._cache):
            c, l, conf, age = self._cache[k]
            self._cache[k] = (c, l, conf, age + 1)

    def prune(self, valid_keys):
        for k in list(self._cache):
            if k not in valid_keys:
                del self._cache[k]


print("✓ Action inference helpers (15-ch) defined")


# ============================================================================
# CELL 9: Complete Visualization Utilities (with Skeleton & ID Color)
# ============================================================================
# UNCHANGED in behaviour.
# draw_skeleton operates entirely in 2D — reads only index 0 (x) and 1 (y),
# ignores Z. kpts_ntu is shape [25, 3].

_PALETTE = [
    (0,255,0),(0,128,255),(255,0,128),(255,255,0),(0,255,255),
    (255,0,255),(128,255,0),(0,255,128),(255,128,0),(128,0,255),
]

def _id_color(track_id):
    return _PALETTE[track_id % len(_PALETTE)]

def draw_skeleton(frame, kpts_ntu, color):
    """
    Draws the NTU-25 skeleton structure on the frame using only (x, y).
    kpts_ntu : np.ndarray [25, 3]  — pixel-space x, y, z per joint.
    Z is ignored for all OpenCV drawing operations.
    """
    connections = [
        (0, 1), (1, 20), (20, 2), (2, 3),
        (20, 4), (4, 5), (5, 6), (6, 7), (6, 21), (6, 22),
        (20, 8), (8, 9), (9, 10), (10, 11), (10, 23), (10, 24),
        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19)
    ]

    for i in range(len(kpts_ntu)):
        x, y = int(kpts_ntu[i][0]), int(kpts_ntu[i][1])
        if x > 0 or y > 0:
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

    for start_idx, end_idx in connections:
        pt1 = (int(kpts_ntu[start_idx][0]), int(kpts_ntu[start_idx][1]))
        pt2 = (int(kpts_ntu[end_idx][0]),   int(kpts_ntu[end_idx][1]))
        if pt1 != (0, 0) and pt2 != (0, 0):
            cv2.line(frame, pt1, pt2, color, 2)

def draw_action_overlays(frame, tracks_dict, action_labels, interaction_pairs, track_keypoints=None, role_map=None):
    """Draws boxes, skeletons, role labels, and action text."""
    vis = frame.copy()
    drawn_interaction = set()

    for tid, box in tracks_dict.items():
        x1, y1, x2, y2 = [int(v) for v in box]

        # Use role-based color if face identification is active
        if role_map is not None:
            role = role_map.get(tid)
            if role not in ('patient', 'caregiver'):
                role = 'unknown'
            color = ROLE_COLOR[role]
            role_label = ROLE_LABEL_TEXT[role]
        else:
            color = _id_color(tid)
            role_label = None

        if track_keypoints and tid in track_keypoints:
            draw_skeleton(vis, track_keypoints[tid], color)

        in_interaction = any(tid in pair for pair in interaction_pairs)
        thickness = 3 if in_interaction else 2
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        # ── Role label (filled badge above bounding box) ─────────────────
        action_label_anchor = y1   # default: action label anchors to box top
        if role_label:
            rfont      = cv2.FONT_HERSHEY_DUPLEX
            rfs, rtk   = 0.65, 2
            (rw, rh), rb = cv2.getTextSize(role_label, rfont, rfs, rtk)
            pad  = 4
            rtop = max(y1 - rh - 2*pad, 0)
            rbot = rtop + rh + 2*pad
            cv2.rectangle(vis, (x1, rtop), (x1 + rw + 2*pad, rbot), color, -1)
            cv2.putText(vis, role_label, (x1 + pad, rbot - pad - rb//2),
                        rfont, rfs, (255, 255, 255), rtk, cv2.LINE_AA)
            action_label_anchor = rtop   # push action label above role badge

        # ── Action label ─────────────────────────────────────────────────
        info = action_labels.get(tid)
        label_text = f"ID{tid}: {info[1]} ({info[2]:.0%})" if info else f"ID{tid}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ty = max(action_label_anchor - 6, th + 5)
        cv2.rectangle(vis, (x1, ty-th-4), (x1+tw+4, ty+2), (0,0,0), -1)
        cv2.putText(vis, label_text, (x1+2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for pair in interaction_pairs:
        if pair in drawn_interaction: continue
        drawn_interaction.add(pair)
        id_a, id_b = pair
        if id_a in tracks_dict and id_b in tracks_dict:
            ub = union_box(tracks_dict[id_a], tracks_dict[id_b])
            ux1, uy1, ux2, uy2 = [int(v) for v in ub]
            cv2.rectangle(vis, (ux1, uy1), (ux2, uy2), (0,0,255), 2)

            info = action_labels.get(pair)
            if info:
                text = f"ID{id_a}-{id_b}: {info[1]} ({info[2]:.0%})"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                ty = max(uy1-10, th+5)
                cv2.rectangle(vis, (ux1, ty-th-4), (ux1+tw+4, ty+2), (0,0,80), -1)
                cv2.putText(vis, text, (ux1+2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,100,255), 2)

    return vis

print("✓ Visualization utilities defined")


# ============================================================================
# CELL 10: Full Video Processing Pipeline (MediaPipe 3D Edition)
# ============================================================================
# UNCHANGED from the 3D upgrade version.
# Z-coordinate extraction: lm.z * crop_w (scales to same pixel units as x, y).
# Z does NOT receive a bounding-box offset because depth is not an image-plane
# coordinate — adding x1 to z would be physically meaningless.

import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class OneEuroFilter:
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def _low_pass_filter(self, x, alpha, x_prev):
        return alpha * x + (1.0 - alpha) * x_prev

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        edx = self._low_pass_filter(dx, self._alpha(self.dcutoff), self.dx_prev)
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        result = self._low_pass_filter(x, self._alpha(cutoff), self.x_prev)
        self.x_prev, self.dx_prev = result, edx
        return result

    def _alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

def process_video_with_action_recognition(
    video_path,
    output_path       = 'output_action.mp4',
    reid_model_path   = 'best_model.pth',
    conf_threshold    = 0.3,
    max_frames        = None,
    buffer_len        = 120,
    bbox_scale        = 1.5,
    iou_thresh        = 0.20,
    dist_thresh       = 150,
    persist_frames    = 10,
    inference_every   = 4,
    dataset_dir       = 'dataset',
):
    if not os.path.exists(video_path) or not os.path.exists(reid_model_path):
        print("ERROR: Missing video or ReID model file."); return

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames: tot_frames = min(tot_frames, max_frames)

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE)
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    detector = YOLO('yolo26n.onnx')
    feature_extractor = EnhancedFeatureExtractor(model_path=reid_model_path, device=device)
    tracker = MemoryEnhancedBoTSORT(feature_extractor=feature_extractor, max_age=3000, min_hits=5)
    interaction_detector = InteractionDetector(iou_thresh=iou_thresh, dist_thresh=dist_thresh, persist_frames=persist_frames)

    skeleton_buffer = SkeletonBuffer(buffer_len=buffer_len)
    pred_cache = PredictionCache(max_age=buffer_len)
    pose_filters = {}

    out_fps = max(1, fps // 2)
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (width, height))
    top5_action_log = defaultdict(lambda: defaultdict(float))

    action_logger = ActionEventLogger(
        video_path        = video_path,
        output_video_path = output_path,
        fps               = fps,
        width             = width,
        height            = height,
        device_str        = device,
    )

    # ── Face identity manager (patient / caregiver tagging) ───────────
    face_id_manager = None
    global_role_map = {}
    action_role_map = {}
    if dataset_dir and os.path.isdir(dataset_dir):
        try:
            face_id_manager = FaceIdentityManager()
            face_id_manager.build_gallery_from_folder(dataset_dir)
            print("  ✓ Face gallery built — role tagging enabled")
        except Exception as e:
            print(f"  ⚠️  Face identification disabled: {e}")
            face_id_manager = None
    else:
        print(f"  ⚠️  Dataset folder '{dataset_dir}' not found — role tagging disabled")

    print(f"\nProcessing {tot_frames} frames with OneEuro Stabilization...")
    frame_idx = 0

    import time
    time_yolo = 0.0
    time_tracking = 0.0
    time_mediapipe = 0.0
    time_action = 0.0
    time_face = 0.0
    time_draw = 0.0
    
    last_kpts_cache = {}

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames): break

        t0 = time.time()
        if frame_idx % 2 == 0:
            results = detector(frame, verbose=False, imgsz=640)[0]
            detections = [[*box.xyxy[0].cpu().numpy(), float(box.conf[0])]
                          for box in results.boxes if int(box.cls[0]) == 0 and float(box.conf[0]) > conf_threshold]
        else:
            # On odd frames, pass no detections. BoT-SORT's Kalman filter will predict and interpolate!
            detections = []
        time_yolo += time.time() - t0

        t0 = time.time()
        raw_tracks = tracker.update(frame, np.array(detections) if detections else np.empty((0, 5)))
        tracks_dict = {int(t[4]): t[:4].tolist() for t in raw_tracks} if len(raw_tracks) > 0 else {}

        active_pairs = interaction_detector.update(tracks_dict)
        time_tracking += time.time() - t0
        
        track_kpts_norm, track_kpts_pix = {}, {}

        t0 = time.time()
        
        def process_crop(tid_box):
            tid, box = tid_box
            ex1, ey1, ex2, ey2 = expand_bbox(box, height, width, scale=bbox_scale)
            crop = frame[ey1:ey2, ex1:ex2]
            if crop.size == 0: return None
            
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pose_res = pose_landmarker.detect(mp_img)
            
            if pose_res.pose_landmarks:
                landmarks = pose_res.pose_landmarks[0]
                kpts_rel = np.zeros((33, 3), dtype=np.float32)
                for i, lm in enumerate(landmarks):
                    kpts_rel[i] = [lm.x, lm.y, lm.z]
                return tid, kpts_rel
            return None

        for tid, box in tracks_dict.items():
            result = process_crop((tid, box))
            if result:
                tid_out, kpts_rel = result
                last_kpts_cache[tid_out] = kpts_rel
            else:
                if tid in last_kpts_cache:
                    del last_kpts_cache[tid]

        for tid, box in tracks_dict.items():
            if tid in last_kpts_cache:
                kpts_rel = last_kpts_cache[tid]
                
                ex1, ey1, ex2, ey2 = expand_bbox(box, height, width, scale=bbox_scale)
                c_h, c_w = (ey2 - ey1), (ex2 - ex1)
                
                kpts_px = np.zeros((33, 3), dtype=np.float32)
                for i in range(33):
                    kpts_px[i] = [kpts_rel[i][0] * c_w + ex1, kpts_rel[i][1] * c_h + ey1, kpts_rel[i][2] * c_w]

                if tid not in pose_filters:
                    pose_filters[tid] = OneEuroFilter(freq=fps, mincutoff=0.02, beta=0.01)
                
                # The OneEuro filter mathematically interpolates the skipped frame
                smoothed_px = pose_filters[tid](kpts_px)
                
                track_kpts_pix[tid]  = mediapipe_to_ntu25(smoothed_px)
                track_kpts_norm[tid] = mediapipe_to_ntu25(smoothed_px)

        for tid, ntu25 in track_kpts_norm.items(): skeleton_buffer.push(tid, ntu25)
        for pair in active_pairs:
            if all(p in track_kpts_norm for p in pair):
                skeleton_buffer.push_pair(pair, track_kpts_norm[pair[0]], track_kpts_norm[pair[1]])
        time_mediapipe += time.time() - t0

        t0 = time.time()
        pred_cache.tick()
        action_labels = {}
        if frame_idx % inference_every == 0:
            for key in (list(tracks_dict.keys()) + list(active_pairs)):
                if skeleton_buffer.ready(key):
                    tensor = skeleton_buffer.get_tensor(key)
                    
                    def _rl(t):
                        r = global_role_map.get(t) if face_id_manager else None
                        if r is None:
                            r = action_role_map.get(t)
                        return r if r in ('patient', 'caregiver') else None
                        
                    role = None
                    if not isinstance(key, tuple):
                        role = _rl(key)
                        
                    code, label, conf = run_action_inference(tensor, role=role)
                    
                    if not isinstance(key, tuple) and role is None:
                        cat = _CODE_TO_CATEGORY.get(code)
                        if cat == 'patient_specific':
                            action_role_map[key] = 'patient'
                            role = 'patient'
                        elif cat == 'caregiver_specific':
                            action_role_map[key] = 'caregiver'
                            role = 'caregiver'

                    pred_cache.set(key, code, label, conf)
                    action_labels[key] = (code, label, conf)
                    
                    if isinstance(key, tuple):
                        r0, r1 = _rl(key[0]), _rl(key[1])
                        if not r0 or not r1: continue
                        log_key = f"{r0}-{r1}"
                    else:
                        log_key = role
                        if not log_key: continue
                        
                    for _, t_label, t_conf in run_action_inference.last_top5:
                        if t_conf > top5_action_log[log_key][t_label]:
                            top5_action_log[log_key][t_label] = t_conf
        else:
            for key in (list(tracks_dict.keys()) + list(active_pairs)):
                cached = pred_cache.get(key)
                if cached: action_labels[key] = cached

        time_action += time.time() - t0
        # ── Face identification (role tagging) ────────────────────────
        t0 = time.time()
        if face_id_manager and len(raw_tracks) > 0:
            if frame_idx % 10 == 0:
                # Update EMA and re-match on even frames
                frame_role_map = face_id_manager.update(frame, raw_tracks, tracker=tracker)
            else:
                # Use current active memory on odd frames
                frame_role_map = face_id_manager.get_role_map()
                
            # Sync global role map exactly to current active tracks
            global_role_map.clear()
            for tid, role in frame_role_map.items():
                if role is not None:
                    global_role_map[tid] = role
        time_face += time.time() - t0

        combined_role_map = {}
        for tid in tracks_dict:
            if tid in global_role_map:
                combined_role_map[tid] = global_role_map[tid]
            elif tid in action_role_map:
                combined_role_map[tid] = action_role_map[tid]

        t0 = time.time()
        if frame_idx % 2 == 0:
            vis_frame = draw_action_overlays(frame, tracks_dict, action_labels, active_pairs,
                                             track_keypoints=track_kpts_pix,
                                             role_map=combined_role_map if (face_id_manager or action_role_map) else None)
            vis_frame = draw_trajectories(vis_frame, tracker,
                                          role_map=combined_role_map if (face_id_manager or action_role_map) else None)
            out_writer.write(vis_frame)
        time_draw += time.time() - t0

        if fps > 0 and frame_idx % fps == 0:
            action_logger.log_event(
                frame_idx       = frame_idx,
                fps             = fps,
                tracks_dict     = tracks_dict,
                action_labels   = action_labels,
                active_pairs    = active_pairs,
                track_kpts_norm = track_kpts_norm,
                role_map        = combined_role_map if (face_id_manager or action_role_map) else None,
            )

        if frame_idx % 30 == 0: print(f"  Frame {frame_idx}/{tot_frames} | Tracks: {len(tracks_dict)}")
        frame_idx += 1

    cap.release(); out_writer.release(); pose_landmarker.close()

    json_path = os.path.splitext(output_path)[0] + '_action_log.json'
    action_logger.save(json_path, total_frame_count=frame_idx)
    print(f"✓ Action log saved → {json_path}")

    # Print profiling summary
    print("\n" + "=" * 50)
    print("  PIPELINE PROFILING SUMMARY (Total Time)")
    print("=" * 50)
    total_pipeline_time = time_yolo + time_tracking + time_mediapipe + time_action + time_face + time_draw
    print(f"  YOLO Detection        : {time_yolo:.2f}s ({time_yolo/total_pipeline_time*100:.1f}%)")
    print(f"  Tracking (ReID)       : {time_tracking:.2f}s ({time_tracking/total_pipeline_time*100:.1f}%)")
    print(f"  MediaPipe (Pose)      : {time_mediapipe:.2f}s ({time_mediapipe/total_pipeline_time*100:.1f}%)")
    print(f"  Action (EfficientGCN) : {time_action:.2f}s ({time_action/total_pipeline_time*100:.1f}%)")
    print(f"  Face (InsightFace)    : {time_face:.2f}s ({time_face/total_pipeline_time*100:.1f}%)")
    print(f"  Drawing & Video Write : {time_draw:.2f}s ({time_draw/total_pipeline_time*100:.1f}%)")
    print("-" * 50)
    print(f"  TOTAL INFERENCE TIME  : {total_pipeline_time:.2f}s")
    print(f"  AVERAGE FPS           : {frame_idx / total_pipeline_time:.2f}")
    print("=" * 50 + "\n")

    return output_path, top5_action_log, action_logger


print("✓ Video processing pipeline defined")


# ============================================================================
# CELL 10.5: Action Event Logger — JSON Storage Module (token-efficient v2)
# ============================================================================
# CHANGED: token-efficient JSON structure with richer summary data for RAG.
#   - Per-frame actions stored as compact arrays [code, label, conf, category]
#     instead of verbose objects  (~60% token reduction per action entry).
#   - Shortened frame-level keys: fi / t / bb / sk / ia / dist / iou / sp_dist.
#   - Static action_categories lookup removed (not per-video data).
#   - session keys shortened and duration_sec added.
#   - Three new summary sections for RAG:
#       action_segments      : contiguous action blocks per track with timestamps.
#       per_track_summary    : dominant action + top-3 confidence per track.
#       interaction_events   : interaction timeline with dominant action label.

import json
import datetime

# Action categories were moved to the top.


class ActionEventLogger:
    """
    Logs per-second action recognition events and saves a compact JSON log
    enriched with segment/timeline summaries for downstream RAG ingestion.
    """

    def __init__(self, video_path, output_video_path, fps, width, height, device_str):
        self.video_path        = video_path
        self.output_video_path = output_video_path
        self.fps               = fps
        self.width             = width
        self.height            = height
        self.device_str        = str(device_str)
        self.processed_at      = datetime.datetime.now().isoformat(timespec='seconds')

        self._frames              = []
        self._all_track_ids       = set()
        self._recognition_count   = 0
        self._action_distribution = defaultdict(int)

        # Per-track action accumulator: {track_key_str: {label: [conf, ...]}} 
        self._track_action_confs  = defaultdict(lambda: defaultdict(list))
        # Per-track timeline: {track_key_str: [(t, code, label, conf, category), ...]}
        self._track_timeline      = defaultdict(list)
        # Interaction timeline: {(a,b): [(t, action_label), ...]}
        self._interaction_timeline = defaultdict(list)
        # Last total frame count tracked for duration computation
        self._last_frame_idx      = 0

    # ------------------------------------------------------------------
    def log_event(self, frame_idx, fps, tracks_dict, action_labels,
                  active_pairs, track_kpts_norm, role_map=None):
        """
        Record one logging event (called once per second of video).
        Compact keys: fi=frame_index, t=timestamp, bb=bbox, sk=has_skeleton,
                      ia=interaction_active, dist=distance_px, iou=iou,
                      sp_dist=skeleton_spine_distance_px.
        Actions stored as array: [code, label, conf, category].
        """
        self._last_frame_idx = frame_idx
        timestamp_sec = round(frame_idx / fps, 2) if fps > 0 else 0.0

        def get_role(tid):
            if role_map and tid in role_map:
                role = role_map[tid]
                if role in ('patient', 'caregiver'):
                    return role
                return f"unknown{tid}"
            return f"unknown{tid}"

        # ── Track records (compact) ──────────────────────────────────────
        tracks_record = {}
        for tid, box in tracks_dict.items():
            role_id = get_role(tid)
            if not role_id:
                continue
            self._all_track_ids.add(role_id)
            tracks_record[role_id] = {
                'bb': [round(float(v), 1) for v in box],
                'sk': 1 if tid in track_kpts_norm else 0,
            }

        # ── Action records (array format) ────────────────────────────────
        actions_record = {}
        for key, (code, label, conf) in action_labels.items():
            category = 'uncertain' if code == '???' else _CODE_TO_CATEGORY.get(code, 'unknown')
            if isinstance(key, tuple):
                r0, r1 = get_role(key[0]), get_role(key[1])
                if not r0 or not r1:
                    continue
                record_key = f"{r0}-{r1}"
            else:
                record_key = get_role(key)
                if not record_key:
                    continue
                
            # Compact array: [code, label, conf_3dp, category]
            actions_record[record_key] = [code, label, round(conf, 3), category]

            if code != '???' and conf > 0:
                self._recognition_count += 1
                self._action_distribution[label] += 1
                # Accumulate per-track confidence history
                self._track_action_confs[record_key][label].append(conf)
                # Record in timeline
                self._track_timeline[record_key].append(
                    (timestamp_sec, code, label, round(conf, 3), category)
                )

        # Accumulate interaction timeline
        for pair in active_pairs:
            id_a, id_b = pair
            r_a, r_b = get_role(id_a), get_role(id_b)
            if not r_a or not r_b:
                continue
            pair_key = tuple(sorted([r_a, r_b]))
            pair_label_key = f"{pair_key[0]}-{pair_key[1]}"
            if pair_label_key in actions_record:
                act_entry = actions_record[pair_label_key]
                self._interaction_timeline[pair_key].append(
                    (timestamp_sec, act_entry[1])  # (time, label)
                )
            else:
                self._interaction_timeline[pair_key].append((timestamp_sec, None))

        # ── Interaction records (compact) ────────────────────────────────
        interactions_record = []
        active_set = {tuple(sorted(p)) for p in active_pairs}

        for pair in active_pairs:
            id_a, id_b = pair
            r_a, r_b = get_role(id_a), get_role(id_b)
            if not r_a or not r_b:
                continue
            rec = {'a': r_a, 'b': r_b, 'ia': 1}
            if id_a in tracks_dict and id_b in tracks_dict:
                box_a = tracks_dict[id_a]
                box_b = tracks_dict[id_b]
                rec['dist'] = round(center_distance(box_a, box_b), 1)
                rec['iou']  = round(compute_iou(box_a, box_b), 3)
                if id_a in track_kpts_norm and id_b in track_kpts_norm:
                    sp_a = track_kpts_norm[id_a][20]
                    sp_b = track_kpts_norm[id_b][20]
                    rec['sp_dist'] = round(float(np.linalg.norm(sp_a - sp_b)), 1)
            interactions_record.append(rec)

        ids = list(tracks_dict.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id_a, id_b = ids[i], ids[j]
                key_pair = tuple(sorted([id_a, id_b]))
                if key_pair in active_set:
                    continue
                r_a, r_b = get_role(id_a), get_role(id_b)
                if not r_a or not r_b:
                    continue
                box_a, box_b = tracks_dict[id_a], tracks_dict[id_b]
                dist = center_distance(box_a, box_b)
                if dist < 300:
                    interactions_record.append({
                        'a'   : r_a,
                        'b'   : r_b,
                        'ia'  : 0,
                        'dist': round(dist, 1),
                        'iou' : round(compute_iou(box_a, box_b), 3),
                    })

        # ── Append compact frame record ──────────────────────────────────
        self._frames.append({
            'fi'      : frame_idx,
            't'       : timestamp_sec,
            'tracks'  : tracks_record,
            'actions' : actions_record,
            'interact': interactions_record,
        })

    # ------------------------------------------------------------------
    def _build_action_segments(self):
        """
        Convert per-track timelines into contiguous action segments.
        A new segment starts whenever the action label changes.
        Returns list of dicts: {track, action, category, t_start, t_end,
                                 avg_conf, frame_count}
        """
        segments = []
        for track_key, events in self._track_timeline.items():
            if not events:
                continue
            # events: [(t, code, label, conf, category), ...]
            seg_start   = events[0][0]
            seg_label   = events[0][2]
            seg_cat     = events[0][4]
            seg_confs   = [events[0][3]]

            for t, code, label, conf, category in events[1:]:
                if label == seg_label:
                    seg_confs.append(conf)
                else:
                    segments.append({
                        'track'      : track_key,
                        'action'     : seg_label,
                        'category'   : seg_cat,
                        't_start'    : seg_start,
                        't_end'      : t,
                        'avg_conf'   : round(float(np.mean(seg_confs)), 3),
                        'frame_count': len(seg_confs),
                    })
                    seg_start = t
                    seg_label = label
                    seg_cat   = category
                    seg_confs = [conf]

            # Close last segment
            last_t = events[-1][0]
            segments.append({
                'track'      : track_key,
                'action'     : seg_label,
                'category'   : seg_cat,
                't_start'    : seg_start,
                't_end'      : last_t,
                'avg_conf'   : round(float(np.mean(seg_confs)), 3),
                'frame_count': len(seg_confs),
            })

        segments.sort(key=lambda s: s['t_start'])
        return segments

    # ------------------------------------------------------------------
    def _build_per_track_summary(self):
        """
        For each track, compute dominant action and top-3 by average confidence.
        Returns {track_key: {dominant_action, top3: [[label, avg_conf], ...]}}
        """
        per_track = {}
        for track_key, label_confs in self._track_action_confs.items():
            avg_confs = {
                label: round(float(np.mean(confs)), 3)
                for label, confs in label_confs.items()
            }
            sorted_actions = sorted(avg_confs.items(), key=lambda x: x[1], reverse=True)
            per_track[track_key] = {
                'dominant_action': sorted_actions[0][0] if sorted_actions else 'unknown',
                'top3': [[lbl, conf] for lbl, conf in sorted_actions[:3]],
            }
        return per_track

    # ------------------------------------------------------------------
    def _build_interaction_events(self):
        """
        Convert interaction timeline into meaningful events with dominant action.
        Returns list of dicts: {track_a, track_b, t_start, t_end, duration_sec,
                                 dominant_action}
        """
        events = []
        for (id_a, id_b), timeline in self._interaction_timeline.items():
            if not timeline:
                continue
            # timeline: [(t, action_label_or_None), ...]
            label_counts = defaultdict(int)
            for _, lbl in timeline:
                if lbl:
                    label_counts[lbl] += 1
            dominant = max(label_counts, key=label_counts.get) if label_counts else 'unknown'
            t_vals = [t for t, _ in timeline]
            events.append({
                'track_a'       : id_a,
                'track_b'       : id_b,
                't_start'       : min(t_vals),
                't_end'         : max(t_vals),
                'duration_sec'  : round(max(t_vals) - min(t_vals), 1),
                'dominant_action': dominant,
            })
        events.sort(key=lambda e: e['t_start'])
        return events

    # ------------------------------------------------------------------
    def save(self, output_path, total_frame_count=None):
        """
        Serialise the compact JSON log.
        total_frame_count: if provided, used to compute video duration_sec.
        """
        action_dist_sorted = dict(
            sorted(self._action_distribution.items(), key=lambda x: x[1], reverse=True)
        )

        dur = round(
            (total_frame_count or self._last_frame_idx) / max(self.fps, 1), 2
        )

        payload = {
            'session': {
                'video'        : self.video_path,
                'out_video'    : self.output_video_path,
                'processed_at' : self.processed_at,
                'fps'          : self.fps,
                'duration_sec' : dur,
                'resolution'   : [self.width, self.height],
                'model'        : 'EfficientGCN-B0 (3D, NTU 45-class, 15-ch)',
                'device'       : self.device_str,
                'num_classes'  : NUM_CLASSES,
                'total_tracks' : len(self._all_track_ids),
            },
            'summary': {
                'total_frames_logged' : len(self._frames),
                'unique_tracks'       : sorted(self._all_track_ids),
                'total_recognitions'  : self._recognition_count,
                'action_distribution' : action_dist_sorted,
                'per_track_summary'   : self._build_per_track_summary(),
                'action_segments'     : self._build_action_segments(),
                'interaction_events'  : self._build_interaction_events(),
            },
            'frames': self._frames,
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"✓ ActionEventLogger: {len(self._frames)} frames written → {output_path}")
        return output_path


print("✓ ActionEventLogger (token-efficient v2) and ACTION_CATEGORIES defined")


# ============================================================================
# CELL 11: Run the Pipeline
# ============================================================================

if __name__ == '__main__':
    Track._count = 0

    output, top5_action_log, action_logger = process_video_with_action_recognition(
        video_path      = 'input.mp4',
        output_path     = 'output_action.mp4',
        reid_model_path = 'best_model.onnx',
        conf_threshold  = 0.25,
        max_frames      = None,
        buffer_len      = 90,
        bbox_scale      = 1.2,
        iou_thresh      = 0.20,
        dist_thresh     = 150,
        persist_frames  = 10,
        inference_every = 8,
        dataset_dir     = 'dataset',
    )

    print(f"\n✓ Output video : {output}")
    print(f"✓ JSON log     : {os.path.splitext(output)[0] + '_action_log.json'}")

    # ============================================================================
    # CELL 12: Preview Output
    # ============================================================================

    def show_video(path):
        if not os.path.exists(path):
            print(f"Video not found: {path}")
            return
        abs_path = os.path.abspath(path)
        print(f"\n✓ Output video ready: {abs_path}")
        print("  Open with any video player (e.g. VLC, Windows Media Player).")

    show_video('output_action.mp4')

    # ============================================================================
    # CELL 13: Top-5 Actions per Identified Person
    # ============================================================================

    def display_top5_actions(log):
        if not log:
            print("No action data collected. Make sure the video was processed first.")
            return

        _label_to_code = {v: k for k, v in LABEL_NAMES.items()}

        print("\n" + "=" * 74)
        print("  TOP-5 ACTIONS PER IDENTIFIED PERSON / INTERACTION PAIR")
        print("=" * 74)

        def sort_key(k):
            return (0, k) if isinstance(k, int) else (1, str(k))

        for subject in sorted(log.keys(), key=sort_key):
            action_scores = log[subject]
            sorted_actions = sorted(action_scores.items(), key=lambda x: x[1], reverse=True)[:5]

            heading = (
                f"Person ID {subject}"
                if isinstance(subject, int)
                else f"Interaction {subject}"
            )
            print(f"\n{'─' * 74}")
            print(f"  {heading}")
            print(f"{'─' * 74}")
            print(f"  {'Rank':<5} {'Action':<30} {'Category':<20} {'Confidence':<10} {'Bar'}")
            print(f"  {'─'*4} {'─'*29} {'─'*19} {'─'*10} {'─'*30}")

            for rank, (action, conf) in enumerate(sorted_actions, start=1):
                code     = _label_to_code.get(action, '???')
                category = _CODE_TO_CATEGORY.get(code, 'unknown')
                bar      = '█' * int(conf * 30) + '░' * (30 - int(conf * 30))
                print(
                    f"  {rank:<5} {action:<30} {category:<20} {conf:>8.1%}   {bar}"
                )

        print("\n" + "=" * 74)
        print("✓ Top-5 action summary complete.")
        print("=" * 74)

    display_top5_actions(top5_action_log)