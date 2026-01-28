#!/usr/bin/env python3
"""
相機偵測器類別 - 支持多相機
使用全局共享模型，只載入一次
"""
import torch
import cv2
import numpy as np
import os
import random
from groundingdino.util.inference import predict, load_image, annotate
from sklearn.decomposition import PCA
from PIL import Image
import pyrealsense2 as rs
from shared_models import shared_models
import torch.nn.functional as F
import threading
import gc
from scipy.spatial import KDTree
import time
def list_cameras():
        """列出所有連接的 RealSense 相機"""
        ctx = rs.context()
        devices = ctx.query_devices()
        
        print(f"找到 {len(devices)} 個 RealSense 相機:")
        for i, device in enumerate(devices):
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            print(f"  [{i}] 序列號: {serial}, 名稱: {name}")
        
        return devices
class CameraDetector:
    """單個相機的偵測器 - 使用全局共享模型"""
    # 添加類級別的 SAM 鎖（所有實例共享）
    _sam_lock = threading.Lock()
    def __init__(self, realsense_serial="923322070636", camera_id=0, max_objects=2, 
                 candidate_phrases=None):
        """
        初始化相機偵測器
        
        參數:
            realsense_serial: 相機序列號
            camera_id: 相機 ID (0:head camera, 1:left camera, 2:right camera)
            max_objects: 最多檢測物品數
            candidate_phrases: 檢測的物品類別
        """
        self.camera_serial = realsense_serial
        self.camera_id = camera_id
        self.max_objects = max_objects
        
        # 使用全局共享模型
        self.gd_model = shared_models.gd_model
        self.predictor = shared_models.predictor
        self.clip_model = shared_models.clip_model
        self.clip_processor = shared_models.clip_processor
        # self.handle_reference_features = shared_models.handle_reference_features
        self.handle_reference_features = None
        self.device = shared_models.device
        
        # 偵測參數
        if candidate_phrases is None:
            self.candidate_phrases = [
                "tool", "blue vacuum cleaner", "broom", "dustpan tool", "brush tool"
            ]
        else:
            self.candidate_phrases = candidate_phrases
        
        self.caption = " . ".join(self.candidate_phrases)
        
        # 相機狀態
        self.running = True
        self.latest_color_image = None
        self.latest_depth_frame = None
        self.depth_image = None
        self.detection_count = 0
        self.detected_threshold = 5
        self.objects_info = []
        self.intrinsics = None
        
        # 輸出目錄
        self.output_dir = f"/home/gairobots/camera/GroundingDINO/output/camera_{camera_id}"
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化相機
        self.init_camera()
        self.camera_started = True
        print(f"\n✅ 相機 {self.camera_id} (序列號: {self.camera_serial}) 初始化完成")

    def init_camera(self):
        """初始化 RealSense 相機"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.camera_serial)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 獲取相機內參
        self.intrinsics = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile().get_intrinsics()
        
        print(f"  相機內參: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
        print(f"  預熱中...")
        # 跳過前 40 幀
        for i in range(40):
            self.pipeline.wait_for_frames()
       
    def pause_camera(self):
        """暫停相機（臨時停止但保留設定）"""
        if not self.camera_started:
            return
        
        print(f"[Camera {self.camera_id}] ⏸️  暫停相機...")
        
        if self.pipeline:
            self.pipeline.stop()
        
        self.camera_started = False
        print(f"  ✅ 相機已暫停")
    def resume_camera(self):
        """恢復相機（快速重啟）"""
        if self.camera_started:
            print(f"[Camera {self.camera_id}] 相機已在運行")
            return
        
        print(f"[Camera {self.camera_id}] ▶️  恢復相機...")
        
        # 重新啟動（保留之前的配置）
        config = rs.config()
        config.enable_device(self.camera_serial)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 快速預熱（只需 10 幀）
        for i in range(10):
            self.pipeline.wait_for_frames()
        
        self.camera_started = True
        print(f"  ✅ 相機已恢復")
    def test_buffer_without_reading(self):
        """測試完全不讀取時的緩衝區積累"""
        if not self.camera_started:
            print("❌ 相機未啟動")
            return
        
        print(f"\n{'='*60}")
        print(f"[測試] 相機 {self.camera_id} - 完全不讀取測試")
        print(f"{'='*60}")
        
        # 步驟 1：清空
        print("步驟 1: 清空所有現有幀...")
        count_before = 0
        while True:
            frames = self.pipeline.poll_for_frames()
            if frames:
                count_before += 1
            else:
                break
        print(f"  清空了 {count_before} 幀")
        
        # 步驟 2：完全不讀取，等待 5 秒
        print("步驟 2: 完全不讀取，等待 5 秒...")
        print("  （如果有其他線程在讀取，緩衝區不會積累）")
        
        start_time = time.time()
        time.sleep(5.0)
        elapsed = time.time() - start_time
        
        print(f"  實際等待時間: {elapsed:.3f}s")
        
        # 步驟 3：立即計數緩衝區
        print("步驟 3: 計數緩衝區中的幀...")
        
        count_after = 0
        timestamps = []
        
        for i in range(300):  # 最多檢查 300 幀
            frames = self.pipeline.poll_for_frames()
            if frames:
                count_after += 1
                ts = frames.get_timestamp()
                timestamps.append(ts)
                
                # 只打印前 5 幀和最後 5 幀
                if count_after <= 5 or i >= 295:
                    print(f"  幀 {count_after}: timestamp={ts:.2f}ms")
            else:
                break
        
        # 結果分析
        print(f"\n{'='*60}")
        print("測試結果:")
        print(f"  等待時間: {elapsed:.1f}s")
        print(f"  緩衝區幀數: {count_after}")
        print(f"  理論值 (30 FPS × 5s): 150 幀")
        print(f"  實際比例: {count_after/150*100:.0f}%")
        
        if count_after < 5:
            print(f"\n  ❌ 嚴重問題：幾乎沒有幀積累！")
            print(f"     可能原因：")
            print(f"     1. 有後台線程在持續讀取相機")
            print(f"     2. 有其他程序在訪問相機")
            print(f"     3. 相機沒有正常串流")
        elif count_after < 100:
            print(f"\n  ⚠️ 幀積累不足（可能有程序在讀取）")
        else:
            print(f"\n  ✅ 緩衝區積累正常（沒有其他程序干擾）")
        
        # 時間戳分析
        if len(timestamps) > 1:
            time_span = (timestamps[-1] - timestamps[0]) / 1000.0  # 秒
            print(f"\n  幀時間跨度: {time_span:.2f}s")
            
            if time_span < 1.0:
                print(f"  ⚠️ 時間跨度太小！這些可能是舊幀")
        
        print(f"{'='*60}\n")
        
        return count_after
    

    
    def check_for_background_threads(self):
        """檢查是否有後台線程在讀取相機"""
        import threading
        
        print(f"\n{'='*60}")
        print(f"[診斷] 檢查相機 {self.camera_id} 的後台線程")
        print(f"{'='*60}")
        
        # 列出所有活動線程
        all_threads = threading.enumerate()
        print(f"當前活動線程數: {len(all_threads)}")
        
        for thread in all_threads:
            print(f"  - {thread.name} (daemon={thread.daemon}, alive={thread.is_alive()})")
        
        # 檢查類成員變量
        print(f"\n檢查相機 {self.camera_id} 的成員變量:")
        
        if hasattr(self, 'camera_thread'):
            print(f"  ⚠️ 發現 camera_thread: {self.camera_thread}")
            if self.camera_thread:
                print(f"     - Alive: {self.camera_thread.is_alive()}")
        else:
            print(f"  ✅ 無 camera_thread")
        
        if hasattr(self, 'thread_running'):
            print(f"  ⚠️ 發現 thread_running: {self.thread_running}")
        else:
            print(f"  ✅ 無 thread_running")
        
        if hasattr(self, 'running'):
            print(f"  發現 running: {self.running}")
        
        print(f"{'='*60}\n")

        

    

    def get_current_frame(self):
        # self.check_for_background_threads()
        # self.test_buffer_without_reading()
        """
        獲取當前幀（適配 RealSense 小隊列設計）
        
        RealSense 特性：
        - 隊列只保留 1-2 幀（不積累）
        - 這是設計特性，不是 bug
        - 必須用 wait_for_frames() 主動等待新幀
        """
        if not self.camera_started:
            print(f"[Camera {self.camera_id}] ❌ 相機未啟動")
            return None, None
        
        print(f"[Camera {self.camera_id}] 獲取最新幀...")
        
        # ========== 步驟 1：清空隊列（只有 1-2 幀）==========
        flush_count = 0
        while self.pipeline.poll_for_frames():
            flush_count += 1
        print(f"  Flushed {flush_count} frames")
        
        # ========== 步驟 2：主動等待新幀（關鍵！）==========
        # 不用 sleep 等待積累，而是用 wait 主動獲取新幀
        print(f"  Waiting for 15 new frames (~0.6s)...")
        
        for i in range(15):
            try:
                # wait_for_frames() 會阻塞直到相機產生「下一個」新幀
                frames = self.pipeline.wait_for_frames()
                
                # 稍微延遲，讓相機有時間產生下一幀
                # 30 FPS = 33.33ms/幀，等 40ms 確保是新幀
                time.sleep(0.040)
                
            except Exception as e:
                print(f"  ⚠️ 等待幀失敗: {e}")
                break
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        self.latest_depth_frame = depth_frame
        self.depth_image = np.asanyarray(depth_frame.get_data())
        self.latest_color_image = np.asanyarray(color_frame.get_data())
        
        return self.latest_color_image, self.depth_image
    
    def depth_to_point_cloud(self, depth_image, mask):
        """深度轉 3D 點雲"""
        points_3d = []
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy
        # 找到遮罩中的所有像素
        ys, xs = np.where(mask > 0)
        
        for x, y in zip(xs, ys):
            depth = depth_image[y, x] / 1000.0 # 轉換為公尺
            
            if depth == 0 or depth > 3.0:
                continue
             # 深度轉3D座標（相機坐標系，針孔成像原理）
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            points_3d.append([X, Y, Z])
        print(f"最小z: {(min([p[2] for p in points_3d]) if len(points_3d)>0 else 0)}")
        return np.array(points_3d)
    
    def compute_3d_bounding_box(self, points_3d):
        """使用 PCA 計算 3D Bounding Box"""
        if len(points_3d) < 10:
            return None
        # 計算質心
        centroid = np.mean(points_3d, axis=0)
        # PCA 找主方向
        points_centered = points_3d - centroid
        pca = PCA(n_components=3)
        pca.fit(points_centered)
        # 主軸（特徵向量）
        axes = pca.components_ # 3x3 矩陣
        # 將點投影到主軸坐標系
        points_rotated = points_centered @ axes.T
        # 計算在主軸坐標系中的邊界
        min_bound = np.min(points_rotated, axis=0)
        max_bound = np.max(points_rotated, axis=0)
        # 3D bounding box 的尺寸（長寬高）
        size = max_bound - min_bound
        # 計算方向角度（yaw, pitch, roll）
        # 主要方向向量
        main_axis = axes[0]# 第一主成分
        # Yaw 角度（繞 Z 軸，XY 平面）
        yaw = np.arctan2(main_axis[1], main_axis[0]) * 180 / np.pi
        # Pitch 角度（繞 Y 軸，XZ 平面）
        pitch = np.arcsin(-main_axis[2]) * 180 / np.pi
        # 計算8個角點（在主軸坐標系）
        corners_local = np.array([
            [min_bound[0], min_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], max_bound[2]],
        ])
        
        # 轉回原始坐標系
        corners_world = corners_local @ axes + centroid
        corners_xy = corners_world[:, :2]  # 只取 X, Y
        center_xy = np.mean(corners_xy, axis=0)  # (x_center, y_center)
        kdtree_xy = KDTree(points_3d[:, :2])
        distances, indices = kdtree_xy.query(center_xy, k=10)  # 找10個最近點
        
        nearby_points = points_3d[indices]
        
        if len(nearby_points) > 0:
            center_z = np.mean(nearby_points[:, 2])
        else:
            center_z = centroid[2]
        
        new_center = np.array([center_xy[0], center_xy[1], center_z])
        print(f"      3D BBox 中心 Z 調整: {centroid[2]:.4f} -> {new_center[2]:.4f} ")
        return {
            'center': new_center,
            'size': size,
            'orientation': {'yaw': yaw, 'pitch': pitch},
            'rotation_matrix': axes,
            'corners': corners_world,
            'volume': np.prod(size)
        }
    
    def project_3d_to_2d(self, point_3d):
        """3D 投影到 2D 圖像座標"""
        X, Y, Z = point_3d
        if Z == 0:
            return None
        
        u = int(X * self.intrinsics.fx / Z + self.intrinsics.ppx)
        v = int(Y * self.intrinsics.fy / Z + self.intrinsics.ppy)
        
        return (u, v)
    
    def visualize_3d_bbox(self, image, bbox_3d, color=(0, 255, 0)):
        """繪製 3D Bounding Box"""
        corners = bbox_3d['corners']
        # 投影所有角點到 2D
        points_2d = []
        for corner in corners:
            pt_2d = self.project_3d_to_2d(corner)
            if pt_2d:
                points_2d.append(pt_2d)
        
        if len(points_2d) < 8:
            return image
        # 繪製 12 條邊（只繪製框架）
        edges = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5),
            (2, 3), (2, 6), (3, 7), (4, 5), (4, 6),
            (5, 7), (6, 7)
        ]
        
        for edge in edges:
            pt1 = points_2d[edge[0]]
            pt2 = points_2d[edge[1]]
            cv2.line(image, pt1, pt2, color, 2)
        
        return image
    
    def compute_iou(self, box1, box2):
        """計算 IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def non_maximum_suppression(self, detections, iou_threshold=0.5):
        """NMS - 移除重疊檢測"""
        if len(detections) == 0:
            return []
        
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            current = detections[0]
            keep.append(current)
            detections = detections[1:]
            
            filtered = []
            for det in detections:
                iou = self.compute_iou(current['bbox_2d'], det['bbox_2d'])
                if iou < iou_threshold:
                    filtered.append(det)
            detections = filtered
        
        return keep
    
    def camera_to_ee_transform(self, point_camera_mm): 
        # horizontal_offset = 29 #mm
        vertical_offset = 57.5 #mm
        
        camera_offset_x = 31 #mm 30
        z_offset = 87.32 #mm 62.32 72.32
        if self.camera_id == 0:
            """頭相機座標轉脖子末端座標"""
            X_camera, Y_camera, Z_camera = point_camera_mm
            # X_ee = Z_camera
            # Y_ee = -X_camera + head_offset
            # Z_ee = -Y_camera
            # X_ee = Z_camera
            # Y_ee = Y_camera 
            # Z_ee = X_camera
            X_ee = Z_camera
            Y_ee = -X_camera + camera_offset_x
            Z_ee = -Y_camera

            

        elif self.camera_id == 1:
            """左相機座標轉末端座標"""
            X_camera, Y_camera, Z_camera = point_camera_mm
            X_ee = -X_camera + camera_offset_x
            Y_ee = -Y_camera + vertical_offset
            Z_ee =  Z_camera - z_offset
            # X_ee = Y_camera + horizontal_offset
            # Y_ee = X_camera + vertical_offset
            # Z_ee =  Z_camera - z_offset

        elif self.camera_id == 2:
            """右相機座標轉末端座標"""
            X_camera, Y_camera, Z_camera = point_camera_mm
            X_ee = -X_camera + camera_offset_x
            Y_ee = -Y_camera + vertical_offset
            Z_ee = Z_camera - z_offset
            # X_ee = Y_camera + horizontal_offset
            # Y_ee = X_camera + vertical_offset
            # Z_ee =  Z_camera - z_offset
        return (X_ee, Y_ee, Z_ee)
    
    
    def compute_handle_similarity(self, crop_img):
        """計算握柄相似度（CLIP）"""
        if self.handle_reference_features is None:
            return 0.0
        
        # 轉 PIL
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        # CLIP 特徵提取
        inputs = self.clip_processor(images=crop_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            crop_features = self.clip_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
        
        # 計算相似度
        similarity = F.cosine_similarity(
            self.handle_reference_features, crop_features, dim=-1
        ).item()
        
        return similarity
    
    def find_handle_by_sliding_window(self, object_crop, step_size=20, window_sizes=None):
        """滑動窗口搜索握柄"""
        if self.handle_reference_features is None:
            print("    ⚠️  沒有載入握柄參考特徵")
            return None
        h, w = object_crop.shape[:2]
        
        if window_sizes is None:
            window_sizes = [(100, 100), (120, 120), (180, 180)]
            # window_sizes = [(200,200), (250,250), (300,300)]
            # window_sizes = [(h, w), (180,180)]
            # window_sizes = [
            #     (int(h * 0.3), 180),  # 中窗口
            
                
                
            #     (180, 180),
            # ]
    
        best_similarity = 0.0
        best_bbox = None
        best_size = None
        
        print(f"    滑動窗口搜索握柄 ({h}x{w})...")
        
        for win_h, win_w in window_sizes:
            if win_h >= h or win_w >= w:
                continue
            
            # 滑動窗口
            for y in range(0, h - win_h, step_size):
                for x in range(0, w - win_w, step_size):
                    # 提取窗口
                    window_crop = object_crop[y:y+win_h, x:x+win_w].copy()
                    
                    # 計算相似度
                    similarity = self.compute_handle_similarity(window_crop)
                    
                    # 更新最佳
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_bbox = (x, y, x + win_w, y + win_h)
                        best_size = (win_w, win_h)
            print(f"      窗口 ({win_h}x{win_w}) : 相似度 {similarity:.3f}")
        
        print(f"    窗口大小: {best_size}, 最佳相似度: {best_similarity:.3f}")
        
        if best_similarity > 0.35 and best_bbox is not None:
            print(f"    ✓ 找到握柄 (相似度: {best_similarity:.3f}, 尺寸: {best_size})")
            return {
                'bbox': best_bbox,
                'similarity': best_similarity,
                'size': best_size
            }
        else:
            print(f"    ❌ 握柄搜索失敗 (最佳相似度: {best_similarity:.3f})")
            return None


    def _get_initial_detections(self, image_source, image):
        self.caption = " . ".join(self.candidate_phrases)
        """第一步：使用 GroundingDINO 進行初始偵測"""
        boxes, logits, phrases = predict(
            self.gd_model, image, self.caption, 0.20, 0.10, self.device
        )
        
        print(f"   GroundingDINO 偵測到 {len(boxes)} 個候選框")
        # ===== 收集所有候選檢測，並按信心度排序 =====
        all_detections = []
        H, W = image_source.shape[:2]
        
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            if logit < 0.25: # 過濾低信心度
                continue
            # 轉像素座標
            cx, cy, w, h = box
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)
            
            all_detections.append({
                'box': box,
                'bbox_2d': (x1, y1, x2, y2),
                'logit': float(logit),
                'phrase': phrase,
                'confidence': float(logit)
            })
        
        return all_detections
    
    def _filter_detections(self, all_detections):
        """第二步：NMS 和篩選頂部檢測"""
        if len(all_detections) == 0:
            print("未偵測到任何物品")
            return []
        
        all_detections = self.non_maximum_suppression(all_detections, iou_threshold=0.5)
         # 選擇前 k 個信心度最高的檢測
        selected_detections = all_detections[:self.max_objects]
        
        print(f"   經過 NMS 和篩選後，剩餘 {len(selected_detections)} 個檢測")
        
        return selected_detections
    
    def _detect_object_region(self, image_source, x1, y1, x2, y2, det_idx):
        """第三步：檢測整體物體區域"""
        print(f"      第一階段：整體物體 3D bbox...")
        
        crop_image = image_source[y1:y2, x1:x2].copy()
        masks_obj, _, _ = self.predictor.predict(
            box=np.array([x1, y1, x2, y2]),
            multimask_output=False
        )
        object_mask = masks_obj[0].astype(np.uint8) * 255
        
        mask_path = os.path.join(self.output_dir, f"mask_object_{det_idx}.jpg")
        cv2.imwrite(mask_path, object_mask)
        
        self.points_3d_object = self.depth_to_point_cloud(
            self.depth_image, object_mask
        )
        
        if len(self.points_3d_object) < 10:
            print(f"      ❌ 物體點雲不足，跳過")
            return None, None, None
        
        bbox_3d_object = self.compute_3d_bounding_box(self.points_3d_object)
        
        if bbox_3d_object is None:
            print(f"      ❌ 無法計算整體物體 3D bbox，跳過")
            return None, None, None
        
        print(f"      ✓ 整體物體 3D bbox 計算成功 ({len(self.points_3d_object)} 個點)")
        
        return bbox_3d_object, object_mask, self.points_3d_object, crop_image
    
    def _detect_handle_region(self, image_source, crop_image, x1, y1, x2, y2, 
                             bbox_3d_object, object_mask, points_3d_object, det_idx):
        """第四步：搜索握柄區域"""
        print(f"      第二階段：CLIP 滑動窗口搜索握柄...")
        
        handle_result = self.find_handle_by_sliding_window(crop_image, step_size=10)
        use_handle = False
        
        if handle_result is not None:
            handle_x1_crop, handle_y1_crop, handle_x2_crop, handle_y2_crop = handle_result['bbox']
            similarity_score = handle_result['similarity']
            
            handle_x1 = x1 + handle_x1_crop
            handle_y1 = y1 + handle_y1_crop
            handle_x2 = x1 + handle_x2_crop
            handle_y2 = y1 + handle_y2_crop
            
            print(f"      ✓ 找到握柄區域: ({handle_x1}, {handle_y1}) -> ({handle_x2}, {handle_y2})")
            print(f"      握柄 CLIP 相似度: {similarity_score:.3f}")
            with torch.no_grad():
                masks_handle, _, _ = self.predictor.predict(
                    box=np.array([handle_x1, handle_y1, handle_x2, handle_y2]),
                    multimask_output=False
                )
            handle_mask = masks_handle[0].astype(np.uint8) * 255
            del masks_handle
            gc.collect()
            torch.cuda.empty_cache()
            handle_mask_path = os.path.join(
                self.output_dir, f"mask_handle_final_{det_idx}.jpg"
            )
            cv2.imwrite(handle_mask_path, handle_mask)
            
            handle_points_3d = self.depth_to_point_cloud(
                self.depth_image, handle_mask
            )
            
            if len(handle_points_3d) >= 10:
                handle_bbox_3d = self.compute_3d_bounding_box(handle_points_3d)
                
                if handle_bbox_3d is not None:
                    print(f"      ✓ 握柄 3D bbox 計算成功 ({len(handle_points_3d)} 個點)")
                    use_handle = True
                    
                    return {
                        'bbox_3d': handle_bbox_3d,
                        'mask': handle_mask,
                        'bbox_2d': (handle_x1, handle_y1, handle_x2, handle_y2),
                        'points_3d': handle_points_3d,
                        'region_type': 'handle'
                    }
                else:
                    print(f"      ⚠️  無法計算握柄 3D bbox，改用整個物體")
            else:
                print(f"      ⚠️  握柄點雲不足，改用整個物體")
        else:
            print(f"      ⚠️  CLIP 搜索未找到握柄，改用整個物體")
        
        # 回退到整體物體
        print(f"      使用整個物體 3D bounding box")
        return {
            'bbox_3d': bbox_3d_object,
            'mask': object_mask,
            'bbox_2d': (x1, y1, x2, y2),
            'points_3d': points_3d_object,
            'region_type': 'object'
        }
    
    def _extract_3d_info(self, bbox_3d):
        """第五步：提取 3D 資訊"""
        corners = bbox_3d['corners']
        print(f"      3D Bounding Box 角點:\n{corners}")
        center_3d = np.mean(corners, axis=0)
        print(f"      3D Bounding Box 中心點: {center_3d}")
    
        center_3d = np.array([center_3d[0] * 1000, center_3d[1] * 1000, center_3d[2] * 1000])
        size_3d = np.array([bbox_3d['size'][0]*1000, bbox_3d['size'][1]*1000, bbox_3d['size'][2]*1000])
   
        yaw = bbox_3d['orientation']['yaw']

        pitch = bbox_3d['orientation']['pitch']
        center_base = self.camera_to_ee_transform(center_3d)
        
        return {
            'center_3d': center_3d,
            'center_base': center_base,
            'size_3d': size_3d,
            'yaw': yaw,
            'pitch': pitch,
            'corners': corners
        }
    
    def _calculate_endpoints(self, bbox_3d, center_3d):
        """第六步：計算長邊兩端點"""
        corners = bbox_3d['corners']
        size_3d = bbox_3d['size']
        axes = bbox_3d['rotation_matrix']
        
        sorted_indices = np.argsort(size_3d)[::-1]
        long_idx = sorted_indices[0]
        long_dir = axes[long_idx]
        
        projections = []
        for i, corner in enumerate(corners):
            vec_to_corner = corner - center_3d
            projection = np.dot(vec_to_corner, long_dir)
            projections.append((projection, i, corner))
        
        projections.sort(key=lambda x: x[0])
        
        left_face_corners = [p[2] for p in projections[:4]]
        right_face_corners = [p[2] for p in projections[4:]]
        
        left_endpoint_3d = np.mean(left_face_corners, axis=0)
        right_endpoint_3d = np.mean(right_face_corners, axis=0)

        left_endpoint_3d = np.array([left_endpoint_3d[0]*1000, left_endpoint_3d[1]*1000, left_endpoint_3d[2]*1000])
        right_endpoint_3d = np.array([right_endpoint_3d[0]*1000, right_endpoint_3d[1]*1000, right_endpoint_3d[2]*1000])


        left_endpoint_base = self.camera_to_ee_transform(left_endpoint_3d)
        right_endpoint_base = self.camera_to_ee_transform(right_endpoint_3d)
        
        max_distance = np.linalg.norm(right_endpoint_3d - left_endpoint_3d)
        edge_direction = right_endpoint_3d - left_endpoint_3d

        edge_yaw = np.arctan2(edge_direction[1], edge_direction[0]) * 180 / np.pi
        
        left_endpoint_2d = self.project_3d_to_2d(left_endpoint_3d)
        right_endpoint_2d = self.project_3d_to_2d(right_endpoint_3d)
        
        return {
            'left_3d': left_endpoint_3d,
            'right_3d': right_endpoint_3d,
            'left_base': left_endpoint_base,
            'right_base': right_endpoint_base,
            'left_2d': left_endpoint_2d,
            'right_2d': right_endpoint_2d,
            'distance': max_distance,
            'edge_yaw': edge_yaw
        }
    
    def _verify_with_clip(self, image_source, final_bbox_2d, final_mask):
        """第七步：CLIP 驗證"""
        crop = image_source[final_bbox_2d[1]:final_bbox_2d[3], 
                           final_bbox_2d[0]:final_bbox_2d[2]].copy()
        crop_mask_2d = final_mask[final_bbox_2d[1]:final_bbox_2d[3], 
                                  final_bbox_2d[0]:final_bbox_2d[2]]
        crop = cv2.bitwise_and(crop, crop, mask=crop_mask_2d)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        inputs = self.clip_processor(
            images=crop_pil, text=self.candidate_phrases, 
            return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            img_feat = self.clip_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            txt_feat = self.clip_model.get_text_features(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"]
            )
            sims = (img_feat @ txt_feat.T).squeeze(0)
            clip_label = self.candidate_phrases[sims.argmax().item()]
        
        return clip_label
    
    def _visualize_detection(self, vis, info, color, W, H, det_idx):
        """第八步：繪製檢測結果"""
        bbox_3d = info['bbox_3d']
        center_base = info['3d_info']['center_base']
        center_3d = info['3d_info']['center_3d']
        size_3d = info['3d_info']['size_3d']
        yaw = info['3d_info']['yaw']
        pitch = info['3d_info']['pitch']
        endpoints = info['endpoints']
        region_type = info['region_type']
        final_label = info['final_label']
        logit = info['logit']
        points_3d = info['points_3d']
        
        # 繪製 3D bbox
        vis = self.visualize_3d_bbox(vis, bbox_3d, color)
        
        # 繪製中心點
        center_2d = self.project_3d_to_2d(center_3d)
        if center_2d is not None:
            cx, cy = center_2d
            cv2.circle(vis, (int(cx), int(cy)), 10, color, -1)
            cv2.circle(vis, (int(cx), int(cy)), 10, (255, 255, 255), 2)
            cv2.putText(vis, "C", (int(cx) - 20, int(cy) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 繪製端點
        left_endpoint_2d = endpoints['left_2d']
        right_endpoint_2d = endpoints['right_2d']
        
        if left_endpoint_2d and right_endpoint_2d:
            Lx, Ly = left_endpoint_2d
            Rx, Ry = right_endpoint_2d
            
            cv2.circle(vis, (Lx, Ly), 8, (0, 255, 0), -1)
            cv2.circle(vis, (Rx, Ry), 8, (0, 0, 255), -1)
            cv2.line(vis, (Lx, Ly), (Rx, Ry), (255, 255, 255), 3)
            cv2.putText(vis, "L", (Lx - 20, Ly - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "R", (Rx + 10, Ry - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 繪製文字資訊
        text = (f"#{det_idx+1} {final_label}: {logit:.2f}\n"
            f"Region: {region_type}\n"
            f"Center: ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})mm\n"
            f"Size: ({size_3d[0]:.3f}, {size_3d[1]:.3f}, {size_3d[2]:.3f})mm\n"
            f"Yaw: {yaw:.1f} deg / Edge: {endpoints['edge_yaw']:.1f} deg\n"
            f"Pitch: {pitch:.1f} deg")
        
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        lines = text.split('\n')
        line_height = cv2.getTextSize("Test", font, fs, th)[0][1] + 5
        
        tx = info['bbox_2d'][0]
        ty = info['bbox_2d'][1] - 10
        
        max_width = max([cv2.getTextSize(line, font, fs, th)[0][0] for line in lines])
        total_height = len(lines) * line_height
        
        tx = max(0, min(tx, W - max_width - 10))
        ty = max(total_height, min(ty, H - 10))
        
        cv2.rectangle(vis, (tx, ty - total_height), (tx + max_width, ty + 5), 
                    color, cv2.FILLED)
        
        for i, line in enumerate(lines):
            y_pos = ty - total_height + (i + 1) * line_height
            cv2.putText(vis, line, (tx, y_pos), font, fs, (0, 0, 0), th, cv2.LINE_AA)
        
        return vis
    
    def _save_object_info(self, info):
        """第九步：儲存物品資訊"""
      
        yaw = info['3d_info']['yaw']
        if yaw <0:
            yaw = yaw+180
        if info['3d_info']['size_3d'][2] > info['3d_info']['size_3d'][0] and info['3d_info']['size_3d'][2] > info['3d_info']['size_3d'][1]:
            pick_mode="side"
        else:
            pick_mode="down"
        obj_info = {
            "name": info['final_label'],
            "region_type": info['region_type'],
            "center_pos": tuple(info['3d_info']['center_base']),
            "3d_size": tuple(info['3d_info']['size_3d']),
            "angle":float(yaw),
            "left_endpoint": tuple(info['endpoints']['left_base']),
            "right_endpoint": tuple(info['endpoints']['right_base']),
            "camera_id": self.camera_id,
            "confidence": float(info['logit']),
            "pick_mode":pick_mode,
            "center_vector": tuple(info.get('center_vector', (None, None, None))),
            "endpoints_total_obj": tuple(info.get('endpoints_total_obj', (None, None, None))),
            "longest_length": float(info.get('longest_length', None)),
            "shortest_length": float(info.get('shortest_length', None)),
        }
        
        self.objects_info.append(obj_info)
    
    # ========================================
    # 主要偵測函數
    # ========================================
    

    def _calculate_unit_vector(self, pointA, pointB):
        """計算兩點之間的向量"""
        vector = np.array(pointB) - np.array(pointA)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def detect_objects(self):
        
        self.handle_reference_features = shared_models.get_handle_features(self.candidate_phrases[0])

        """對當前畫面進行物品偵測並計算3D資訊（CLIP 握柄比對版本）"""
        if self.latest_color_image is None:
            print(f"❌ 相機 {self.camera_id}: 沒有可用的影像進行偵測")
            return False
            
        print(f"\n🔍 相機 {self.camera_id} 開始物品偵測（最多 {self.max_objects} 個物品）...")
        
        # 載入和準備影像
        temp_image_path = f"/home/gairobots/camera/GroundingDINO/data/tmp/temp_detect_camera_{self.camera_id}.png"
        cv2.imwrite(temp_image_path, self.latest_color_image)
        image_source, image = load_image(temp_image_path)
        
        rgb_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        gc.collect()
        torch.cuda.empty_cache()
        with self._sam_lock:
            self.predictor.set_image(rgb_image)
        
        # 第一步：初始偵測
        with torch.no_grad():
            all_detections = self._get_initial_detections(image_source, image)
        if len(all_detections) == 0:
            print(f"   ❌ 相機 {self.camera_id}: 未偵測到任何物品")
            return False
        
        # 第二步：篩選
        selected_detections = self._filter_detections(all_detections)
        
        # 視覺化準備
        vis = image_source.copy()
        random.seed(42)
        any_detected = False
        H, W = image_source.shape[:2]
        
        colors_list = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
            (0, 128, 255), (128, 255, 0)
        ]
        
        # 處理每個檢測
        for det_idx, detection in enumerate(selected_detections):
            x1, y1, x2, y2 = detection['bbox_2d']
            logit = detection['logit']
            phrase = detection['phrase']
            color = colors_list[det_idx % len(colors_list)]
            
            print(f"\n   處理第 {det_idx + 1} 個物品: {phrase}，信心度: {logit:.2f}")
            
            # 第三步：檢測整體物體
            result = self._detect_object_region(image_source, x1, y1, x2, y2, det_idx)
            if result[0] is None:
                gc.collect()
                torch.cuda.empty_cache()
                continue
            bbox_3d_object, object_mask, points_3d_object, crop_image = result
            
            # 第四步：檢測握柄或回退到整體物體
            region_info = self._detect_handle_region(
                image_source, crop_image, x1, y1, x2, y2,
                bbox_3d_object, object_mask, points_3d_object, det_idx
            )
            
            bbox_3d = region_info['bbox_3d'] # 握柄3d bounding box
            final_mask = region_info['mask']
            final_bbox_2d = region_info['bbox_2d']
            points_3d = region_info['points_3d']
            region_type = region_info['region_type']
            
            # 第五步：提取 3D 資訊
            info_3d = self._extract_3d_info(bbox_3d)
            gc.collect()
            torch.cuda.empty_cache()

            print(f"      提取 3D 資訊（{region_type}）...")
            print(f"      3D中心: ({info_3d['center_base'][0]:.3f}, {info_3d['center_base'][1]:.3f}, {info_3d['center_base'][2]:.3f}) mm")
            print(f"      3D尺寸: ({info_3d['size_3d'][0]:.3f}, {info_3d['size_3d'][1]:.3f}, {info_3d['size_3d'][2]:.3f}) mm")
            
            total_center = np.mean(bbox_3d_object['corners'], axis=0) * 1000.0 #mm
            handle_center = info_3d['center_3d'] 
            center_vector = self._calculate_unit_vector(handle_center, total_center)

            # 第六步：計算端點
            endpoints = self._calculate_endpoints(bbox_3d, info_3d['center_3d'])
            endpoints_total_obj = self._calculate_endpoints(bbox_3d_object, total_center)

            print(f"     整體物體左端點（mm）: （{endpoints_total_obj['left_base'][0]:.3f}, {endpoints_total_obj['left_base'][1]:.3f}, {endpoints_total_obj['left_base'][2]:.3f}）")
            print(f"     整體物體右端點（mm）: （{endpoints_total_obj['right_base'][0]:.3f}, {endpoints_total_obj['right_base'][1]:.3f}, {endpoints_total_obj['right_base'][2]:.3f}）")            
            print(f"      左端點基座: ({endpoints['left_base'][0]:.3f}, {endpoints['left_base'][1]:.3f}, {endpoints['left_base'][2]:.3f}) mm")
            print(f"      右端點基座: ({endpoints['right_base'][0]:.3f}, {endpoints['right_base'][1]:.3f}, {endpoints['right_base'][2]:.3f}) mm")
            
            # 將 tuple 轉換為 numpy array 再計算
            left_array_base = np.array([endpoints_total_obj['left_base'][0], endpoints_total_obj['left_base'][1]])
            right_array_base = np.array([endpoints_total_obj['right_base'][0], endpoints_total_obj['right_base'][1]])
            center_array_base = np.array([info_3d['center_base'][0], info_3d['center_base'][1]])

            length1 = np.linalg.norm(left_array_base - center_array_base)
            length2 = np.linalg.norm(right_array_base - center_array_base)
            longest_length = max(length1, length2)
            shortest_length = min(length1, length2)
            print(f"      物體中心到長邊端點最長距離: {longest_length:.3f} mm")
            print(f"      物體中心到長邊端點最短距離: {shortest_length:.3f} mm")
            # 第七步：CLIP 驗證
            print(f"      CLIP 驗證...")
            with torch.no_grad():
                final_label = self._verify_with_clip(image_source, final_bbox_2d, final_mask)
            
            any_detected = True
            
            # 收集所有資訊
            detection_info = {
                'bbox_3d': bbox_3d,
                'bbox_2d': final_bbox_2d,
                '3d_info': info_3d,
                'endpoints': endpoints,
                'region_type': region_type,
                'final_label': final_label,
                'logit': logit,
                'points_3d': points_3d,
                'center_vector': center_vector,
                'endpoints_total_obj': endpoints_total_obj,
                'longest_length': longest_length,
                'shortest_length': shortest_length,

            }
            
            # 第八步：視覺化
            vis = self._visualize_detection(vis, detection_info, color, W, H, det_idx)
            
            # 第九步：儲存資訊
            self._save_object_info(detection_info)
        
        # 儲存結果
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(result_path, vis_bgr)
        result_path = os.path.join(self.output_dir, "detection_3d.jpg")
        cv2.imwrite(result_path, vis_bgr)

        if not any_detected:
            print(f"   ❌ 相機 {self.camera_id}: 未偵測到任何物品")
            self.clear_detection_data()
            
            return False
        else:
            print(f"\n   ✅ 相機 {self.camera_id}: 3D偵測結果已儲存: {result_path}")
            print(f"   成功檢測到 {len(self.objects_info)} 個物品")
            self.clear_detection_data()
        
            return True

    
    def detect_objects_simple(self):
        
        """對當前畫面進行物品偵測並計算3D資訊（簡化版：僅整體物體檢測）"""
        if self.latest_color_image is None:
            print(f"❌ 相機 {self.camera_id}: 沒有可用的影像進行偵測")
            return False
            
        print(f"\n🔍 相機 {self.camera_id} 開始物品偵測（簡化版 - 最多 {self.max_objects} 個物品）...")
        print(f"phase list: {self.candidate_phrases}")
        
        # 載入和準備影像
        temp_image_path = f"/home/gairobots/camera/GroundingDINO/data/tmp/temp_detect_camera_{self.camera_id}.png"
        cv2.imwrite(temp_image_path, self.latest_color_image)
        image_source, image = load_image(temp_image_path)
        
        rgb_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

        gc.collect()
        torch.cuda.empty_cache()
        with self._sam_lock:
            self.predictor.set_image(rgb_image)
        
        # 第一步：初始偵測
        with torch.no_grad():
            all_detections = self._get_initial_detections(image_source, image)
        if len(all_detections) == 0:
            print(f"   ❌ 相機 {self.camera_id}: 未偵測到任何物品")
            return False
        
        # 第二步：篩選
        selected_detections = self._filter_detections(all_detections)
        
        # 視覺化準備
        vis = image_source.copy()
        random.seed(42)
        any_detected = False
        H, W = image_source.shape[:2]
        
        colors_list = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
            (0, 128, 255), (128, 255, 0)
        ]
        
        # 處理每個檢測
        for det_idx, detection in enumerate(selected_detections):
            x1, y1, x2, y2 = detection['bbox_2d']
            logit = detection['logit']
            phrase = detection['phrase']
            color = colors_list[det_idx % len(colors_list)]
            
            print(f"\n   處理第 {det_idx + 1} 個物品: {phrase}，信心度: {logit:.2f}")
            
            # 簡化版：僅檢測整體物體（跳過握柄檢測）
            print(f"      檢測整體物體 3D bbox...")
            
            # 使用 SAM 生成 mask
            with torch.no_grad():
                masks_obj, _, _ = self.predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False
                )
            object_mask = masks_obj[0].astype(np.uint8) * 255

            del masks_obj
            gc.collect()
            torch.cuda.empty_cache()

            mask_path = os.path.join(self.output_dir, f"mask_simple_{det_idx}.jpg")
            cv2.imwrite(mask_path, object_mask)
            
            # 計算 3D bbox
            points_3d = self.depth_to_point_cloud(
                self.depth_image, object_mask
            )
            
            if len(points_3d) < 10:
                print(f"      ❌ 點雲不足，跳過")
                del object_mask, points_3d
                gc.collect()
                torch.cuda.empty_cache()
                continue
            
            bbox_3d = self.compute_3d_bounding_box(points_3d)
            
            if bbox_3d is None:
                print(f"      ❌ 無法計算 3D bbox，跳過")
                del object_mask, points_3d
                gc.collect()
                torch.cuda.empty_cache()
                continue
            
            print(f"      ✓ 3D bbox 計算成功 ({len(points_3d)} 個點)")
            
            # 第五步：提取 3D 資訊
            info_3d = self._extract_3d_info(bbox_3d)
            
            print(f"      提取 3D 資訊...")
            print(f"      3D中心: ({info_3d['center_base'][0]:.3f}, {info_3d['center_base'][1]:.3f}, {info_3d['center_base'][2]:.3f}) mm")
            print(f"      3D尺寸: ({info_3d['size_3d'][0]:.3f}, {info_3d['size_3d'][1]:.3f}, {info_3d['size_3d'][2]:.3f}) mm")
            
            # 第六步：計算端點
            endpoints = self._calculate_endpoints(bbox_3d, info_3d['center_3d'])
            
            print(f"      左端點基座: ({endpoints['left_base'][0]:.3f}, {endpoints['left_base'][1]:.3f}, {endpoints['left_base'][2]:.3f}) mm")
            print(f"      右端點基座: ({endpoints['right_base'][0]:.3f}, {endpoints['right_base'][1]:.3f}, {endpoints['right_base'][2]:.3f}) mm")
            print(f"      長邊長度: {endpoints['distance']:.3f} m")
            
            # 第七步：CLIP 驗證
            print(f"      CLIP 驗證...")
            with torch.no_grad():
                final_label = self._verify_with_clip(image_source, (x1, y1, x2, y2), object_mask)
            
            any_detected = True
            
            # 收集所有資訊（簡化版無 region_type 和握柄資訊）
            detection_info = {
                'bbox_3d': bbox_3d,
                'bbox_2d': (x1, y1, x2, y2),
                '3d_info': info_3d,
                'endpoints': endpoints,
                'region_type': 'object',  # 簡化版固定為 object
                'final_label': final_label,
                'logit': logit,
                'points_3d': points_3d
            }
            
            # 第八步：視覺化（簡化版輸出）
            vis = self._visualize_detection_simple(vis, detection_info, color, W, H, det_idx)
            
            # 第九步：儲存資訊（簡化版）
            self._save_object_info_simple(detection_info)
        
        # 儲存結果
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        result_path = os.path.join(self.output_dir, "detection_3d_simple.jpg")
        cv2.imwrite(result_path, vis_bgr)
        
        if not any_detected:
            print(f"   ❌ 相機 {self.camera_id}: 未偵測到任何物品")
            self.clear_detection_data()
            return False
        else:
            print(f"\n   ✅ 相機 {self.camera_id}: 3D偵測結果（簡化版）已儲存: {result_path}")
            print(f"   成功檢測到 {len(self.objects_info)} 個物品")
            self.clear_detection_data()
            return True
    
    def _visualize_detection_simple(self, vis, info, color, W, H, det_idx):
        """簡化版繪製檢測結果（無握柄資訊）"""
        bbox_3d = info['bbox_3d']
        center_base = info['3d_info']['center_base']
        center_3d = info['3d_info']['center_3d']
        size_3d = info['3d_info']['size_3d']
        yaw = info['3d_info']['yaw']
        pitch = info['3d_info']['pitch']
        endpoints = info['endpoints']
        final_label = info['final_label']
        logit = info['logit']
        
        # 繪製 3D bbox
        vis = self.visualize_3d_bbox(vis, bbox_3d, color)
        
        # 繪製中心點
        center_2d = self.project_3d_to_2d(center_3d)
        if center_2d is not None:
            cx, cy = center_2d
            cv2.circle(vis, (int(cx), int(cy)), 10, color, -1)
            cv2.circle(vis, (int(cx), int(cy)), 10, (255, 255, 255), 2)
            cv2.putText(vis, "C", (int(cx) - 20, int(cy) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # 繪製端點
        left_endpoint_2d = endpoints['left_2d']
        right_endpoint_2d = endpoints['right_2d']
        
        if left_endpoint_2d and right_endpoint_2d:
            Lx, Ly = left_endpoint_2d
            Rx, Ry = right_endpoint_2d
            
            cv2.circle(vis, (Lx, Ly), 8, (0, 255, 0), -1)
            cv2.circle(vis, (Rx, Ry), 8, (0, 0, 255), -1)
            cv2.line(vis, (Lx, Ly), (Rx, Ry), (255, 255, 255), 3)
            cv2.putText(vis, "L", (Lx - 20, Ly - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(vis, "R", (Rx + 10, Ry - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 簡化版文字資訊（無 Region 和 Edge Yaw）
        text = (f"#{det_idx+1} {final_label}: {logit:.2f}\n"
            f"Center: ({center_base[0]:.3f}, {center_base[1]:.3f}, {center_base[2]:.3f})mm\n"
            f"Size: ({size_3d[0]:.3f}, {size_3d[1]:.3f}, {size_3d[2]:.3f})mm\n"
            f"Yaw: {yaw:.1f} deg\n"
            f"Pitch: {pitch:.1f} deg")
        
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        lines = text.split('\n')
        line_height = cv2.getTextSize("Test", font, fs, th)[0][1] + 5
        
        tx = info['bbox_2d'][0]
        ty = info['bbox_2d'][1] - 10
        
        max_width = max([cv2.getTextSize(line, font, fs, th)[0][0] for line in lines])
        total_height = len(lines) * line_height
        
        tx = max(0, min(tx, W - max_width - 10))
        ty = max(total_height, min(ty, H - 10))
        
        cv2.rectangle(vis, (tx, ty - total_height), (tx + max_width, ty + 5), 
                    color, cv2.FILLED)
        
        for i, line in enumerate(lines):
            y_pos = ty - total_height + (i + 1) * line_height
            cv2.putText(vis, line, (tx, y_pos), font, fs, (0, 0, 0), th, cv2.LINE_AA)
        
        return vis
    
    def _save_object_info_simple(self, info):
        yaw = info['3d_info']['yaw']
        if yaw <0:
            yaw = yaw+180
        
        if info['3d_info']['size_3d'][2] > info['3d_info']['size_3d'][0] and info['3d_info']['size_3d'][2] > info['3d_info']['size_3d'][1]:
            pick_mode="side"
        else:
            pick_mode="down"
        
        """簡化版儲存物品資訊（無握柄資訊）"""
        obj_info = {
            "name": info['final_label'],
            "region_type": "object",
            "center_pos": tuple(info['3d_info']['center_base']),
            "3d_size": tuple(info['3d_info']['size_3d']),
            "angle":float(yaw),
            "left_endpoint": tuple(info['endpoints']['left_base']),
            "right_endpoint": tuple(info['endpoints']['right_base']),
            "camera_id": self.camera_id,
            "confidence": float(info['logit']),
            "pick_mode":pick_mode,
        }
        
        self.objects_info.append(obj_info)


    
    def get_objects_info(self):
        """取得偵測結果"""
        print(f"\n📋 相機 {self.camera_id} 物品偵測結果:")
        for obj in self.objects_info:
            print(f" - {obj['name']}: {obj['confidence']:.2f}")
            print(f"   3D中心: {obj['center_pos']}, 尺寸: {obj['3d_size']}, 方向: {obj['angle']}")
           
        return self.objects_info
    
    def cleanup(self):
        """清理資源"""
        if self.pipeline:
            self.pipeline.stop()
        print(f"✓ 相機 {self.camera_id} 已釋放")
        shared_models.release()
    def clear_detection_data(self):
        """清理偵測快取資料"""
 
        
        # 清空物品資訊列表
        self.objects_info.clear()
        
        # 清空圖像快取
        self.latest_color_image = None
        self.latest_depth_frame = None
        self.depth_image = None
        
        # 🔑 關鍵：清理 SAM predictor 的圖像快取
        if hasattr(self, 'predictor') and hasattr(self.predictor, 'model'):
            try:
                # 清理 image encoder 的快取
                self.predictor.model.image_encoder.features = None
                self.predictor.image_embedding = None
                
                # 重置 image_size
                self.predictor.image_size = None
                
            except Exception as e:
                pass
        
        # 垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        print(f"✓ 相機 {self.camera_id} 的偵測資料已清理")

    def clear_detection_data(self):
        """清理檢測過程中的所有暫存資料（徹底版）"""
        
        
        print(f"   [清理] 開始清理相機 {self.camera_id} 的檢測資料...")
        
        # 記錄清理前的記憶體
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1024**2
            reserved_before = torch.cuda.memory_reserved() / 1024**2
            print(f"   [清理] 清理前 - 已分配: {mem_before:.1f}MB, 已保留: {reserved_before:.1f}MB")
        
        # 1️⃣ 清理檢測結果
        if hasattr(self, 'detected_objects'):
            for obj in self.detected_objects:
                for key in list(obj.keys()):
                    if key in ['mask', 'depth_data', 'point_cloud', 'image', 'crop_image', 
                            'bbox_3d', 'points_3d', 'handle_features']:
                        try:
                            del obj[key]
                        except:
                            pass
                obj.clear()
            self.detected_objects.clear()
            del self.detected_objects
            self.detected_objects = []
        
        # 2️⃣ ⭐ 關鍵：完全重置 SAM predictor（重新創建）
        if hasattr(self, 'predictor') and self.predictor is not None:
            # 嘗試官方重置
            try:
                self.predictor.reset_image()
            except:
                pass
            
            # ⭐ 刪除 predictor 的所有內部張量
            if hasattr(self.predictor, 'features'):
                del self.predictor.features
            if hasattr(self.predictor, 'original_size'):
                del self.predictor.original_size
            if hasattr(self.predictor, 'input_size'):
                del self.predictor.input_size
            if hasattr(self.predictor, 'is_image_set'):
                self.predictor.is_image_set = False
            
            # ⭐ 關鍵：刪除 image_embedding（最大的記憶體佔用）
            if hasattr(self.predictor, 'image_embedding'):
                del self.predictor.image_embedding
                self.predictor.image_embedding = None
            
            # 清除 interm_features（中間特徵）
            if hasattr(self.predictor, 'interm_features'):
                if self.predictor.interm_features is not None:
                    for feat in self.predictor.interm_features:
                        if feat is not None:
                            del feat
                self.predictor.interm_features = None
        
        # 3️⃣ 清理可能的 GroundingDINO 快取
        # 某些版本的 GroundingDINO 會快取特徵
        if hasattr(self, 'gd_model'):
            if hasattr(self.gd_model, 'cache'):
                try:
                    self.gd_model.cache.clear()
                except:
                    pass
        
        # 4️⃣ 清理 CLIP 的快取特徵
        if hasattr(self, 'handle_reference_features'):
            # 不刪除握柄特徵（需要重複使用），但清除臨時副本
            if hasattr(self, '_temp_handle_features'):
                del self._temp_handle_features
        
        # 5️⃣ 強制 Python 垃圾回收（執行多次）
        print(f"   [清理] 執行垃圾回收...")
        total_collected = 0
        for i in range(5):
            collected = gc.collect()
            total_collected += collected
            if collected > 0:
                print(f"   [清理]   第 {i+1} 次 GC: 回收 {collected} 個物件")
        print(f"   [清理] 總共回收 {total_collected} 個物件")
        
        # 6️⃣ ⭐ 關鍵：強制清空 CUDA 快取（多次）
        if torch.cuda.is_available():
            print(f"   [清理] 清空 CUDA 快取...")
            
            # 執行多次 empty_cache（確保徹底清理）
            for _ in range(5):
                torch.cuda.empty_cache()
            
            # 同步所有 CUDA 操作
            torch.cuda.synchronize()
            
            # ⭐ 嘗試重置記憶體統計（有時有幫助）
            try:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()
            except:
                pass
            
            # 記錄清理後的記憶體
            mem_after = torch.cuda.memory_allocated() / 1024**2
            reserved_after = torch.cuda.memory_reserved() / 1024**2
            mem_freed = mem_before - mem_after
            reserved_freed = reserved_before - reserved_after
            
            print(f"   [清理] 清理後 - 已分配: {mem_after:.1f}MB, 已保留: {reserved_after:.1f}MB")
            print(f"   [清理] 釋放 - 已分配: {mem_freed:.1f}MB, 已保留: {reserved_freed:.1f}MB")
            
            # 如果釋放的記憶體很少，發出警告
            if mem_freed < 50:
                print(f"   ⚠️  警告: 只釋放了 {mem_freed:.1f}MB，可能有記憶體洩漏")
                print(f"   ⚠️  保留但未分配: {reserved_after:.1f}MB")
        
        print(f"   ✓ 相機 {self.camera_id} 的檢測資料已清理")

