import torch
from groundingdino.util.inference import load_model, load_image, predict, annotate
import random
import os
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import pyrealsense2 as rs
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torchvision import transforms
import timm
import torch.backends.cudnn as cudnn
from groundingdino.models.GroundingDINO import transformer
import torch.utils.checkpoint as checkpoint

# 禁用 checkpoint 函數
original_checkpoint = checkpoint.checkpoint
checkpoint.checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)
transformer.checkpoint.checkpoint = lambda f, *args, **kwargs: f(*args, **kwargs)
# 禁用所有 checkpoint 操作
# transformer.checkpoint = None
# object_device = torch.device("cpu")
cudnn.enabled = False
cudnn.benchmark = False

object_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用設備: {object_device}")
print(f"cuDNN 啟用: {cudnn.enabled}")

# 初始化模型

CONFIG_PATH = "/home/gairobots/camera/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "/home/gairobots/camera/GroundingDINO/weights/groundingdino_swint_ogc.pth"


Dino_MODEL = load_model(CONFIG_PATH, WEIGHTS_PATH, device=object_device)

SAM = sam_model_registry["default"](checkpoint="/home/gairobots/camera/GroundingDINO/sam_checkpoints/sam_vit_h_4b8939.pth")
SAM.to(object_device)
SAM_PREDICTOR = SamPredictor(SAM)

# os.environ["FORCE_CPU"] = "1"
# USE_PYTHON_IMPL = True

class CameraDetector:
    def __init__(self, realsense_number="923322070636", max_objects=2, candidate_phrases=[
           "tool", "blue vacuum cleaner", "broom", "dustpan", "brush tool",
        ]):  # 新增參數：最大檢測物品數量
        # 初始化模型
        self.gd_model = Dino_MODEL
        self.predictor = SAM_PREDICTOR
        self.device = object_device
        
        # CLIP 模型
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        

        
        self.realsense_number  = realsense_number
        self.init_camera()
        
        # 偵測設定
        
        self.candidate_phrases = candidate_phrases
        self.caption = " . ".join(self.candidate_phrases)
        
        # 新增：最大檢測物品數量
        self.max_objects = max_objects
        
        # 相機狀態
        self.running = True
        self.latest_color_image = None
        self.latest_depth_frame = None
        self.depth_image = None
        self.detection_count = 0
        self.detected_threshold = 5
        self.objects_info = []



        #  # 載入 DINO ViT-S/8 模型
        # print("載入 DINO ViT-S/8 模型...")
        # self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        # self.dino_model.to(self.device)
        # self.dino_model.eval()
        
        # # DINO 參數（論文設定）
        # self.dino_patch_size = 8
        # self.num_keypoints = 10
        # self.dino_threshold = 0.15
        # # 圖像預處理
        # self.dino_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
        #                        std=[0.229, 0.224, 0.225])
        # ])
      
    
        # 👇 新增：握柄特徵
        self.handle_reference_features = None
         # 嘗試從文件載入，如果不存在則為 None
        if not self.load_handle_features_from_file():
            print("ℹ️  提示：還未收集握柄特徵，請先執行 collect_handle_reference_interactive()")

    def add_demonstration_image(self, rgb_image, mask=None):
        """
        加入一張示範圖像（手動抓取握柄時拍攝）
        
        參數:
            rgb_image: RGB 影像
            mask: 物體遮罩（可選，用於只提取物體區域特徵）
        """
        if mask is not None:
            rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
        
        # 提取 DINO 特徵
        features, (h, w) = self.extract_dino_features(rgb_image)
        
        self.demonstration_images.append(rgb_image)
        self.demonstration_features.append({
            'features': features,
            'shape': (h, w)
        })
        
        print(f"已加入第 {len(self.demonstration_images)} 張示範圖像")    
    def extract_dino_features(self, image_rgb):
        """
        提取 DINO 特徵描述符
        
        參數:
            image_rgb: RGB 影像 (H, W, 3), numpy array
        
        返回:
            features: 特徵圖 (num_patches_h, num_patches_w, feature_dim)
        """
        # 調整大小（保持縱橫比，最短邊為 224）
        h, w = image_rgb.shape[:2]
        scale = 224 / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 確保尺寸是 patch_size 的倍數
        new_h = (new_h // self.dino_patch_size) * self.dino_patch_size
        new_w = (new_w // self.dino_patch_size) * self.dino_patch_size
        
        image_resized = cv2.resize(image_rgb, (new_w, new_h))
        
        # 轉為 tensor
        img_tensor = self.dino_transform(image_resized).unsqueeze(0).to(self.device)
        
        # 提取特徵
        with torch.no_grad():
            # DINO 輸出：[B, num_patches + 1, feature_dim]
            features = self.dino_model.get_intermediate_layers(img_tensor, n=1)[0]
            
            # 移除 CLS token
            features = features[:, 1:, :]  # [1, num_patches, 384]
            
            # Reshape 為 2D 特徵圖
            num_patches_h = new_h // self.dino_patch_size
            num_patches_w = new_w // self.dino_patch_size
            
            features = features.reshape(1, num_patches_h, num_patches_w, -1)
            features = features.squeeze(0)  # [num_patches_h, num_patches_w, 384]
        
        return features, (new_h, new_w)
    
    def find_correspondences(self, features, query_point, threshold=0.15, top_k=10):
        """
        找到與查詢點相似的關鍵點
        
        參數:
            features: 特徵圖 (H, W, D)
            query_point: 查詢點座標 (y, x) 在 patch 空間
            threshold: 相似度閾值
            top_k: 返回 top-k 個點
        
        返回:
            keypoints: [(y, x), ...] 在 patch 空間的座標
            similarities: 對應的相似度分數
        """
        qy, qx = query_point
        H, W, D = features.shape
        
        # 確保查詢點在範圍內
        qy = np.clip(qy, 0, H - 1)
        qx = np.clip(qx, 0, W - 1)
        
        # 提取查詢特徵
        query_feat = features[qy, qx].unsqueeze(0)  # [1, D]
        
        # 計算所有點的相似度
        features_flat = features.reshape(-1, D)  # [H*W, D]
        
        # 余弦相似度
        similarities = F.cosine_similarity(
            features_flat.unsqueeze(1),  # [H*W, 1, D]
            query_feat.unsqueeze(0),     # [1, 1, D]
            dim=2
        ).squeeze(1)  # [H*W]
        
        # 過濾低相似度
        valid_mask = similarities > threshold
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return [], []
        
        # 選擇 top-k
        valid_sims = similarities[valid_indices]
        top_k = min(top_k, len(valid_indices))
        top_values, top_idx = torch.topk(valid_sims, k=top_k)
        
        # 轉為 2D 座標
        selected_indices = valid_indices[top_idx]
        keypoints = []
        
        for idx in selected_indices:
            idx_val = idx.item()
            py = idx_val // W
            px = idx_val % W
            keypoints.append((py, px))
        
        return keypoints, top_values.cpu().numpy()
    
    def extract_dino_keypoints(self, image_crop, mask_crop, bbox_2d):
        """
        使用 DINO 提取物體關鍵點（完整實現）
        
        參數:
            image_crop: 裁切後的影像 (BGR)
            mask_crop: 對應遮罩
            bbox_2d: 原始 bounding box (x1, y1, x2, y2)
        
        返回:
            keypoints_2d: [(x, y), ...] 原始圖像座標
            keypoints_3d: [(X, Y, Z), ...] 相機座標系
        """
        # 1. 轉 RGB 並應用遮罩
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_crop)
        
        # 2. 提取 DINO 特徵
        features, (resized_h, resized_w) = self.extract_dino_features(masked_image)
        
        # 3. 找到遮罩中心點（作為查詢點）
        crop_h, crop_w = mask_crop.shape
        moments = cv2.moments(mask_crop)
        
        if moments['m00'] == 0:
            # 使用幾何中心
            center_y, center_x = crop_h // 2, crop_w // 2
        else:
            # 使用質心
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])
        
        # 映射到 patch 空間
        num_patches_h = resized_h // self.dino_patch_size
        num_patches_w = resized_w // self.dino_patch_size
        
        query_py = int((center_y / crop_h) * num_patches_h)
        query_px = int((center_x / crop_w) * num_patches_w)
        
        # 4. 找到相似關鍵點
        keypoints_patch, similarities = self.find_correspondences(
            features, 
            (query_py, query_px),
            threshold=self.dino_threshold,
            top_k=self.num_keypoints
        )
        
        if not keypoints_patch:
            print("警告：未找到符合閾值的 DINO 關鍵點")
            return [], []
        
        # 5. 轉回原始圖像座標
        x1, y1, x2, y2 = bbox_2d
        keypoints_2d = []
        keypoints_3d = []
        
        for py, px in keypoints_patch:
            # patch -> crop 座標
            kp_y = int((py / num_patches_h) * crop_h)
            kp_x = int((px / num_patches_w) * crop_w)
            
            # crop -> 原始圖像座標
            orig_x = x1 + kp_x
            orig_y = y1 + kp_y
            
            # 確保在範圍內
            orig_x = np.clip(orig_x, 0, self.depth_image.shape[1] - 1)
            orig_y = np.clip(orig_y, 0, self.depth_image.shape[0] - 1)
            
            keypoints_2d.append((orig_x, orig_y))
            
            # 轉 3D
            # 使用 3x3 鄰域的中值深度（更穩定）
            y_start = max(0, orig_y - 1)
            y_end = min(self.depth_image.shape[0], orig_y + 2)
            x_start = max(0, orig_x - 1)
            x_end = min(self.depth_image.shape[1], orig_x + 2)
            
            depth_window = self.depth_image[y_start:y_end, x_start:x_end]
            valid_depths = depth_window[(depth_window > 0) & (depth_window < 3000)]
            
            if len(valid_depths) > 0:
                depth_value = np.median(valid_depths) / 1000.0
                
                X = (orig_x - self.intrinsics.ppx) * depth_value / self.intrinsics.fx
                Y = (orig_y - self.intrinsics.ppy) * depth_value / self.intrinsics.fy
                Z = depth_value
                
                keypoints_3d.append((X, Y, Z))
            else:
                keypoints_3d.append(None)
        
        return keypoints_2d, keypoints_3d


    def get_median_depth(self, x, y, window_size=3):
        """取得穩健的深度值"""
        y_start = max(0, y - window_size // 2)
        y_end = min(self.depth_image.shape[0], y + window_size // 2 + 1)
        x_start = max(0, x - window_size // 2)
        x_end = min(self.depth_image.shape[1], x + window_size // 2 + 1)
        
        depth_window = self.depth_image[y_start:y_end, x_start:x_end]
        valid_depths = depth_window[(depth_window > 0) & (depth_window < 3000)]
        
        if len(valid_depths) > 0:
            return np.median(valid_depths) / 1000.0
        return None
    def extract_cross_image_keypoints(self, query_rgb, query_mask, bbox_2d, K=10):
        """
        論文方法：透過跨圖像匹配找到任務相關的關鍵點
        
        參數:
            query_rgb: 新圖像（BGR）
            query_mask: 物體遮罩
            bbox_2d: bounding box
            K: 關鍵點數量
        
        返回:
            keypoints_2d: 在新圖像中的關鍵點座標
            keypoints_3d: 對應的 3D 座標
        """
        if len(self.demonstration_features) == 0:
            print("警告：沒有示範圖像，無法使用跨圖像匹配")
            return [], []
        
        # 1. 提取新圖像的 DINO 特徵
        query_rgb_masked = cv2.bitwise_and(query_rgb, query_rgb, mask=query_mask)
        query_features, (query_h, query_w) = self.extract_dino_features(query_rgb_masked)
        
        num_patches_h = query_h // self.dino_patch_size
        num_patches_w = query_w // self.dino_patch_size
        
        # 2. 對新圖像的每個 patch，找到在示範圖像中最相似的對應點
        query_features_flat = query_features.reshape(-1, query_features.shape[-1])  # [N_query, D]
        
        # 儲存每個 query patch 的最大相似度（與所有示範圖像比較）
        max_similarities = torch.zeros(query_features_flat.shape[0])
        
        for demo_feat_dict in self.demonstration_features:
            demo_features = demo_feat_dict['features']
            demo_features_flat = demo_features.reshape(-1, demo_features.shape[-1])  # [N_demo, D]
            
            # 計算 query 與 demo 之間的相似度矩陣 [N_query, N_demo]
            sim_matrix = F.cosine_similarity(
                query_features_flat.unsqueeze(1),  # [N_query, 1, D]
                demo_features_flat.unsqueeze(0),   # [1, N_demo, D]
                dim=2
            )
            
            # 對每個 query patch，取與該示範圖像的最大相似度
            max_sim_per_demo, _ = torch.max(sim_matrix, dim=1)  # [N_query]
            
            # 累積最大相似度
            max_similarities = torch.maximum(max_similarities, max_sim_per_demo)
        
        # 3. 選擇相似度最高的 K 個 patch 作為關鍵點
        top_k_values, top_k_indices = torch.topk(max_similarities, k=min(K, len(max_similarities)))
        
        # 過濾低相似度（論文中閾值 0.12）
        valid_mask = top_k_values > 0.12
        selected_indices = top_k_indices[valid_mask]
        
        if len(selected_indices) == 0:
            print("警告：未找到高相似度的跨圖像關鍵點")
            return [], []
        
        # 4. 轉為 2D 座標並投影到 3D
        x1, y1, x2, y2 = bbox_2d
        crop_h, crop_w = query_mask.shape
        
        keypoints_2d = []
        keypoints_3d = []
        
        for idx in selected_indices:
            idx_val = idx.item()
            py = idx_val // num_patches_w
            px = idx_val % num_patches_w
            
            # patch -> crop 座標
            kp_y = int((py / num_patches_h) * crop_h)
            kp_x = int((px / num_patches_w) * crop_w)
            
            # crop -> 原始圖像座標
            orig_x = x1 + kp_x
            orig_y = y1 + kp_y
            
            orig_x = np.clip(orig_x, 0, self.depth_image.shape[1] - 1)
            orig_y = np.clip(orig_y, 0, self.depth_image.shape[0] - 1)
            
            keypoints_2d.append((orig_x, orig_y))
            
            # 轉 3D（使用鄰域中值深度）
            depth_value = self.get_median_depth(orig_x, orig_y)
            
            if depth_value is not None:
                X = (orig_x - self.intrinsics.ppx) * depth_value / self.intrinsics.fx
                Y = (orig_y - self.intrinsics.ppy) * depth_value / self.intrinsics.fy
                Z = depth_value
                keypoints_3d.append((X, Y, Z))
            else:
                keypoints_3d.append(None)
        
        print(f"跨圖像匹配找到 {len(keypoints_2d)} 個任務相關關鍵點")
        
        return keypoints_2d, keypoints_3d
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
# list_cameras()
    def init_camera(self):
        """初始化相機"""

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.realsense_number)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) 

        self.profile = self.pipeline.start(config)
        # 對齊深度與彩色影像
        self.align = rs.align(rs.stream.color)
        
        # 獲取相機內參
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        # 跳過前40幀讓相機穩定
        for i in range(40):
            self.pipeline.wait_for_frames()
    
    def depth_to_point_cloud(self, depth_image, mask, intrinsics):
        """將深度圖和遮罩轉換為3D點雲"""
        points_3d = []
        
        # 獲取相機內參
        fx, fy = intrinsics.fx, intrinsics.fy
        cx, cy = intrinsics.ppx, intrinsics.ppy
        
        # 找到遮罩中的所有像素
        ys, xs = np.where(mask > 0)
        
        for x, y in zip(xs, ys):
            # 獲取深度值（公尺）
            depth = depth_image[y, x] / 1000.0  # 轉換為公尺
            
            if depth == 0 or depth > 3.0:  # 過濾無效深度
                continue
            
            # 深度轉3D座標（相機坐標系，針孔成像原理）
            X = (x - cx) * depth / fx
            Y = (y - cy) * depth / fy
            Z = depth
            
            points_3d.append([X, Y, Z])
        
        return np.array(points_3d)

    def compute_3d_bounding_box(self, points_3d):
        """使用PCA計算oriented 3D bounding box"""
        if len(points_3d) < 10:
            return None
        
        # 計算質心
        centroid = np.mean(points_3d, axis=0)
        
        # PCA 找主方向
        points_centered = points_3d - centroid
        pca = PCA(n_components=3)
        pca.fit(points_centered)
        
        # 主軸（特徵向量）
        axes = pca.components_  # 3x3 矩陣
        
        # 將點投影到主軸坐標系
        points_rotated = points_centered @ axes.T
        
        # 計算在主軸坐標系中的邊界
        min_bound = np.min(points_rotated, axis=0)
        max_bound = np.max(points_rotated, axis=0)
        
        # 3D bounding box 的尺寸（長寬高）
        size = max_bound - min_bound
        
        # 計算方向角度（yaw, pitch, roll）
        # 主要方向向量
        main_axis = axes[0]  # 第一主成分
        
        # Yaw 角度（繞 Z 軸，XY 平面）
        yaw = np.arctan2(main_axis[1], main_axis[0]) * 180 / np.pi
        
        # Pitch 角度（繞 Y 軸）
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
        
        bbox_3d = {
            'center': centroid,
            'size': size,
            'orientation': {
                'yaw': yaw,
                'pitch': pitch
            },
            'rotation_matrix': axes,
            'corners': corners_world,
            'volume': np.prod(size)
        }
        
        return bbox_3d
    
    def project_3d_to_2d(self, point_3d, intrinsics):
        """將3D點投影回2D圖像座標"""
        X, Y, Z = point_3d
        if Z == 0:
            return None
        
        u = int(X * intrinsics.fx / Z + intrinsics.ppx)
        v = int(Y * intrinsics.fy / Z + intrinsics.ppy)
        
        return (u, v)
    
    #     return image
    def visualize_3d_bbox(self, image, bbox_3d, intrinsics, color=(0, 255, 0)):
        """繪製 3D bounding box（不繪製中心點）"""
        corners = bbox_3d['corners']
        
        # 投影所有角點
        points_2d = []
        for corner in corners:
            pt_2d = self.project_3d_to_2d(corner, intrinsics)
            if pt_2d:
                points_2d.append(pt_2d)
        
        if len(points_2d) < 8:
            return image
        
        # 繪製 12 條邊（只繪製框架）
        edges = [
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5),
            (4, 6), (5, 7), (6, 7)
        ]
        
        for edge in edges:
            pt1 = points_2d[edge[0]]
            pt2 = points_2d[edge[1]]
            cv2.line(image, pt1, pt2, color, 2)
     
        return image


    
    def compute_iou(self, box1, box2):
        """計算兩個 bounding box 的 IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # 計算交集區域
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
           
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0 # 無交集 
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        
        # 計算聯集區域
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def non_maximum_suppression(self, detections, iou_threshold=0.5):
        """移除重疊的檢測"""
        if len(detections) == 0: # no bounding boxes
            return []
        
        # 按信心度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while len(detections) > 0:
            current = detections[0]
            keep.append(current)
            detections = detections[1:] # 移除當前框
            
            # 移除與當前檢測重疊度高的其他檢測
            filtered = []
            for det in detections:
                iou = self.compute_iou(current['bbox_2d'], det['bbox_2d'])
                if iou < iou_threshold:
                    filtered.append(det)
            detections = filtered
        
        return keep
    
    
    def camera_to_base_transform(self, point_camera):
        """
        將相機坐標系的點轉換到機器人基座坐標系
        變換規則（座標軸映射）：
        X_base = Y_camera
        Y_base = X_camera
        Z_base = -Z_camera
        參數:
            point_camera: (X, Y, Z) 相機坐標系中的點（公尺）
        
        返回:
            (X_base, Y_base, Z_base) 機器人基座坐標系中的點（公尺）
        """
        neck_height = 0.1  # base to 頸部高度（公尺）
        neck_length = 0.05  # 頸部長度（公尺）
        X_camera, Y_camera, Z_camera = point_camera
    
        # 直接映射
        # X_base = Y_camera
        # Y_base = X_camera
        # Z_base = -Z_camera


        # head
        X_base = -Y_camera + neck_length
        Y_base = -X_camera
        Z_base = -Z_camera + neck_height

        # 手
        X_base = -Y_camera 
        Y_base = -X_camera
        Z_base = -Z_camera 


        return (X_base, Y_base, Z_base)

    def yaw_camera_to_base(self, yaw_camera_deg):
        """
        將相機坐標系的 Yaw 角度轉換到機器人基座坐標系
        
        參數:
            yaw_camera_deg: 相機坐標系中的 Yaw 角度（度）
        
        返回:
            yaw_base_deg: 機器人基座坐標系中的 Yaw 角度（度）
        """
        # 座標系旋轉導致 Yaw 角度的變換
        # 繞 Z 軸旋轉 -90° 會使 Yaw 角度減少 90°
        yaw_base_deg = yaw_camera_deg - 90.0
        
        # 標準化到 [-180, 180) 範圍
        while yaw_base_deg < -180:
            yaw_base_deg += 360
        while yaw_base_deg >= 180:
            yaw_base_deg -= 360
        
        return yaw_base_deg

    def capture_and_display(self):
        """持續拍攝並顯示相機畫面"""
        try:
            while self.running:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # 更新最新的影像資料
                self.latest_depth_frame = depth_frame
                self.depth_image = np.asanyarray(depth_frame.get_data())
                self.latest_color_image = np.asanyarray(color_frame.get_data())
                # 儲存當前畫面
                self.save_current_frame()
                result = self.detect_objects()
                
                if result:
                    print("偵測完成")
                    self.detection_count = 0
                    break
                else:
                    self.detection_count += 1
                    print("未偵測到物品")
                    if self.detection_count >= self.detected_threshold:
                        print(f"連續{self.detected_threshold}次未偵測到物品，結束偵測")
                        break
                        
        except KeyboardInterrupt:
            print("中斷程式，釋放相機資源…")
        finally:
            self.running = False
            self.pipeline.stop()
            cv2.destroyAllWindows()
    
    def save_current_frame(self):
        """儲存當前畫面"""
        if self.latest_color_image is not None:
            self.latest_color_image = cv2.cvtColor(self.latest_color_image, cv2.COLOR_BGR2RGB)
            cv2.imwrite("/home/gairobots/camera/GroundingDINO/data/camera_image.png", 
                       self.latest_color_image)
            print("已儲存畫面: camera_image.png")
            np.save("/home/gairobots/camera/GroundingDINO/data/depth_image.npy", self.depth_image)
            print("已儲存深度資料: depth_image.npy")
    



    def find_handle_by_sliding_window(self, object_crop, step_size=20, window_sizes=None):
        """用滑動窗口在物體內搜索握柄位置
        返回最相似的握柄區域座標
        """
        if self.handle_reference_features is None:
            print("❌ 握柄特徵未載入")
            return None
        
        if window_sizes is None:
            window_sizes = [  (80,80), (100, 100), (120, 120), (180,180) ] # 
        
        h, w = object_crop.shape[:2]
        best_similarity = 0.0
        best_bbox = None
        best_size = None
        
        print(f"    🔍 滑動窗口搜索握柄 (物體大小: {w}x{h})...")
        
        for win_h, win_w in window_sizes:
            if win_h > h or win_w > w:
                continue
            
            # 滑動窗口
            for y in range(0, h - win_h, step_size):
                for x in range(0, w - win_w, step_size):
                    # 提取窗口區域
                    window_crop = object_crop[y:y+win_h, x:x+win_w].copy()
                    
                    # 計算相似度
                    similarity = self.compute_handle_similarity(window_crop)
                    
                    # 更新最佳結果
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_bbox = (x, y, x + win_w, y + win_h)
                        best_size = (win_w, win_h)
            
            print(f"    窗口大小 {win_w}x{win_h}: 最佳相似度 = {best_similarity:.3f}")
        
        if best_similarity > 0.35 and best_bbox is not None:
            print(f"    ✓ 找到握柄 (相似度: {best_similarity:.3f}, 大小: {best_size})")
            return {
                'bbox': best_bbox,
                'similarity': best_similarity,
                'size': best_size
            }
        else:
            print(f"    ⚠️  未找到握柄 (最佳相似度: {best_similarity:.3f})")
            return None



    def detect_objects(self):
        """對當前畫面進行物品偵測並計算3D資訊（CLIP 握柄比對版本）"""
        if self.latest_color_image is None:
            print("沒有可用的影像進行偵測")
            return False
            
        print(f"開始物品偵測（最多 {self.max_objects} 個物品）...")
        
        # 儲存當前畫面用於偵測
        temp_image_path = "/home/gairobots/camera/GroundingDINO/data/temp_detect.png"
        cv2.imwrite(temp_image_path, self.latest_color_image)
        
        # 載入影像進行偵測
        image_source, image = load_image(temp_image_path)
        
        # 設定 SAM
        rgb_image = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb_image)
        
        # GroundingDINO 偵測物體
        boxes, logits, phrases = predict(self.gd_model, image, self.caption, 0.20, 0.10, self.device)
        
        print(f"GroundingDINO 偵測到 {len(boxes)} 個候選框")
        
        # ===== 收集所有候選檢測，並按信心度排序 =====
        all_detections = []
        H, W = image_source.shape[:2]
        
        for i, (box, logit, phrase) in enumerate(zip(boxes, logits, phrases)):
            if logit < 0.25:  # 過濾低信心度
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
        
        if len(all_detections) == 0:
            print("未偵測到任何物品")
            return False
        
        # 執行 NMS（非極大值抑制）移除重疊檢測
        all_detections = self.non_maximum_suppression(all_detections, iou_threshold=0.5)
        
        # 選擇前 k 個信心度最高的檢測
        selected_detections = all_detections[:self.max_objects]
        
        print(f"經過 NMS 和篩選後，剩餘 {len(selected_detections)} 個檢測")
        
        # 視覺化結果
        vis = image_source.copy()
        random.seed(42)
        any_detected = False
        
        # 為每個檢測生成不同顏色
        colors_list = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0),
            (0, 128, 255), (128, 255, 0)
        ]
        
        for det_idx, detection in enumerate(selected_detections):
            box = detection['box']
            x1, y1, x2, y2 = detection['bbox_2d']
            logit = detection['logit']
            phrase = detection['phrase']
            
            color = colors_list[det_idx % len(colors_list)]
            
            print(f"\n処理第 {det_idx + 1} 個物品: {phrase}，信心度: {logit:.2f}")
            
            # ===== 第一次檢測：使用 SAM 生成整體物體的 mask =====
            print(f"  第一次檢測：整體物體 3D bbox...")
            crop_image = image_source[y1:y2, x1:x2].copy()
            masks_obj, _, _ = self.predictor.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False
            )
            object_mask = masks_obj[0].astype(np.uint8) * 255
            
            # 保存整體 object mask
            cv2.imwrite(f"/home/gairobots/camera/GroundingDINO/output/mask_object_{det_idx}.jpg", object_mask)
            print(f"  已保存整體 mask: mask_object_{det_idx}.jpg")

            # 計算整體物體的 3D bbox
            points_3d_object = self.depth_to_point_cloud(self.depth_image, object_mask, self.intrinsics)
            
            if len(points_3d_object) < 10:
                print(f"    ❌ 物體點雲不足，跳過")
                continue
            
            bbox_3d_object = self.compute_3d_bounding_box(points_3d_object)
            
            if bbox_3d_object is None:
                print(f"    ❌ 無法計算整體物體 3D bbox，跳過")
                continue
            
            print(f"    ✓ 整體物體 3D bbox 計算成功 ({len(points_3d_object)} 個點)")
                    # ===== 第二次檢測：用滑動窗口 CLIP 搜索握柄 =====
            print(f"  第二次檢測：CLIP 滑動窗口搜索握柄...")

            # 在整個物體內搜索握柄
            # handle_result = self.find_handle_by_sliding_window(crop_image, step_size=10, window_sizes=[(50, 80)])
            handle_result = self.find_handle_by_sliding_window(crop_image, step_size=10)
            # handle_result = self.find_handle_by_sliding_window(crop_image, step_size=10, window_sizes=(180,180))
            use_handle = False

            if handle_result is not None:
                # 提取握柄區域信息
                handle_x1_crop, handle_y1_crop, handle_x2_crop, handle_y2_crop = handle_result['bbox']
                similarity_score = handle_result['similarity']
                
                # 映射到原始圖像座標
                handle_x1 = x1 + handle_x1_crop
                handle_y1 = y1 + handle_y1_crop
                handle_x2 = x1 + handle_x2_crop
                handle_y2 = y1 + handle_y2_crop
                
                print(f"    ✓ 找到握柄區域: ({handle_x1}, {handle_y1}) -> ({handle_x2}, {handle_y2})")
                print(f"    握柄 CLIP 相似度: {similarity_score:.3f}")
                
                # 用 SAM 生成 mask
                print(f"    用 SAM 生成握柄遮罩...")
                masks_handle, _, _ = self.predictor.predict(
                    box=np.array([handle_x1, handle_y1, handle_x2, handle_y2]),
                    multimask_output=False
                )
                handle_mask = masks_handle[0].astype(np.uint8) * 255
                
                # 保存握柄 mask
                cv2.imwrite(f"/home/gairobots/camera/GroundingDINO/output/mask_handle_final_{det_idx}.jpg", handle_mask)
                print(f"  已保存握柄最終 mask: mask_handle_final_{det_idx}.jpg")
                
                # 計算握柄 3D bbox
                print(f"    計算握柄 3D bounding box...")
                
                handle_points_3d = self.depth_to_point_cloud(self.depth_image, handle_mask, self.intrinsics)
                
                if len(handle_points_3d) >= 10:
                    handle_bbox_3d = self.compute_3d_bounding_box(handle_points_3d)
                    
                    if handle_bbox_3d is not None:
                        print(f"    ✓ 握柄 3D bbox 計算成功 ({len(handle_points_3d)} 個點)")
                        use_handle = True
                        
                        bbox_3d = handle_bbox_3d
                        final_mask = handle_mask
                        final_bbox_2d = (handle_x1, handle_y1, handle_x2, handle_y2)
                        points_3d = handle_points_3d
                        region_type = "handle"
                    else:
                        print(f"    ⚠️  無法計算握柄 3D bbox，改用整個物體")
                else:
                    print(f"    ⚠️  握柄點雲不足 ({len(handle_points_3d)})，改用整個物體")
            else:
                print(f"    ⚠️  CLIP 搜索未找到握柄，改用整個物體")

            # ===== 如果握柄檢測失敗，使用整個物體 =====
            if not use_handle:
                print(f"    使用整個物體 3D bounding box")
                bbox_3d = bbox_3d_object
                final_mask = object_mask
                final_bbox_2d = (x1, y1, x2, y2)
                points_3d = points_3d_object
                region_type = "object"


            # 👇 ===== 所有資訊都基於最終確定的 bbox_3d =====
            # ===== 提取 3D 資訊 =====
            print(f"  提取 3D 資訊（{region_type}）...")
            
            corners = bbox_3d['corners']
            center_3d = np.mean(corners, axis=0)
            size_3d = bbox_3d['size']
            yaw = bbox_3d['orientation']['yaw']
            pitch = bbox_3d['orientation']['pitch']
            center_base = self.camera_to_base_transform(center_3d)
            
            print(f"  3D中心: ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f}) m")
            print(f"  3D尺寸: ({size_3d[0]:.3f}, {size_3d[1]:.3f}, {size_3d[2]:.3f}) m")
            print(f"  Yaw角度: {yaw:.1f}°, Pitch角度: {pitch:.1f}°")
            
            # 繪製 3D bounding box
            vis = self.visualize_3d_bbox(vis, bbox_3d, self.intrinsics, color)
            
            # 繪製中心點
            print(f"  投影中心點...")
            center_2d = self.project_3d_to_2d(center_3d, self.intrinsics)
            
            if center_2d is not None:
                cx, cy = center_2d
                print(f"  中心點 2D: ({cx}, {cy})")
                
                cv2.circle(vis, (int(cx), int(cy)), 10, color, -1)
                cv2.circle(vis, (int(cx), int(cy)), 10, (255, 255, 255), 2)
                cv2.putText(vis, "C", (int(cx) - 20, int(cy) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                print(f"  ⚠️  無法投影中心點")
            
            any_detected = True
            
            # ===== 計算長邊兩端點 =====
            print(f"  計算長邊兩端點...")
            
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
            print(f"  左端點 3D: ({left_endpoint_3d[0]:.3f}, {left_endpoint_3d[1]:.3f}, {left_endpoint_3d[2]:.3f}) m")
            print(f"  右端點 3D: ({right_endpoint_3d[0]:.3f}, {right_endpoint_3d[1]:.3f}, {right_endpoint_3d[2]:.3f}) m")
            left_endpoint_base = self.camera_to_base_transform(left_endpoint_3d)
            right_endpoint_base = self.camera_to_base_transform(right_endpoint_3d)
            
            max_distance = np.linalg.norm(right_endpoint_3d - left_endpoint_3d)
            edge_direction = right_endpoint_3d - left_endpoint_3d
            edge_yaw = np.arctan2(edge_direction[1], edge_direction[0]) * 180 / np.pi
            
            print(f"  左端點基座: ({left_endpoint_base[0]:.3f}, {left_endpoint_base[1]:.3f}, {left_endpoint_base[2]:.3f}) m")
            print(f"  右端點基座: ({right_endpoint_base[0]:.3f}, {right_endpoint_base[1]:.3f}, {right_endpoint_base[2]:.3f}) m")
            print(f"  長邊長度: {max_distance:.3f} m")
            print(f"  長邊 Yaw: {edge_yaw:.1f}°")
            
            # 投影到 2D
            left_endpoint_2d = self.project_3d_to_2d(left_endpoint_3d, self.intrinsics)
            right_endpoint_2d = self.project_3d_to_2d(right_endpoint_3d, self.intrinsics)
            print(f"  左端點 2D: {left_endpoint_2d if left_endpoint_2d else '無法投影'}")
            print(f"  右端點 2D: {right_endpoint_2d if right_endpoint_2d else '無法投影'}")
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
            
            # ===== CLIP 驗證 =====
            print(f"  CLIP 驗證...")
            
            crop = image_source[final_bbox_2d[1]:final_bbox_2d[3], final_bbox_2d[0]:final_bbox_2d[2]].copy()
            crop_mask_2d = final_mask[final_bbox_2d[1]:final_bbox_2d[3], final_bbox_2d[0]:final_bbox_2d[2]]
            crop = cv2.bitwise_and(crop, crop, mask=crop_mask_2d)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            inputs = self.clip_processor(images=crop_pil, text=self.candidate_phrases, 
                                        return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                img_feat = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
                txt_feat = self.clip_model.get_text_features(
                    input_ids=inputs["input_ids"], 
                    attention_mask=inputs["attention_mask"]
                )
                sims = (img_feat @ txt_feat.T).squeeze(0)
                clip_label = self.candidate_phrases[sims.argmax().item()]
            
            final_label = clip_label
            print(f"  CLIP 驗證: {clip_label}")
            
            # ===== 儲存物品資訊 =====
            center_base_mm = [center_base[0] * 1000, center_base[1] * 1000, center_base[2] * 1000 ] # 轉換為毫米
            left_endpoint_base_mm = [left_endpoint_base[0] * 1000, left_endpoint_base[1] * 1000, left_endpoint_base[2] * 1000 ]  # 轉換為毫米
            right_endpoint_base_mm = [right_endpoint_base[0] * 1000, right_endpoint_base[1] * 1000, right_endpoint_base[2] * 1000 ]  # 轉換為毫米
            obj_info = {
                "name": final_label,
                "region_type": region_type,
                "3d_center_base": tuple(center_base_mm),  # 轉換為毫米
                "3d_size": tuple(size_3d),
                "orientation": {
                    "yaw": float(yaw),
                    "pitch": float(pitch),
                    "edge_yaw": float(edge_yaw)
                },
                "left_endpoint_base": tuple(left_endpoint_base_mm), # 轉換為毫米
                "right_endpoint_base": tuple(right_endpoint_base_mm), # 轉換為毫米
            }
            
            self.objects_info.append(obj_info)
            
            # ===== 繪製文字資訊 =====
            text = (f"#{det_idx+1} {final_label}: {logit:.2f}\n"
                f"Region: {region_type}\n"
                f"Center: ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})m\n"
                f"Size: ({size_3d[0]:.3f}, {size_3d[1]:.3f}, {size_3d[2]:.3f})m\n"
                f"Yaw: {yaw:.1f}° / Edge: {edge_yaw:.1f}°\n"
                f"Pitch: {pitch:.1f}°\n"
                f"Length: {max_distance:.3f}m\n"
                f"Points: {len(points_3d)}")
            
            font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            lines = text.split('\n')
            line_height = cv2.getTextSize("Test", font, fs, th)[0][1] + 5
            
            tx = final_bbox_2d[0]
            ty = final_bbox_2d[1] - 10
            
            max_width = max([cv2.getTextSize(line, font, fs, th)[0][0] for line in lines])
            total_height = len(lines) * line_height
            
            tx = max(0, min(tx, W - max_width - 10))
            ty = max(total_height, min(ty, H - 10))
            
            cv2.rectangle(vis, (tx, ty - total_height), (tx + max_width, ty + 5), 
                        color, cv2.FILLED)
            
            for i, line in enumerate(lines):
                y_pos = ty - total_height + (i + 1) * line_height
                cv2.putText(vis, line, (tx, y_pos), font, fs, (0, 0, 0), th, cv2.LINE_AA)
        
        # 儲存結果
        cv2.imwrite("/home/gairobots/camera/GroundingDINO/output/detection_3d.jpg", vis)
        
        if not any_detected:
            print("未偵測到任何物品")
            return False
        else:
            print(f"\n✅ 3D偵測結果已儲存: detection_3d.jpg")
            print(f"成功檢測到 {len(self.objects_info)} 個物品")
            return True




    def get_objects_info(self):
        """取得偵測到的物品資訊"""
        return self.objects_info
    def capture_handle_reference(self, tool_name="name"):
        """收集握柄示範圖像並保存特徵
        使用方式：在初始化後調用一次，或按需重新收集
        """
        if self.latest_color_image is None:
            print("❌ 沒有可用的攝像頭影像")
            return False
        
        # 保存原始握柄圖像
        handle_ref_path = f"/home/gairobots/camera/GroundingDINO/data/handle_features/{tool_name}.jpg"
        cv2.imwrite(handle_ref_path, self.latest_color_image)
        print(f"✓ 握柄示範圖像已保存: {handle_ref_path}")
        
        # 載入並提取特徵
        self.load_handle_reference_features(tool_name=tool_name)
        return True

    def load_handle_reference_features(self, tool_name="name"):
        """載入握柄示範圖像並提取 CLIP 特徵"""
        handle_ref_path = f"/home/gairobots/camera/GroundingDINO/data/handle_features/{tool_name}.jpg"
        
        if not os.path.exists(handle_ref_path):
            print(f"❌ 握柄示範圖像不存在: {handle_ref_path}")
            self.handle_reference_features = None
            return
        
        print(f"📸 載入握柄示範圖像...")
        
        # 讀取圖像
        handle_img = cv2.imread(handle_ref_path)
        if handle_img is None:
            print(f"❌ 無法讀取握柄示範圖像")
            return
        
        handle_img_rgb = cv2.cvtColor(handle_img, cv2.COLOR_BGR2RGB)
        handle_img_pil = Image.fromarray(handle_img_rgb)
        
        # 用 CLIP 提取特徵
        inputs = self.clip_processor(images=handle_img_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            self.handle_reference_features = self.clip_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            # 正規化特徵
            self.handle_reference_features = self.handle_reference_features / (
                self.handle_reference_features.norm(dim=-1, keepdim=True) + 1e-5
            )
        
        print(f"✓ 握柄特徵已提取，維度: {self.handle_reference_features.shape}")

    def compute_handle_similarity(self, crop_img):
        """計算握柄相似度 (0 ~ 1)"""
        if self.handle_reference_features is None:
            return 0.0
        
        # 轉換為 PIL 圖像
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        # 用 CLIP 提取特徵
        inputs = self.clip_processor(images=crop_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            crop_features = self.clip_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            # 正規化特徵
            crop_features = crop_features / (
                crop_features.norm(dim=-1, keepdim=True) + 1e-5
            )
        
        # 計算余弦相似度 (0 ~ 1)
        similarity = torch.nn.functional.cosine_similarity(
            self.handle_reference_features,
            crop_features,
            dim=-1
        ).item()
        
        return similarity
    def collect_handle_reference_interactive(self,tool_name="name"):
        """互動式收集握柄示範圖像
        - 即時顯示攝像頭畫面
        - 按 ENTER 拍照並保存
        - 按 ESC 取消
        - 收集結束直接返回，不執行檢測
        """
        print("\n" + "="*50)
        print("📸 握柄示範圖像收集模式")
        print("="*50)
        print("✨ 調整角度和位置，使握柄清晰可見")
        print("⌨️  按 ENTER 拍照")
        print("🚪 按 ESC 取消")
        print("="*50 + "\n")
        
        if not self.pipeline:
            print("❌ 攝像頭未初始化，請先初始化")
            return False
        
        try:
            while True:
                # 獲取當前幀
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    print("⚠️  無法獲取攝像頭信號")
                    continue
                
                # 轉換為 OpenCV 格式
                color_image = np.asanyarray(color_frame.get_data())
                display_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # 添加提示文字
                cv2.putText(display_image, "Press ENTER to capture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_image, "Press ESC to cancel", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 在中心添加十字準線
                h, w = display_image.shape[:2]
                cv2.line(display_image, (w//2 - 50, h//2), (w//2 + 50, h//2), (255, 255, 0), 2)
                cv2.line(display_image, (w//2, h//2 - 50), (w//2, h//2 + 50), (255, 255, 0), 2)
                
                # 顯示實時畫面
                cv2.imshow("Handle Reference Capture", display_image)
                
                # 等待按鍵輸入
                key = cv2.waitKey(1) & 0xFF
                
                if key == 13:  # ENTER 鍵
                    print("\n✓ 拍照中...")
                    
                    # 保存握柄示範圖像
                    handle_ref_path = f"/home/gairobots/camera/GroundingDINO/data/handle_features/{tool_name}.jpg"
                    cv2.imwrite(handle_ref_path, display_image)
                    print(f"✓ 握柄示範圖像已保存: {handle_ref_path}")
                    
                    # 提取特徵
                    print("🧠 正在提取握柄特徵...")
                    self.load_handle_reference_features(tool_name=tool_name)
                    
                    # 保存特徵到文件
                    if self.handle_reference_features is not None:
                        self.save_handle_features(tool_name=tool_name)
                        print("✅ 握柄特徵已提取並保存！")
                        print(f"📊 特徵維度: {self.handle_reference_features.shape}")
                        cv2.destroyAllWindows()
                        return True
                    else:
                        print("❌ 特徵提取失敗，請重試")
                        continue
                
                elif key == 27:  # ESC 鍵
                    print("\n❌ 已取消握柄收集\n")
                    cv2.destroyAllWindows()
                    return False
        
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            cv2.destroyAllWindows()
            return False
    def save_handle_features(self,tool_name="name"):
        """將握柄特徵保存到文件"""
        if self.handle_reference_features is None:
            print("⚠️  沒有握柄特徵需要保存")
            return False

        features_path = f"/home/gairobots/camera/GroundingDINO/data/handle_features/{tool_name}_handle_features.pt"

        try:
            torch.save(self.handle_reference_features, features_path)
            print(f"✓ 握柄特徵已保存: {features_path}")
            return True
        except Exception as e:
            print(f"❌ 保存特徵失敗: {e}")
            return False

    def load_handle_features_from_file(self):
        """從文件載入握柄特徵"""

        # features_path = "/home/aiRobots/Software/GroundingDINO/data/brush_handle_features.pt"
        features_path = "/home/gairobots/camera/GroundingDINO/data/handle_features.pt"
        if not os.path.exists(features_path):
            print(f"⚠️  握柄特徵文件不存在: {features_path}")
            print("   請先執行 collect_handle_reference_interactive() 收集握柄")
            return False
        
        try:
            self.handle_reference_features = torch.load(features_path, map_location=self.device)
            print(f"✓ 握柄特徵已從文件載入: {features_path}")
            print(f"📊 特徵維度: {self.handle_reference_features.shape}")
            return True
        except Exception as e:
            print(f"❌ 載入特徵失敗: {e}")
            return False
def collect_demo_images(realsense_number="923322070636", num_demos=3, tool_name="brush"):

    detector = CameraDetector(realsense_number=realsense_number, max_objects=1, candidate_phrases=[tool_name])


    print("="*60)
    print("🎯 握柄特徵收集工具")
    print("="*60)

    # 收集握柄
    success = detector.collect_handle_reference_interactive(tool_name=tool_name)

    if success:
        print("\n✅ 握柄特徵收集成功！")
        print("💾 特徵已保存，下次執行偵測時會自動載入")
    else:
        print("\n❌ 握柄特徵收集失敗")

    print("="*60)

def object_detection(realsense_number="923322070636", max_objects=2, obj_prompt=[
           "tool", "blue vacuum cleaner", "broom", "dustpan", "brush tool",
        ]):
    """
    執行物品檢測
    
    參數:
        max_objects: 最多檢測的物品數量（預設為2）
    """
    detector = CameraDetector(realsense_number=realsense_number, max_objects=max_objects, candidate_phrases=obj_prompt)
    print("="*60)
    print("🎯 物體偵測系統")
    print("="*60)

    # 檢查握柄特徵是否已載入
    if detector.handle_reference_features is None:
        print("❌ 握柄特徵未載入！")
        print("   請先執行: python collect_handle.py")
        exit(1)

    print("✅ 握柄特徵已載入，開始偵測...")
    print("="*60 + "\n")

    
    detector.capture_and_display()

    
    objects = detector.get_objects_info()
    
    if objects:
        print("\n========== 偵測到的3D物品資訊 ==========")
        for i, obj in enumerate(objects, 1):
            print(f"\n物品 {i}:")
            print(f"  名稱: {obj['name']}")
            # print(f"  3D中心座標: {obj['3d_center']} 公尺")
            print(f"  3D中心基座座標: {obj['3d_center_base']} mm")
            print(f"  3D尺寸 (長x寬x高): {obj['3d_size']} 公尺")
            print(f"  Yaw角度: {obj['orientation']['yaw']:.1f}°")
            print(f"  Pitch角度: {obj['orientation']['pitch']:.1f}°")
            # print(f"  體積: {obj['volume']:.4f} 立方公尺")
            # print(f"  信心度: {obj['confidence']:.2f}")
            # print(f"  點雲數量: {obj['num_points']}")
    else:
        print("未偵測到任何物品")
        return None
    
    return objects

if __name__ == "__main__":
    # 檢測最多 2 個物品
    # objects = object_detection(realsense_number="923322070636", max_objects=2, obj_prompt=[
    #        "tool", "blue vacuum cleaner", "broom", "dustpan", "brush tool",
    #     ])

    # objects = object_detection(realsense_number="923322070636", max_objects=1, 
    #                             obj_prompt=[
    #                                     # "broom", "dustpan", "brush tool","handle of dustpan"
    #                                     "dustpan tool","brush tool"
    #                                     # "thin dustpan handle","thin brush handle","slim grip of dustpan", "slim grip of brush",
    #                                     # "dustpan handle and brush handle",
    #                                     # "brush tool",
    #                                 ])


    collect_demo_images(realsense_number="243222072706", num_demos=3, tool_name="brush")
    

    # 如果要檢測 3 個物品，可以改為：
    # objects = object_detection(max_objects=3)


