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

# object_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用設備: {object_device}")
# print(f"cuDNN 啟用: {cudnn.enabled}")


class CameraDetector:
    def __init__(self, realsense_number="923322070636", max_objects=2, candidate_phrases=[
           "tool", "blue vacuum cleaner", "broom", "dustpan", "brush tool",
        ]):  # 新增參數：最大檢測物品數量
        # 初始化模型

 
        

 
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


      
    def init_camera(self):
        """初始化 RealSense 相機"""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.realsense_number)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        self.profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        
        # 獲取相機內參
        self.intrinsics = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile().get_intrinsics()
        
        print(f"  相機內參: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
        
        # 跳過前 40 幀
        for i in range(40):
            self.pipeline.wait_for_frames()

    def extract_dino_keypoints(self, image_crop, mask_crop, bbox_2d):
     
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




    def collect_handle_reference_interactive(self,name="name"):
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
        num = 0
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
                display_image = color_image.copy()
                # display_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
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
                    handle_ref_path = f"/home/gairobots/camera/GroundingDINO/traing_data/{name}/{num}.jpg"
                    cv2.imwrite(handle_ref_path, color_image)
                    print(f"✓ 握柄示範圖像已保存: {handle_ref_path}")
                    num += 1
                    
                
                elif key == 27:  # ESC 鍵
                    print("\n❌ 已取消握柄收集\n")
                    cv2.destroyAllWindows()
                    return False
        
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            cv2.destroyAllWindows()
            return False
   

def collect_demo_images(realsense_number="923322070636", num_demos=3, name="pick"):

    detector = CameraDetector(realsense_number=realsense_number, max_objects=1, candidate_phrases=[name])


    print("="*60)
    print("🎯 握柄特徵收集工具")
    print("="*60)

    # 收集握柄
    success = detector.collect_handle_reference_interactive(name=name)
    print("="*60)

if __name__ == "__main__":

    collect_demo_images(realsense_number="923322070636", num_demos=3, name="sweep")



