"""
全局共享模型模組 - 所有 CameraDetector 實例共用
模型只載入一次，節省記憶體
"""
import torch
import os
from groundingdino.util.inference import load_model
from segment_anything import SamPredictor, sam_model_registry
from transformers import CLIPProcessor, CLIPModel
import gc
from pathlib import Path
class SharedModels:
    """全局模型單例 - 確保模型只載入一次"""
    _instance = None
    _lock = None
    
    def __new__(cls):
        if cls._instance is None:
            import threading
            cls._lock = threading.Lock()
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化模型（只執行一次）"""
        if self._initialized:
            return
        
        print("\n" + "="*60)
        print("🔄 初始化全局模型（一次性）")
        print("="*60)
        
        # self.device = torch.device("cpu")
        # os.environ["FORCE_CPU"] = "1"
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用設備: {self.device}")
        # print(f"cuDNN 啟用: {cudnn.enabled}")
        # 1️⃣ GroundingDINO
        print("\n[1/4] 載入 GroundingDINO...")
        CONFIG_PATH = "/home/gairobots/camera/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        WEIGHTS_PATH = "/home/gairobots/camera/GroundingDINO/weights/groundingdino_swint_ogc.pth"
        self.gd_model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=self.device)
        print("      ✓ GroundingDINO 載入完成")
        
        # 2️⃣ SAM
        print("[2/4] 載入 SAM...")
        sam = sam_model_registry["default"](
            checkpoint="/home/gairobots/camera/GroundingDINO/sam_checkpoints/sam_vit_h_4b8939.pth"
        )
        sam.to(self.device)
        self.predictor = SamPredictor(sam)
        print("      ✓ SAM 載入完成")
        
        # 3️⃣ CLIP
        print("[3/4] 載入 CLIP...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("      ✓ CLIP 載入完成")
        
        # 握柄特徵
        print("[4/4] 载入握柄特征...")
        self.handle_features = {}  # 字典存储
        self.load_handle_features()

        # print("[4/4] 載入握柄特徵...")
        # self.handle_reference_features = None
        # features_path = "/home/gairobots/camera/GroundingDINO/data/handle_features.pt"
        # if os.path.exists(features_path):
        #     try:
        #         self.handle_reference_features = torch.load(features_path, map_location=self.device)
        #         print(f"      ✓ 握柄特徵已載入 (維度: {self.handle_reference_features.shape})")
        #     except Exception as e:
        #         print(f"      ⚠️  握柄特徵載入失敗: {e}")
        # else:
        #     print(f"      ⚠️  握柄特徵檔案不存在: {features_path}")
        
        self._initialized = True
        print("\n" + "="*60)
        print("✅ 全局模型初始化完成！所有相機將共享這些模型")
        print("="*60 + "\n")

    def load_handle_features(self):
        """载入所有工具的握柄特征"""
        features_dir = Path("/home/gairobots/camera/GroundingDINO/data/handle_features")
        tool_categories = {
            'brush tool': 'brush_handle_features.pt',
            'dustpan tool': 'dustpan_handle_features.pt',
           
        }
        loaded_count = 0
        for tool_name, filename in tool_categories.items():
            feature_path = features_dir / filename
            
            if feature_path.exists():
                try:
                    features = torch.load(feature_path, map_location=self.device)
                    self.handle_features[tool_name] = features
                    loaded_count += 1
                    print(f"      ✓ {tool_name} 特征已载入 (维度: {features.shape})")
                except Exception as e:
                    print(f"      ⚠️  {tool_name} 载入失败: {e}")
            else:
                print(f"      ⚠️  找不到 {tool_name} 特征: {feature_path}")
        
        print(f"      总计载入 {loaded_count}/{len(tool_categories)} 个工具特征")
    def get_handle_features(self, tool_class):
        """获取特定工具的把手特征"""
        # 优先返回特定工具特征
        if tool_class in self.handle_features:
            return self.handle_features[tool_class]
        print(f"      ⚠️  无可用特征 for {tool_class}")
        return None    
    def release(self):
        """釋放所有模型資源並清空 GPU/CPU 記憶體"""
        if not self._initialized:
            print("⚠️  模型尚未初始化，無需釋放")
            return
        
        print("\n" + "="*60)
        print("🗑️  開始釋放模型資源...")
        print("="*60)
        
        # 記錄釋放前的記憶體使用
        if torch.cuda.is_available():
            gpu_mem_before = torch.cuda.memory_allocated() / 1024**3  # GB
            print(f"釋放前 GPU 記憶體: {gpu_mem_before:.2f} GB")
        
        # 1️⃣ 釋放 GroundingDINO
        print("\n[1/4] 釋放 GroundingDINO...")
        if hasattr(self, 'gd_model') and self.gd_model is not None:
            del self.gd_model
            self.gd_model = None
            print("      ✓ GroundingDINO 已釋放")
        
        # 2️⃣ 釋放 SAM
        print("[2/4] 釋放 SAM...")
        if hasattr(self, 'predictor') and self.predictor is not None:
            del self.predictor
            self.predictor = None
            print("      ✓ SAM 已釋放")
        
        # 3️⃣ 釋放 CLIP
        print("[3/4] 釋放 CLIP...")
        if hasattr(self, 'clip_model') and self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
        if hasattr(self, 'clip_processor') and self.clip_processor is not None:
            del self.clip_processor
            self.clip_processor = None
        print("      ✓ CLIP 已釋放")
        
        # 4️⃣ 釋放握柄特徵
        print("[4/4] 釋放握柄特徵...")
        if hasattr(self, 'handle_reference_features') and self.handle_reference_features is not None:
            del self.handle_reference_features
        if hasattr(self, 'handle_features') and self.handle_features is not None:
            del self.handle_features
            self.handle_reference_features = None
            print("      ✓ 握柄特徵已釋放")
        
        # 強制垃圾回收
        print("\n🔄 執行垃圾回收...")
        gc.collect()
        
        # 清空 CUDA 快取
        if torch.cuda.is_available():
            print("🔄 清空 CUDA 快取...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # 確保所有 CUDA 操作完成
            
            gpu_mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_cache_after = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"釋放後 GPU 記憶體: {gpu_mem_after:.2f} GB")
            print(f"釋放的記憶體: {gpu_mem_before - gpu_mem_after:.2f} GB")
            print(f"快取記憶體: {gpu_cache_after:.2f} GB")
        
        self._initialized = False
        print("\n" + "="*60)
        print("✅ 資源釋放完成！")
        print("="*60 + "\n")
    
    @classmethod
    def reset_singleton(cls):
        """重置單例實例（用於完全重新初始化）"""
        if cls._instance is not None:
            print("\n🔄 重置單例實例...")
            if hasattr(cls._instance, 'release'):
                cls._instance.release()
            cls._instance = None
            print("✓ 單例已重置\n")
    
    def get_memory_usage(self):
        """獲取當前記憶體使用情況"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return {
                "device": "GPU",
                "allocated_gb": f"{allocated:.2f}",
                "reserved_gb": f"{reserved:.2f}",
                "device_name": torch.cuda.get_device_name(0)
            }
        else:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                "device": "CPU",
                "rss_gb": f"{memory_info.rss / 1024**3:.2f}",
                "vms_gb": f"{memory_info.vms / 1024**3:.2f}"
            }

# 全局模型單例
shared_models = SharedModels()
