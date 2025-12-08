#!/usr/bin/env python3
"""
多相機管理系統 - 同時管理多台相機
"""
import threading
import time
from camera_detector import CameraDetector
import shared_object
import torch
import gc

class MultiCameraSystem:
    """多相機系統管理"""
    
    def __init__(self, cameras_config):
        """
        初始化多相機系統
        
        參數:
            cameras_config: 相機配置清單
            [
                {"serial": "923322070636", "id": 0},
                {"serial": "123456789012", "id": 1},
            ]
        """
        self.cameras_config = cameras_config
        self.detectors = []
        self.threads = []
        self.results = {}

        self.object_list = {}
        shared_object.left = {}
        shared_object.right = {}
        self.detection_lock = threading.Lock()
        print("\n" + "="*60)
        print("🎯 多相機物體檢測系統")
        print("="*60)
        
        # 初始化所有相機
        for config in cameras_config:
            try:
                
                detector = CameraDetector(
                    realsense_serial=config["serial"],
                    camera_id=config["id"],
                    max_objects=config.get("max_objects", 1),
                    candidate_phrases=config.get("phrases", None)
                )
                self.detectors.append(detector)
                print(f"✅ 相機 {config['id']} 初始化成功")
            except Exception as e:
                print(f"❌ 相機 {config['id']} 初始化失敗: {e}")
    
    def run_camera_detection(self, detector_idx):
        self.clear_all_results(detector_idx)
        with self.detection_lock:
            """單個相機的偵測執行函數（在線程中執行）"""
            detector = self.detectors[detector_idx]
            
            try:
                
                # 先清理記憶體
                gc.collect()
                torch.cuda.empty_cache()

                # 獲取幀
                rgb, depth = detector.get_current_frame()
                
                if rgb is None or depth is None:
                    print(f"⚠️  相機 {detector.camera_id}: 無法獲取幀")
                    return
                print(f"GPU 記憶體使用前: {torch.cuda.memory_allocated() / 1024 / 1024:.1f}MB")
                # 執行偵測
                if detector_idx == 0:

                    success = detector.detect_objects_simple()
                else:
                    
                    success = detector.detect_objects()
                
                # 儲存結果
                self.results[detector.camera_id] = {
                    'success': success,
                    'objects': detector.get_objects_info(),
                    'timestamp': time.time()
                }
                self.info_process(detector_idx)

            except Exception as e:
                print(f"❌ 相機 {detector.camera_id} 偵測出錯: {e}")
                self.results[detector.camera_id] = {
                    'success': False,
                    'error': str(e)
                }
            finally:
                # 先清理記憶體
                detector.clear_detection_data() 
                gc.collect()
                torch.cuda.empty_cache()
    
    def update_camera_phrases(self, detector_idx, phrases):
        """更新指定相機的候選短語"""
        if 0 <= detector_idx < len(self.detectors):
            detector = self.detectors[detector_idx]
            detector.candidate_phrases = phrases
            num = len(phrases) if phrases is not None else 0
            detector.max_objects = num
            print(f"✅ 相機 {detector.camera_id} 的候選短語已更新")
        else:
            print(f"❌ 無效的相機索引: {detector_idx}")

    # def run_sequential(self):
    #     """順序執行所有相機偵測"""
    #     print("\n[順序模式] 逐台相機偵測\n")
        
    #     for i, detector in enumerate(self.detectors):
    #         print(f"\n{'='*60}")
    #         print(f"處理相機 {i+1}/{len(self.detectors)}")
    #         print(f"{'='*60}")
            
    #         # 獲取幀
    #         rgb, depth = detector.get_current_frame()
            
    #         if rgb is None or depth is None:
    #             print(f"⚠️  無法獲取幀，跳過")
    #             continue
            
    #         # 執行偵測
    #         success = detector.detect_objects()
            
    #         # 儲存結果
    #         self.results[detector.camera_id] = {
    #             'success': success,
    #             'objects': detector.get_objects_info()
    #         }
        
    #     # 打印彙總結果
    #     self.print_summary()
    
    def run_parallel(self):
        """並行執行雙手相機偵測"""
        print("\n[並行模式] 同時偵測雙手相機\n")
        
        # 第一步：所有相機同時獲取幀
        print("📷 獲取所有相機幀...\n")
        for detector in self.detectors:
            if detector.camera_id == 1 or detector.camera_id == 2:
                rgb, depth = detector.get_current_frame()
                if rgb is None:
                    print(f"⚠️  相機 {detector.camera_id}: 無法獲取幀")
        
        # 第二步：並行執行偵測
        print("🔍 並行執行偵測...\n")
        
        self.threads = []
        # for i in range(len(self.detectors)):
        for i in [1, 2]:
            thread = threading.Thread(
                target=self.run_camera_detection,
                args=(i,),
                daemon=False
            )
            self.threads.append(thread)
            thread.start()
        
        # 等待所有線程完成
        for thread in self.threads:
            thread.join()
        
        
        # 打印彙總結果
        # self.print_summary()
        self.info_process(1)
        self.info_process(2)
        torch.cuda.empty_cache()
    
    def get_all_results(self):
        """取得所有結果"""
        return self.results
    def clear_all_results(self, idx):
        detector = self.detectors[idx]
        detector.objects_info.clear()
    def info_process(self, idx):
        detector = self.detectors[idx]
        
        if idx == 1:
            shared_object.left = detector.objects_info
            # detector.objects_info.clear()
            save_obj= shared_object.left

        if idx == 2:
            shared_object.right = detector.objects_info
            
            save_obj= shared_object.right

        if idx == 0:
            shared_object.total = detector.objects_info
         
            save_obj= shared_object.total

        for obj in save_obj:
            print(f" - {obj['name']}: ")
            print(f"   3D中心: ({obj['center_pos'][0]:.1f}, {obj['center_pos'][1]:.1f}, {obj['center_pos'][2]:.1f})mm, 方向: {obj['angle']:.1f}")

            print(f"   尺寸: ({obj['3d_size'][0]:.1f}, {obj['3d_size'][1]:.1f}, {obj['3d_size'][2]:.1f} )mm, pick mode:{obj['pick_mode']}")
    
                    
        
    def cleanup(self):
        """清理所有資源"""
        print("\n🛑 清理資源...")
        for detector in self.detectors:
            detector.cleanup()
        print("✅ 所有資源已釋放\n")
