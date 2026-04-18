"""
Hey Anna 喚醒詞檢測推理模型

使用隨機森林分類器 + 官方特徵提取器 + 音頻緩衝機制

使用方法:
    detector = HeyAnnaDetector()
    score, detected = detector.predict_from_audio_file("audio.wav")
    
    或者串流使用：
    
    score, detected = detector.predict_from_array(audio_chunk, sr=16000)
    detector.reset()  # 在檢測到喚醒詞後重置
"""

import numpy as np
import librosa
import joblib
from openwakeword.utils import AudioFeatures


class HeyAnnaDetector:
    def __init__(self, model_path="hey_anna_rf_classifier.pkl", scaler_path="hey_anna_scaler.pkl", threshold=0.6):
        """
        初始化檢測器
        
        Args:
            model_path: 隨機森林模型 pickle 路徑
            scaler_path: 特徵縮放器 pickle 路徑
            threshold: 檢測閾值 (0-1)
        """
        # 加載模型
        self.rf_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.threshold = threshold
        
        # 初始化特徵提取器
        try:
            self.feature_extractor = AudioFeatures(inference_framework="onnx")
        except:
            self.feature_extractor = AudioFeatures(inference_framework="tflite")
        
        # 音頻緩衝：OpenWakeWord 需要至少 ~960ms 的音頻 (16kHz = 15360 樣本)
        self.audio_buffer = np.array([], dtype=np.int16)
        self.buffer_size = 16000 * 1  # 1 秒的音頻
        self.min_samples = 16000 * 0.6  # 至少需要 600ms
        
        print(f"✅ HeyAnnaDetector 初始化完成 (閾值: {threshold})")
    
    def reset(self):
        """重置音頻緩衝器"""
        self.audio_buffer = np.array([], dtype=np.int16)
    
    def extract_features(self, audio_array):
        """
        從音檔陣列提取特徵（使用足夠長的音頻緩衝）
        
        Args:
            audio_array: int16 類型的音訊陣列（可以是短片段）
            
        Returns:
            特徵向量 (96,) 或 None
        """
        try:
            # 確保是 int16
            if audio_array.dtype != np.int16:
                if np.max(np.abs(audio_array)) <= 1.0:
                    audio_array = (audio_array * 32767).astype(np.int16)
                else:
                    audio_array = audio_array.astype(np.int16)
            
            # 添加到緩衝區
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_array])
            
            # 只保留最新的數據
            if len(self.audio_buffer) > self.buffer_size:
                self.audio_buffer = self.audio_buffer[-self.buffer_size:]
            
            # 如果緩衝區不足，暫時返回 None
            if len(self.audio_buffer) < self.min_samples:
                return None
            
            # 提取特徵
            audio_batch = np.array([self.audio_buffer])
            features = self.feature_extractor.embed_clips(audio_batch)
            
            if len(features) > 0:
                feat = features[0]
                # 如果特徵太長，取均值
                if feat.ndim > 1:
                    feat = np.mean(feat, axis=0)
                return feat
            return None
        except Exception as e:
            # 靜默失敗，不打印錯誤（太頻繁）
            return None
    
    def predict_from_features(self, features):
        """
        從特徵向量預測（內部方法）
        
        Args:
            features: 96 維特徵向量
            
        Returns:
            (score, is_detected) 其中 score 在 [0, 1] 之間
        """
        try:
            # 縮放特徵
            feat_reshaped = np.array([features])
            scaled = self.scaler.transform(feat_reshaped)
            
            # 預測概率
            probabilities = self.rf_model.predict_proba(scaled)
            score = probabilities[0, 1]  # 類別 1 (喚醒詞) 的概率
            
            is_detected = score >= self.threshold
            return float(score), is_detected
        except Exception as e:
            return 0.0, False
    
    def predict(self, audio_array):
        """
        預測音訊陣列（主要推理方法）
        
        Args:
            audio_array: int16 音訊陣列（自動緩衝和提取）
            
        Returns:
            (score, is_detected) 或 (0.0, False) 如果緩衝區尚未充滿
        """
        try:
            # 提取特徵（會自動管理緩衝區）
            features = self.extract_features(audio_array)
            
            if features is None:
                # 緩衝區不足，返回進度分數 (0-0.3 表示正在緩衝)
                buffer_progress = len(self.audio_buffer) / self.min_samples
                return float(buffer_progress * 0.3), False
            
            # 預測
            return self.predict_from_features(features)
        except Exception as e:
            return 0.0, False
    
    def predict_from_audio_file(self, audio_path):
        """
        從完整音檔預測
        
        Args:
            audio_path: 音檔路徑
            
        Returns:
            (score, is_detected)
        """
        try:
            # 加載音檔
            audio, sr = librosa.load(audio_path, sr=16000)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # 重置緩衝區（完整文件預測，不使用流式模式）
            self.reset()
            
            # 一次性提取特徵
            features = self.extract_features(audio_int16)
            if features is None:
                return 0.0, False
            
            # 預測
            return self.predict_from_features(features)
        except Exception as e:
            return 0.0, False
    
    def predict_from_array(self, audio_array, sr=16000):
        """
        從音訊陣列預測（流式處理）
        
        Args:
            audio_array: 音訊陣列 (浮點 [-1, 1] 或 int16)
            sr: 採樣率
            
        Returns:
            (score, is_detected)
        """
        try:
            # 確保 16 kHz
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            
            # 轉換為 int16
            if audio_array.dtype != np.int16:
                if np.max(np.abs(audio_array)) <= 1.0:
                    audio_int16 = (audio_array * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_array.astype(np.int16)
            else:
                audio_int16 = audio_array
            
            # 使用 predict 方法（會管理緩衝區）
            return self.predict(audio_int16)
        except Exception as e:
            return 0.0, False
    
    def set_threshold(self, threshold):
        """調整檢測閾值"""
        self.threshold = np.clip(threshold, 0.0, 1.0)
        print(f"閾值已調整為: {self.threshold}")


# 範例使用
if __name__ == "__main__":
    # 初始化檢測器
    detector = HeyAnnaDetector(threshold=0.6)
    
    # 測試喚醒詞
    print("\n測試喚醒詞:")
    score, detected = detector.predict_from_audio_file("my_real_audio/wake_word_001.wav")
    print(f"  分數: {score:.4f}, 檢測到: {'✅ YES' if detected else '❌ NO'}")
    
    # 測試雜音
    print("\n測試雜音:")
    score, detected = detector.predict_from_audio_file("my_real_audio/noise_001.wav")
    print(f"  分數: {score:.4f}, 檢測到: {'✅ YES' if detected else '❌ NO'}")
    
    print("\n✅ 推理演示完成")
