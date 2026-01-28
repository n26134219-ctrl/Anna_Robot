import pyaudio
# from faster_whisper import WhisperModel
import numpy as np
from scipy import signal  # ← 新增：用於重取樣
from openwakeword.model import Model
import collections
import wave
import webrtcvad
import time
import requests
import os
import asyncio
import edge_tts
from langdetect import detect
import os
 # server_url：這裡要填接收端的 IP (192.168.2.108)

import rospy
from std_msgs.msg import String
import json
import signal as sys_signal
rospy.init_node('speech_publisher', anonymous=True)   # 初始化 ROS 節點
    
    # 建立 Publisher，發佈到 'visual_object_command' topic
pub = rospy.Publisher('visual_object_command', String, queue_size=10)
task_pub = rospy.Publisher('task_explanation', String, queue_size=10)
task_type_pub = rospy.Publisher('task_type', String, queue_size=10)
time.sleep(1)
rospy.loginfo("語音發佈器已啟動")


def gripper_response_callback(msg):

    command = msg.data
    asyncio.run(text_to_speech(command))


# send_wav_and_get_response("/home/gairobots/speech/command.wav")
# ============ 取樣率設定 ============
DEVICE_RATE = 48000      # 麥克風硬體支援的取樣率
TARGET_RATE = 16000      # openWakeWord/Whisper 需要的取樣率
TARGET_CHUNK = 1280      # 80ms @ 16kHz
DEVICE_CHUNK = int(DEVICE_RATE * 0.08)  # 80ms @ 48kHz = 3840 樣本
DEVICE_INDEX = 5         # 外接麥克風 index
DEVICE_CHANNELS = 1      # 2:立體聲
print(f"裝置取樣率: {DEVICE_RATE} Hz")
print(f"目標取樣率: {TARGET_RATE} Hz")
print(f"裝置區塊大小: {DEVICE_CHUNK} 樣本")
print(f"目標區塊大小: {TARGET_CHUNK} 樣本\n")

# ============ 全局初始化 ============
oww_model = Model(wakeword_models=["/home/gairobots/camera/speech/hey_anna.onnx"], inference_framework='onnx')
# command_model = WhisperModel("medium", device="cpu", compute_type="int8")
vad = webrtcvad.Vad(1)

print("正在初始化音訊設備...")
audio = pyaudio.PyAudio()

# 預熱模型（用 16kHz 數據）
print("預熱模型...")
dummy_audio = np.zeros(TARGET_CHUNK, dtype=np.int16)
_ = oww_model.predict(dummy_audio)
print("✓ 模型已就緒\n")


# 全局音訊串流
main_stream = None
last_detection_time = 0  # ← 新增：全局冷卻時間記錄
exit_flag = False  # 退出
def signal_handler(signum, frame):
    """處理 Ctrl+C """
    global exit_flag
    print("\n\n收到中断 (Ctrl+C)，正在安全關閉...")
    exit_flag = True  # ← 设置退出标志

sys_signal.signal(sys_signal.SIGINT, signal_handler)
sys_signal.signal(sys_signal.SIGTERM, signal_handler)

def publisher_object_name(visual_object_name):
    """發佈物體名稱指令"""
    msg = String()
    # msg.data = visual_object_name
    msg.data = json.dumps(visual_object_name)  # 轉換為 JSON 字串(ex '["water bottle", "cup"]')
    pub.publish(msg)
    rospy.loginfo(f"發送物體名稱指令: {visual_object_name}")
    rospy.sleep(0.1)  # 確保訊息發送完成
    
def publisher_task_explanation(task):
    msg = String()
    msg.data = task
    task_pub.publish(msg)
    rospy.loginfo(f"發送任務解釋: {task}")
    rospy.sleep(0.1)  # 確保訊息發送完成


def publisher_task_type(task_type):
    msg = String()
    msg.data = task_type
    task_type_pub.publish(msg)
    rospy.loginfo(f"發送任務類型: {task_type}")
    rospy.sleep(0.1)  # 確保訊息發送完成

async def text_to_speech(answer):
    """異步：文字轉語音並播放"""
    try:
        lang = detect(answer)
        print(f"Detected language: {lang}")
        
        if lang in ['zh-cn', 'zh-tw', 'zh']:
            # voice = "zh-CN-XiaoxiaoNeural" 
            voice = "zh-TW-HsiaoChenNeural"
        elif lang == 'en':
            voice = "en-US-JennyNeural"
        else:
            voice = "zh-TW-HsiaoChenNeural"  # 預設英文
        
        communicate = edge_tts.Communicate(answer, voice)
        await communicate.save("output.mp3")
        
        print("🔊 播放語音中...")
        os.system("ffplay -nodisp -autoexit output.mp3")
        
    except Exception as e:
        print(f"語音合成錯誤: {e}")




def send_wav_and_get_response(file_path):
    server_url = "http://192.168.2.108:9000/upload_wav"
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (file_path, f, 'audio/wav')}
            response = requests.post(server_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 上傳成功")
            print(f"辨識結果: {result['user_command']}")
            print(f"LLM 回覆: {result['response']}")  # 這裡拿到回覆
            llm_response = result.get('response', '')
            if llm_response:
                    asyncio.run(text_to_speech(llm_response))
            return result
        else:
            print(f"✗ 失敗: {response.text}")
            return None
    except Exception as e:
        print(f"✗ 錯誤: {e}")
        return None




def init_main_stream():
    """初始化或重新初始化主音訊串流（48kHz）"""
    global main_stream
    
    # 如果串流已存在且開啟，先關閉
    if main_stream is not None:
        try:
            if main_stream.is_active():
                main_stream.stop_stream()
            main_stream.close()
        except:
            pass
    
    # 開啟新串流（48kHz）
    main_stream = audio.open(
        format=pyaudio.paInt16,
        channels=DEVICE_CHANNELS,
        rate=DEVICE_RATE,           # ← 改用 48kHz
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=DEVICE_CHUNK  # ← 改用 3840
    )
    
    # 清空初始緩衝區
    for _ in range(10):
        try:
            main_stream.read(DEVICE_CHUNK, exception_on_overflow=False)
        except:
            pass
    
    return main_stream


# 初始化主串流
print("清空音訊緩衝區...")
init_main_stream()
print("✓ 音訊串流已就緒\n")

def detect_wake_word():
    """使用 Anna 模型檢測喚醒詞（48kHz → 16kHz）"""
    global main_stream
    global last_detection_time  # ← 新增：聲明使用全局變數
    
    print("等待喚醒詞: 'ANNA'（按 Ctrl+C 退出）...")
    print("（顯示即時檢測分數，用於調試）\n")
    
    cooldown_period = 2.0
    frame_count = 0
    
    while True:
        if exit_flag:
            print("\n检测到退出信号...")
            return False
        try:
            if not main_stream.is_active():
                print("偵測到串流關閉，重新初始化...")
                init_main_stream()
            try:
                data = main_stream.read(DEVICE_CHUNK, exception_on_overflow=False)
                audio_raw = np.frombuffer(data, dtype=np.int16)
            except KeyboardInterrupt:
                print("\n捕获到 Ctrl+C，立即退出...")
                raise  
            
            
            # 立體聲 → 單聲道
            if DEVICE_CHANNELS == 2:
                audio_48k = audio_raw.reshape(-1, 2).mean(axis=1).astype(np.int16)
            else:
                audio_48k = audio_raw
            
            # 重取樣到 16kHz
            audio_16k = signal.resample(audio_48k, TARGET_CHUNK).astype(np.int16)
            
            # 用 16kHz 音訊預測
            prediction = oww_model.predict(audio_16k)
            current_time = time.time()
            frame_count += 1
            
            for model_name, score in prediction.items():
                # ===== 改進：顯示冷卻期狀態 =====
                if score > 0.15:
                    # 檢查是否在冷卻期
                    if (current_time - last_detection_time) < cooldown_period:
                        remaining = cooldown_period - (current_time - last_detection_time)
                        print(f"[冷卻中 {remaining:.1f}s] 分數: {score:.3f}  ", end='\r')
                    else:
                        print(f"[即時分數] {score:.3f} ", end='\r')
                
                # ===== 在冷卻期內跳過 =====
                if (current_time - last_detection_time) < cooldown_period:
                    continue
                
                if score > 0.45:
                    print(f"\n✓ 檢測到 '{model_name}'! (信心度: {score:.2f})")
                    
                    # 清空緩衝區
                    for _ in range(10):
                        try:
                            main_stream.read(DEVICE_CHUNK, exception_on_overflow=False)
                        except:
                            break
                    
                    last_detection_time = current_time  # ← 更新全局冷卻時間
                    return True
        except KeyboardInterrupt:
            # 立即退出，不处理
            raise
        except Exception as e:
            if "Stream not open" in str(e):
                print("\n串流異常，重新初始化...")
                init_main_stream()
            else:
                raise



def write_wave(path, frames):
    """寫入 WAV 檔案（16kHz）"""
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(TARGET_RATE)  # ← 用 16kHz
    wf.writeframes(b''.join(frames))
    wf.close()
    return path


def record_command():
    """錄製使用者命令（48kHz → 16kHz）"""
    print("開始錄音，請說出指令...")
    
    record_stream = None
    wavfile_path = None
    
    try:
        # ===== 錄音串流也用 48kHz =====
        record_stream = audio.open(
            format=pyaudio.paInt16,
            channels=DEVICE_CHANNELS,  # ← 改成 2
            rate=DEVICE_RATE,
            input=True,
            input_device_index=DEVICE_INDEX,
            frames_per_buffer=DEVICE_CHUNK
        )

        
        # 清空初始緩衝區
        for _ in range(5):
            record_stream.read(DEVICE_CHUNK, exception_on_overflow=False)
        
        audio_buffer = []  # 存放 16kHz 音訊
        voiced_frames = collections.deque(maxlen=int(1.0 / 0.08))  # 改成 80ms 為單位
        triggered = False
        silence_duration = 0
        
        while True:
            # # ===== 讀取 48kHz 音訊 =====
            
            data = record_stream.read(DEVICE_CHUNK, exception_on_overflow=False)
            audio_raw = np.frombuffer(data, dtype=np.int16)

            # ===== 立體聲 → 單聲道 =====
            if DEVICE_CHANNELS == 2:
                audio_48k = audio_raw.reshape(-1, 2).mean(axis=1).astype(np.int16)
            else:
                audio_48k = audio_raw

            # ===== 重取樣到 16kHz =====
            audio_16k = signal.resample(audio_48k, TARGET_CHUNK).astype(np.int16)

            frame_16k = audio_16k.tobytes()
            
            # ===== VAD 判斷（用 16kHz 音訊）=====
            # is_speech_frame = vad.is_speech(frame_16k, TARGET_RATE)
            # ===== VAD 判斷（只用前 30ms = 480 樣本）=====
            vad_chunk = audio_16k[:480].tobytes()  # 取前 480 樣本
            try:
                is_speech_frame = vad.is_speech(vad_chunk, TARGET_RATE)
            except:
                is_speech_frame = False  # VAD 失敗時當作無語音

            
            if not triggered:
                voiced_frames.append((frame_16k, is_speech_frame))
                num_voiced = sum(1 for _, speech in voiced_frames if speech)
                
                if num_voiced > 0.8 * voiced_frames.maxlen:
                    triggered = True
                    # 把之前的幀都加進去
                    for f, _ in voiced_frames:
                        audio_buffer.append(f)
                    voiced_frames.clear()
                    print("偵測到語音...")
            else:
                audio_buffer.append(frame_16k)
                
                if not is_speech_frame:
                    silence_duration += 0.08  # 80ms
                    if silence_duration > 1.5:
                        print("錄音完成，正在保存...")
                        wavfile_path = write_wave("command.wav", audio_buffer)
                        print(f"✓ 音訊已保存到: {wavfile_path}")
                        break
                else:
                    silence_duration = 0
    
    except Exception as e:
        print(f"錄音時發生錯誤: {e}")
    
    finally:
        # 確保關閉錄音串流
        if record_stream is not None:
            try:
                if record_stream.is_active():
                    record_stream.stop_stream()
                record_stream.close()
            except:
                pass
        
        return wavfile_path



def cleanup():
    """清理资源（强化版 - 确保麦克风完全释放）"""
    global main_stream, audio
    
    print("\n正在关闭音频设备...")
    
    # 1. 关闭主流
    if main_stream is not None:
        try:
            if main_stream.is_active():
                main_stream.stop_stream()
            main_stream.close()
            print("✓ 主音频流已关闭")
        except Exception as e:
            print(f"关闭主流时警告: {e}")
        finally:
            main_stream = None
    
    # ===== 添加延迟，等待 ALSA 释放 =====
    import time
    time.sleep(0.5)
    # =====================================
    
    # 2. 终止 PyAudio
    try:
        audio.terminate()
        print("✓ PyAudio 已终止")
    except Exception as e:
        print(f"终止 PyAudio 时警告: {e}")
    
    # ===== 再次延迟 =====
    time.sleep(0.5)
    # ====================
    
    # 3. 强制释放 ALSA 音频设备
    try:
        import subprocess
        subprocess.run("fuser -k /dev/snd/* 2>/dev/null", shell=True, timeout=2)
        print("✓ 已强制释放音频设备")
    except:
        pass
    
    # ===== 最后延迟，确保设备完全空闲 =====
    time.sleep(1.0)
    # =======================================
    
    print("✓ 程序已安全退出")
    
    import os
    os._exit(0)

def send_wav_file(file_path):
    """
    發送 WAV 檔案到 server 並取得回應
    
    Args:
        file_path: WAV 檔案路徑
    
    Returns:
        dict: 包含 user_command 和 response 的字典，失敗時返回 None
    """
    server_url = "http://192.168.2.108:9000/upload_wav"
    
    try:
        # 檢查檔案是否存在
        if not os.path.exists(file_path):
            print(f"✗ 錯誤: 檔案不存在 - {file_path}")
            return None
        
        # 開啟檔案並上傳
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
            
            print(f"📤 正在上傳音訊檔案...")
            response = requests.post(
                server_url, 
                files=files,
                timeout=120  # 2分鐘超時（配合 Whisper 處理時間）
            )
        
        # 檢查 HTTP 狀態碼
        if response.status_code == 200:
            result = response.json()
            print(f"✓ 上傳成功")
            print(f"辨識結果: {result['user_command']}")
            print(f"LLM 回覆: {result['response']}")
            print(f"visual task: {result.get('visual_task', '無')}")
            
            # 如果有 LLM 回覆，播放語音
            llm_response = result.get('response', '')
            visual_task = result.get('visual_task', '')
            if llm_response:
                asyncio.run(text_to_speech(llm_response))
                if visual_task and visual_task != '':
                    visual_object_name = visual_task['target_objects']
                    publisher_object_name(visual_object_name)
                    task_explanation = visual_task['explanation']
                    publisher_task_explanation(task_explanation)
                    task_type = visual_task['task_type']
                    publisher_task_type(task_type)
            
            return result
        else:
            print(f"✗ 上傳失敗 (HTTP {response.status_code})")
            print(f"錯誤訊息: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"✗ 錯誤: 請求超時（可能 Whisper 處理時間過長）")
        return None
    except requests.exceptions.ConnectionError:
        print(f"✗ 錯誤: 無法連線到 server ({server_url})")
        print(f"   請確認 server 是否執行中")
        return None
    except Exception as e:
        print(f"✗ 發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主循環"""
    global main_stream
    global last_detection_time
    global exit_flag 
    print("=" * 50)
    print("語音助手已啟動 (Anna Wake Word)")
    print("喚醒詞: 'ANNA'")
    print(f"硬體取樣率: {DEVICE_RATE} Hz")
    print(f"處理取樣率: {TARGET_RATE} Hz")
    print("提示: 按 Ctrl+C 可以隨時退出程式")
    print("=" * 50)
    
    try:
        while True:
            if exit_flag:
                print("主循环收到退出信号...")
                break
            if detect_wake_word():
                print("释放麦克风...")
                if main_stream is not None:
                    try:
                        if main_stream.is_active():
                            main_stream.stop_stream()
                        main_stream.close()
                        main_stream = None
                    except:
                        pass
                asyncio.run(text_to_speech("我在聽，怎麼了"))

                command_path = record_command()
                
                if command_path:
                    print("正在辨識命令...")
                    send_wav_file(command_path)
                    # segments, _ = command_model.transcribe(
                    #     command_path, 
                    #     language="zh",
                    #     temperature=0.1,
                    #     vad_filter=True
                    # )
                    # result = "".join([seg.text for seg in segments])
                    # print(f"\n【辨識結果】: {result}\n")
                
                # ===== 加強重啟流程 =====
                print("重新初始化串流...")
                try:
                    # 1. 重啟串流
                    init_main_stream()
                    
                    # 2. 驗證串流是否活躍
                    if not main_stream.is_active():
                        print("⚠ 串流未活躍，重試...")
                        init_main_stream()
                    
                    # 3. 等待硬體穩定
                    time.sleep(1.5)  # 從 1.0 增加到 1.5
                    
                    # 4. 大量清空緩衝區（清除所有殘留）
                    print("大量清空緩衝區...")
                    for i in range(60):
                        try:
                            data = main_stream.read(DEVICE_CHUNK, exception_on_overflow=False)
                            # 每 10 幀顯示進度
                            if i % 10 == 0:
                                print(f"  清空進度: {i}/60", end="\r")
                        except:
                            break
                    
                    print("✓ 串流已重置                    ")
                    
                    # 5. 等待 1 秒後再重置冷卻時間（確保時間差）
                    time.sleep(1.0)
                    last_detection_time = time.time()
                    print("✓ 冷卻期已重置（2 秒）")
                    
                except Exception as e:
                    print(f"重新初始化失敗: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 失敗時強制重啟
                    init_main_stream()
                    time.sleep(2.0)
                    last_detection_time = time.time()

                print("-" * 50)
                print("準備下一次喚醒\n")
    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C 中断...")           
    except KeyboardInterrupt:
        print("\n\n收到中斷信號...")
    except Exception as e:
        print(f"\n發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()



if __name__ == "__main__":
    try:
        rospy.Subscriber('voice_command', String, gripper_response_callback)
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("語音發佈器已關閉")
    main()
