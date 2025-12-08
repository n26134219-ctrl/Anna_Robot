#!/usr/bin/env python3
import os
from geometry_msgs.msg import Point
# ⭐⭐⭐ 最重要：在導入 torch 之前設置環境變數
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
"""
主程式 - 示範多相機系統使用
"""
from multi_camera_system import MultiCameraSystem
import time
from camera_detector import CameraDetector
import shared_object
import torch
import threading
import rospy
from std_msgs.msg import String
from multi_camera_system import MultiCameraSystem
import json
# ============ 全域變數 ============
# camera_pub, camera_left_pub, camera_right_pub = None
rospy.init_node('camera_node', anonymous=True)
# 創建 Publisher (發送相機座標)
camera_head_pub = rospy.Publisher('/camera/head', Point, queue_size=10)
camera_right_pub = rospy.Publisher('/camera/right', Point, queue_size=10)
camera_left_pub= rospy.Publisher('/camera/left', Point, queue_size=10)
received_base_positions = []  # 按順序儲存收到的基座標
lock = threading.Lock()

# 923322070636:頭
# 243322074668:左手
# 243222072706:右手



def object_command_callback(msg):
    """接收物體名稱指令並更新 shared_object"""
    visual_object_name = msg.data

    visual_object_name = json.loads(msg.data)  # ["water bottle", "cup"]
    rospy.loginfo(f"收到物體名稱指令: {visual_object_name}")
    system.update_camera_phrases(0, visual_object_name)
    
    # head_camera_capture()
    

def command_callback(msg):
    """接收指令並執行相應功能"""
    global system
    command = msg.data
    rospy.loginfo(f"收到指令: {command}")
    
    
    if command == "capture_head":
        head_camera_capture()
    elif command == "capture_arms":
        arms_camera_capture()
    elif command == "capture_left":
        single_camera_capture(1)
    elif command == "capture_right":
        single_camera_capture(2)
    else:
        rospy.logwarn(f"未知指令: {command}")
        
cameras_config = [
    {
        "serial": "923322070636", # 923322070636
        "id": 0,
        "max_objects": 1, #2
        # "max_objects": 2,
        # "phrases": ["tool", "brush tool", "dustpan tool"]
        "phrases": ["bottle"]
        # "phrases":[ "dustpan tool"]
    },
    # 新增第二台相機
    {
        "serial": "243322074668", # 243322074668:左手
        "id": 1,
        "max_objects": 1,
        "phrases":[ "dustpan tool"]
    },
    # 新增第三台相機
    {
        "serial": "243222072706", # 243222072706:右手
        "id": 2,
        "max_objects": 1,
        "phrases":[ "brush tool"]
    },
]

    # 初始化系統
system = MultiCameraSystem(cameras_config)



   
def arm_object_transform_callback(camera_id=2):
    try:
        rospy.loginfo("=== 开始发送相机坐标 ===")
        if camera_id == 1:
            save_obj= shared_object.left
            camera_pub = camera_left_pub
        elif camera_id == 2:
            save_obj= shared_object.right
            camera_pub = camera_right_pub
        # ✅ 检查列表是否为空
        if not save_obj or len(save_obj) == 0:
            rospy.logwarn("没有檢測到物體")
            return
        
        # 取最后一个
        obj_info = save_obj[-1]
        px, py, pz = obj_info['center_pos']
        
        # 发布相机坐标
        point_msg = Point()
        point_msg.x = float(px)
        point_msg.y = float(py)
        point_msg.z = float(pz)
        camera_pub.publish(point_msg)
        
        rospy.loginfo(f"发送坐标: ({px:.1f}, {py:.1f}, {pz:.1f})")
        
    except (KeyError, IndexError, TypeError) as e:
        rospy.logerr(f"相机检测出错: {e}")


def single_camera_capture(camera_id):
    global system
    
    try:
        print("\n準備照相環境")
        system.run_camera_detection(camera_id)
        if camera_id == 0:
            objectPrompt_callback()
        elif camera_id == 1:
            arm_object_transform_callback(camera_id=1)
        elif camera_id == 2:
            arm_object_transform_callback(camera_id=2)
        
        rospy.loginfo("相機偵測完成")
    except Exception as e:
        rospy.logerr(f"相機偵測出錯: {e}")
    finally:
        torch.cuda.empty_cache()


def arms_camera_capture():
    global system
    
    try:
        print("\n準備照相環境")
        system.run_parallel()
        # system.run_camera_detection(2)
        arm_object_transform_callback(camera_id=1)
        arm_object_transform_callback(camera_id=2)
        # arm_object_transform_callback(camera_id=1)
        # system.run_camera_detection(1)
        
        rospy.loginfo("相機偵測完成")
    except Exception as e:
        rospy.logerr(f"相機偵測出錯: {e}")
    finally:
        torch.cuda.empty_cache()


def head_camera_capture():
    global system
    try:
        print("\n準備照相環境")
        system.run_camera_detection(0)
        # system.run_camera_detection(1)
        objectPrompt_callback()  # info
        rospy.loginfo("相機偵測完成")
    except Exception as e:
        rospy.logerr(f"相機偵測出錯: {e}")
    finally:
        torch.cuda.empty_cache()
        


def ros_node():
    """訂閱節點"""
    global camera_pub, camera_left_pub, camera_right_pub
    # 初始化 ROS 節點
    

    
    # 初始化时创建多个 publisher
    
    

    # 創建 Subscriber (接收基座標)
    rospy.Subscriber('/base/object_point', Point, base_callback)


    # 訂閱 'camera_command' topic，接收 String 型訊息
    rospy.Subscriber('camera_command', String, command_callback)
    
    #
    rospy.Subscriber("visual_object_command", String, object_command_callback)

    rospy.loginfo("相機訂閱器已啟動，等待指令...")
    
    # 保持節點運行
    rospy.spin()


def base_callback(msg):
    """接收轉換後的基座標"""
    global received_base_positions
    
    bx, by, bz = msg.x, msg.y, msg.z
    
    with lock:
        received_base_positions.append((bx, by, bz))
        rospy.loginfo(f"收到基座標 [{len(received_base_positions)-1}]: ({bx:.1f}, {by:.1f}, {bz:.1f})")


def objectPrompt_callback():
    """發布頭部相機物體座標並接收轉換結果"""
    global camera_head_pub, received_base_positions
    
    # 清空之前的結果
    with lock:
        received_base_positions = []
    
    # === 步驟 1: 發送所有物體的相機座標 ===
    rospy.loginfo("=== 開始發送相機座標 ===")
    for idx, obj_info in enumerate(shared_object.total):
        px, py, pz = obj_info['center_pos']
        
        # 發布相機座標
        point_msg = Point()
        point_msg.x = float(px)
        point_msg.y = float(py)
        point_msg.z = float(pz)
        camera_head_pub.publish(point_msg)
        
        rospy.loginfo(f"發送物體 [{idx}] 相機座標: ({px:.1f}, {py:.1f}, {pz:.1f})")
        
        # 稍微延遲，確保訊息發送順序
        rospy.sleep(0.05)
    
    # === 步驟 2: 等待接收所有基座標 ===
    rospy.loginfo("=== 等待接收基座標 ===")
    expected_count = len(shared_object.total)
    timeout = rospy.Time.now() + rospy.Duration(5.0)  # 最多等 5 秒
    rate = rospy.Rate(10)
    
    while rospy.Time.now() < timeout and not rospy.is_shutdown():
        with lock:
            if len(received_base_positions) >= expected_count:
                break
        rate.sleep()
    
    # === 步驟 3: 更新 obj_info ===
    with lock:
        received_count = len(received_base_positions)
    
    rospy.loginfo(f"收到 {received_count}/{expected_count} 個基座標")
    
    camera_information_prompt = "[object] info: \n"
    
    for idx, obj_info in enumerate(shared_object.total):
        px, py, pz = obj_info['center_pos']
        angle = obj_info['angle']
        
        # 更新基座標
        with lock:
            if idx < len(received_base_positions):
                bx, by, bz = received_base_positions[idx]
                obj_info['base_pos'] = (bx, by, bz)
            else:
                obj_info['base_pos'] = None
        
        # 生成提示資訊
        camera_information_prompt += f"object_name: {obj_info['name']}\n"
        camera_information_prompt += f"object_index: {idx}\n"
        camera_information_prompt += f"camera_position: px={px:.1f}mm, py={py:.1f}mm, pz={pz:.1f}mm\n"
        
        if obj_info['base_pos']:
            bx, by, bz = obj_info['base_pos']
            camera_information_prompt += f"base_position: bx={bx:.1f}mm, by={by:.1f}mm, bz={bz:.1f}mm\n"
        else:
            camera_information_prompt += "base_position: not received\n"
        
        camera_information_prompt += f"object_angle: {angle:.1f} deg\n"
        camera_information_prompt += f"pick_mode: {obj_info['pick_mode']}\n"
        camera_information_prompt += "===============================\n"
    
    print(camera_information_prompt)
    rospy.loginfo("=== 處理完成 ===\n")
def main():
    # 相機配置
    global system
    
    try:
        try:
            ros_node()
        except rospy.ROSInterruptException:
            rospy.loginfo("相機訂閱器已關閉")
    
    except KeyboardInterrupt:
        print("\n⚠️  使用者中斷")
    
    finally:
        system.cleanup()


def example_single_camera():
    """範例：單相機使用"""
    
    
    print("\n" + "="*60)
    print("🎯 單相機偵測範例")
    print("="*60 + "\n")
    
    # 創建偵測器實例
    detector = CameraDetector(
        realsense_serial="923322070636",
        camera_id=0,
        max_objects=1,
        candidate_phrases=["brush tool", "dustpan tool"]
    )
    
    try:
        # 獲取幀
        print("📷 獲取相機幀...\n")
        rgb, depth = detector.get_current_frame()
        
        if rgb is None or depth is None:
            print("❌ 無法獲取幀")
            return
        
        # 執行偵測
        
        print("🔍 開始偵測...\n")
        if detector.camera_id == 0:
            success = detector.detect_objects_simple()
        else:
            success = detector.detect_objects()

        if success:
            # 取得結果
            objects = detector.get_objects_info()
            print(f"\n✅ 成功偵測到 {len(objects)} 個物體")
            
            for obj in objects:
                
                print(f"  - {obj['name']}: {obj['center_pos']}")
        else:
            print("\n❌ 偵測失敗")
    
    finally:
        detector.cleanup()
        

def list_cameras():
    import pyrealsense2 as rs
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
if __name__ == '__main__':
    # 執行多相機系統
    main()
    # list_cameras()
    # 或執行單相機範例
    # example_single_camera()
