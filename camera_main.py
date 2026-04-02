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
# import ProcessObjectInfo, ProcessObjectInfoRequest
from robot_core.srv import BatchTransform, BatchTransformRequest, ArmBatchTransform, ArmBatchTransformRequest
print("✓ robot_core 导入完成，md5sum:", BatchTransform._md5sum)
# ============ 全域變數 ============
# camera_pub, camera_left_pub, camera_right_pub = None
rospy.init_node('camera_node', anonymous=True)
# 創建 Publisher (發送相機座標)
# camera_head_pub = rospy.Publisher('/camera/head', Point, queue_size=10)
# camera_right_pub = rospy.Publisher('/camera/right', Point, queue_size=10)
# camera_left_pub= rospy.Publisher('/camera/left', Point, queue_size=10)

total_object_pub = rospy.Publisher('/camera/total_objects', String, queue_size=10)
left_object_pub = rospy.Publisher('/camera/left_objects', String, queue_size=10)
right_object_pub = rospy.Publisher('/camera/right_objects', String, queue_size=10)
camera_ready_pub = rospy.Publisher('/camera/camera_ready', String, queue_size=10)
received_base_positions = []  # 按順序儲存收到的基座標
lock = threading.Lock()

# 923322070636:頭
# 243322074668:左手
# 243222072706:右手
cameras_config = [
    {
        "serial": "923322070636", # 923322070636
        "id": 0,
        # "max_objects": 1, #2
        "max_objects": 2,
        "phrases": ["tool", "brush tool", "dustpan tool"]
        # "phrases": ["bottle"]
        # "phrases":[ "dustpan tool"]
    },
    # 新增第二台相機
    {
        "serial": "243322074668", # 243322074668:左手
        "id": 1,
        "max_objects": 1,
        # "phrases":[ "dustpan tool"]
        "phrases":[ "brush tool"]
    },
    # 新增第三台相機
    {
        "serial": "243222072706", # 243222072706:右手
        "id": 2,
        "max_objects": 1,
        # "phrases":[ "brush tool"]
        "phrases":[ "dustpan tool"]
    },
]

# 初始化系統
system = MultiCameraSystem(cameras_config)


################################## callback functions ##################################
"""語音系統->相機 ：接收辨識物體名稱"""
def object_command_callback(msg):
    """接收物體名稱指令並更新 shared_object"""
    visual_object_name = msg.data
    print(f"ffff")
    visual_object_name = json.loads(msg.data)  # ["water bottle", "cup"]
    rospy.loginfo(f"收到物體名稱指令: {visual_object_name}")
    system.update_camera_phrases(0, visual_object_name)
    camera_ready_pub.publish("head_ready")
    # head_camera_capture()
# def assign_object_phase_callback(msg):
#     """接收物體名稱指令並更新 shared_object"""
#     print(f"sssss")
#     visual_object_name = msg.data
#     rospy.loginfo(f"收到物體名稱指令: {visual_object_name}")
#     system.update_camera_phrases(0, [visual_object_name])
#     camera_ready_pub.publish("head_ready")

def assign_object_phase_callback(msg):
    try:
        # 1. 解析字串
        # msg.data 是 '["rice food"]' -> loads 後變成 ["rice food"] (List)
        visual_object_names = json.loads(msg.data)
        
        # 2. 防呆檢查：確保它是 List
        if isinstance(visual_object_names, str):
            visual_object_names = [visual_object_names]
            
        rospy.loginfo(f"收到物體名稱指令: {visual_object_names}")
        
        # 3. 【關鍵修正】直接傳入變數，絕對不要寫成 [visual_object_names]
        system.update_camera_phrases(0, visual_object_names)
        
        camera_ready_pub.publish("head_ready")
        
    except Exception as e:
        rospy.logerr(f"解析失敗: {e}")
    # head_camera_capture()
"""接收轉換後的基座標"""
def base_callback(msg):
    """接收轉換後的基座標"""
    global received_base_positions
    
    bx, by, bz = msg.x, msg.y, msg.z
    
    with lock:
        received_base_positions.append((bx, by, bz))
        rospy.loginfo(f"收到基座標 [{len(received_base_positions)-1}]: ({bx:.1f}, {by:.1f}, {bz:.1f})")




"""接收指令並執行對應相機拍攝"""
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
        


"""座標轉換回調函數"""
   
# def arm_object_transform_callback(camera_id=2):
#     try:
#         rospy.loginfo("=== 开始发送相机坐标 ===")
#         if camera_id == 1:
#             save_obj= shared_object.left
#             camera_pub = camera_left_pub
#         elif camera_id == 2:
#             save_obj= shared_object.right
#             camera_pub = camera_right_pub
#         # ✅ 检查列表是否为空
#         if not save_obj or len(save_obj) == 0:
#             rospy.logwarn("没有檢測到物體")
#             return
        
#         # 取最后一个
#         obj_info = save_obj[-1]
#         px, py, pz = obj_info['center_pos']
        
#         # 發布相機座標
#         point_msg = Point()
#         point_msg.x = float(px)
#         point_msg.y = float(py)
#         point_msg.z = float(pz)
#         camera_pub.publish(point_msg)
        
#         rospy.loginfo(f"发送坐标: ({px:.1f}, {py:.1f}, {pz:.1f})")
        
#     except (KeyError, IndexError, TypeError) as e:
#         rospy.logerr(f"相机检测出错: {e}")

"""發布頭部相機物體座標並接收轉換結果"""
# def objectPrompt_callback():
#     """發布頭部相機物體座標並接收轉換結果"""
#     global camera_head_pub, received_base_positions
    
#     # 清空之前的結果
#     # with lock:
#     #     received_base_positions = []
    
#     # # === 步驟 1: 發送所有物體的相機座標 ===
#     # rospy.loginfo("=== 開始發送相機座標 ===")
#     # for idx, obj_info in enumerate(shared_object.total):
#     #     px, py, pz = obj_info['center_pos']
        
#     #     # 發布相機座標
#     #     point_msg = Point()
#     #     point_msg.x = float(px)
#     #     point_msg.y = float(py)
#     #     point_msg.z = float(pz)
#     #     camera_head_pub.publish(point_msg)
        
#     #     rospy.loginfo(f"發送物體 [{idx}] 相機座標: ({px:.1f}, {py:.1f}, {pz:.1f})")
        
#     #     # 稍微延遲，確保訊息發送順序
#     #     rospy.sleep(0.05)
    
#     # # === 步驟 2: 等待接收所有基座標 ===
#     # rospy.loginfo("=== 等待接收基座標 ===")
#     # expected_count = len(shared_object.total)
#     # timeout = rospy.Time.now() + rospy.Duration(5.0)  # 最多等 5 秒
#     # rate = rospy.Rate(10)
    
#     # while rospy.Time.now() < timeout and not rospy.is_shutdown():
#     #     with lock:
#     #         if len(received_base_positions) >= expected_count:
#     #             break
#     #     rate.sleep()
    
#     # === 步驟 3: 更新 obj_info ===
#     with lock:
#         received_count = len(received_base_positions)
    
#     # rospy.loginfo(f"收到 {received_count}/{expected_count} 個基座標")
    
#     camera_information_prompt = "[object] info: \n"
    
#     for idx, obj_info in enumerate(shared_object.total):
#         px, py, pz = obj_info['center_pos']
#         angle = obj_info['angle']
        
#         # 更新基座標
#         with lock:
#             if idx < len(received_base_positions):
#                 bx, by, bz = received_base_positions[idx]
#                 obj_info['base_center_pos'] = (bx, by, bz)
#             else:
#                 obj_info['base_center_pos'] = None
        
#         # 生成提示資訊
#         camera_information_prompt += f"object_name: {obj_info['name']}\n"
#         camera_information_prompt += f"object_index: {idx}\n"
#         camera_information_prompt += f"camera_position: px={px:.1f}mm, py={py:.1f}mm, pz={pz:.1f}mm\n"
        
#         if obj_info['base_center_pos']:
#             bx, by, bz = obj_info['base_center_pos']
#             camera_information_prompt += f"base_position: bx={bx:.1f}mm, by={by:.1f}mm, bz={bz:.1f}mm\n"
#         else:
#             camera_information_prompt += "base_position: not received\n"
        
#         camera_information_prompt += f"object_angle: {angle:.1f} deg\n"
#         camera_information_prompt += f"pick_mode: {obj_info['pick_mode']}\n"
#         camera_information_prompt += "===============================\n"
    
#     print(camera_information_prompt)
#     rospy.loginfo("=== 處理完成 ===\n")


def total_object_publish():
    try:
        # 将整个 shared_object.total 转为 JSON
        json_data = json.dumps(shared_object.total, ensure_ascii=False, indent=2)
        
        # 发布到 ROS topic
        msg = String()
        msg.data = json_data
        total_object_pub.publish(msg)
        
        rospy.loginfo("=== 已发布对象信息到 /total_objects ===")
    except Exception as e:
        rospy.logerr(f"发布失败: {e}")
def left_object_publish():

    try:
        # 将整个 shared_object.left 转为 JSON
        json_data = json.dumps(shared_object.left, ensure_ascii=False, indent=2)
        
        # 发布到 ROS topic
        msg = String()
        msg.data = json_data
        left_object_pub.publish(msg)
        
        rospy.loginfo("=== 已发布对象信息到 /left_objects ===")
    except Exception as e:
        rospy.logerr(f"发布失败: {e}")
def right_object_publish():

    try:
        # 将整个 shared_object.right 转为 JSON
        json_data = json.dumps(shared_object.right, ensure_ascii=False, indent=2)
        
        # 发布到 ROS topic
        msg = String()
        msg.data = json_data
        right_object_pub.publish(msg)
        
        rospy.loginfo("=== 已发布对象信息到 /right_objects ===")
    except Exception as e:
        rospy.logerr(f"发布失败: {e}")       




def transform_points_service(field_name, output_field_name, service_name='batch_transform'):
    """
    通用的坐标转换函数
    
    Args:
        field_name: 输入字段名 (例如 'center_pos', 'left_endpoint')
        output_field_name: 输出字段名 (例如 'base_center_pos', 'left_base_pos')
        service_name: ROS Service 名称
    
    Returns:
        成功更新的数量
    """
    rospy.wait_for_service(service_name)
    
    try:
        transform_service = rospy.ServiceProxy(service_name, BatchTransform)
        req = BatchTransformRequest()
        pending_objects_map = {}
        
        # === 步骤 1: 打包数据 ===
        with lock:
            for idx, obj_info in enumerate(shared_object.total):
                if field_name in obj_info and len(obj_info[field_name]) >= 3:
                    px, py, pz = obj_info[field_name]
                    
                    req.ids.append(idx)
                    p = Point()
                    p.x = float(px)
                    p.y = float(py)
                    p.z = float(pz)
                    req.points.append(p)
                    
                    pending_objects_map[idx] = obj_info
                else:
                    rospy.logwarn(f"物件 {idx} 缺少 {field_name}，跳过转换")
        
        if not req.ids:
            rospy.logwarn(f"没有有效的 {field_name} 数据可转换")
            return 0
        
        # === 步骤 2: 调用 Service ===
        res = transform_service(req)
        
        # === 步骤 3: 更新数据 ===
        if res.success:
            update_count = 0
            
            for i in range(len(res.ids)):
                returned_id = res.ids[i]
                returned_pt = res.points[i]
                
                if returned_id in pending_objects_map:
                    target_obj = pending_objects_map[returned_id]
                    target_obj[output_field_name] = (returned_pt.x, returned_pt.y, returned_pt.z)
                    update_count += 1
            
            # 未转换成功的设为 None
            for idx, obj in pending_objects_map.items():
                if output_field_name not in obj:
                    obj[output_field_name] = None
            
            rospy.loginfo(f"成功更新 {update_count} 个物体的 {output_field_name}")
            return update_count
        else:
            rospy.logwarn(f"Service 回传 success=False，{field_name} 转换失败")
            return 0
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return 0
    except Exception as e:
        rospy.logerr(f"Python processing error: {e}")
        return 0


def arm_transform_points_service(field_name, output_field_name, arm_id, service_name='Arm_batch_transform'):
    """
    通用的坐标转换函数
    
    Args:
        field_name: 输入字段名 (例如 'center_pos', 'left_endpoint')
        output_field_name: 输出字段名 (例如 'base_center_pos', 'left_base_pos')
        service_name: ROS Service 名称
    
    Returns:
        成功更新的数量
    """
    rospy.wait_for_service(service_name)
    
    try:
        transform_service = rospy.ServiceProxy(service_name, ArmBatchTransform)
        req = ArmBatchTransformRequest()
        pending_objects_map = {}
        
        if arm_id ==1:
            obj_list= shared_object.left
        elif arm_id ==2:
            obj_list= shared_object.right
        else:
            rospy.logwarn(f"未知的 arm_id: {arm_id}")
            return 0
        
        # === 步骤 1: 打包数据 ===
        with lock:
            for idx, obj_info in enumerate(obj_list):
                if field_name in obj_info and len(obj_info[field_name]) >= 3:
                    px, py, pz = obj_info[field_name]
                    req.arm = arm_id
                    req.ids.append(idx)
                    p = Point()
                    p.x = float(px)
                    p.y = float(py)
                    p.z = float(pz)
                    req.points.append(p)
                    
                    pending_objects_map[idx] = obj_info
                else:
                    rospy.logwarn(f"物件 {idx} 缺少 {field_name}，跳过转换")
        
        if not req.ids:
            rospy.logwarn(f"没有有效的 {field_name} 数据可转换")
            return 0
        
        # === 步骤 2: 调用 Service ===
        res = transform_service(req)
        
        # === 步骤 3: 更新数据 ===
        if res.success:
            update_count = 0
            
            for i in range(len(res.ids)):
                returned_id = res.ids[i]
                returned_pt = res.points[i]
                
                if returned_id in pending_objects_map:
                    target_obj = pending_objects_map[returned_id]
                    target_obj[output_field_name] = (returned_pt.x, returned_pt.y, returned_pt.z)
                    update_count += 1
            
            # 未转换成功的设为 None
            for idx, obj in pending_objects_map.items():
                if output_field_name not in obj:
                    obj[output_field_name] = None
            
            rospy.loginfo(f"成功更新 {update_count} 个物体的 {output_field_name}")
            return update_count
        else:
            rospy.logwarn(f"Service 回传 success=False，{field_name} 转换失败")
            return 0
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
        return 0
    except Exception as e:
        rospy.logerr(f"Python processing error: {e}")
        return 0


def objectPos_callback_service(camera_name):

    """批量轉換所有需要的坐標"""
    if camera_name == 'head':
        # 轉換 center_pos
        transform_points_service('center_pos', 'base_center_pos')
        
        # 轉換 left_endpoint
        transform_points_service('left_endpoint', 'left_base_pos')
        
        # 如果還有其他字段需要轉換，繼續添加
        transform_points_service('right_endpoint', 'right_base_pos')
    elif camera_name == 'left':
        arm_transform_points_service('center_pos', 'base_center_pos', 1)

        arm_transform_points_service('left_endpoint', 'left_base_pos', 1)

        arm_transform_points_service('right_endpoint', 'right_base_pos', 1)


    elif camera_name == 'right':
        arm_transform_points_service('center_pos', 'base_center_pos', 2)
        arm_transform_points_service('left_endpoint', 'left_base_pos', 2)
        arm_transform_points_service('right_endpoint', 'right_base_pos', 2)
################################## 相機拍攝功能 ##################################
"""相機拍攝"""

def single_camera_capture(camera_id):
    global system
    
    try:
        print("\n準備照相環境")
        system.run_camera_detection(camera_id)
        if camera_id == 0:
            objectPos_callback_service('head')
        elif camera_id == 1:
            objectPos_callback_service('left')
            left_object_publish()
        elif camera_id == 2:
            objectPos_callback_service('right')
            right_object_publish()
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
        objectPos_callback_service('left')
        left_object_publish()
        objectPos_callback_service('right')
        right_object_publish()
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


        objectPos_callback_service('head')
        total_object_publish()
        rospy.loginfo("相機偵測完成")
    except Exception as e:
        rospy.logerr(f"相機偵測出錯: {e}")
    finally:
        torch.cuda.empty_cache()
        

"""訂閱節點"""
def ros_node():
    
    global camera_pub, camera_left_pub, camera_right_pub
    

    # 創建 Subscriber (接收基座標)
    rospy.Subscriber('/base/object_point', Point, base_callback)


    # 訂閱 'camera_command' topic，接收 String 型訊息
    rospy.Subscriber('camera_command', String, command_callback)
    
    rospy.Subscriber("visual_object_command", String, object_command_callback)
    rospy.Subscriber("assign_object_phase", String, assign_object_phase_callback)
    rospy.Subscriber("assign_left_object_phase", String, lambda msg: (system.update_camera_phrases(1, [msg.data]), camera_ready_pub.publish("left_ready")))
    rospy.Subscriber("assign_right_object_phase", String, lambda msg: (system.update_camera_phrases(2, [msg.data]), camera_ready_pub.publish("right_ready")))
    rospy.loginfo("相機訂閱器已啟動，等待指令...")
    
    # 保持節點運行
    rospy.spin()





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




####################################### camera test example #######################################
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
