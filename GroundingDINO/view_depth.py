import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================================
# 請在這裡填入你想開啟的相機序號
# 選項: '243322074668', '243222072706', '923322070636'
TARGET_SERIAL = '243322074668' 
# ==========================================

def main():
    # 1. 設定管線
    pipeline = rs.pipeline()
    config = rs.config()

    # 【關鍵修改】指定要開啟的相機序號
    if TARGET_SERIAL:
        config.enable_device(TARGET_SERIAL)
        print(f"嘗試連接指定相機: {TARGET_SERIAL}")

    # 設定解析度 (640x480 是 D435i 最穩定的深度解析度)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 如果你也想看 RGB，可以把下面這行解開註解
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 2. 啟動相機
    print("正在啟動 RealSense... (請確認 USB 3.0 已連接)")
    try:
        pipeline.start(config)
        print(f"✅ 成功開啟相機: {TARGET_SERIAL}")
    except RuntimeError as e:
        print(f"❌ 啟動失敗: {e}")
        print(f"請確認序號 {TARGET_SERIAL} 是否正確，且該相機已連接至 USB 3.0")
        return

    try:
        while True:
            # 3. 等待影像幀
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            # 4. 取得中心點的距離
            width = depth_frame.get_width()
            height = depth_frame.get_height()
            # 取得畫面正中心的距離 (單位：公尺)
            dist = depth_frame.get_distance(width // 2, height // 2)

            # 5. 將深度數據轉換為彩色圖 (方便肉眼觀察)
            # 使用 RealSense 內建的著色器
            colorizer = rs.colorizer()
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_image = np.asanyarray(depth_color_frame.get_data())

            # 6. 在畫面上印出距離
            cv2.putText(depth_image, f"Dev: {TARGET_SERIAL}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(depth_image, f"Center Dist: {dist:.3f} m", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 畫一個中心點十字
            cv2.drawMarker(depth_image, (width//2, height//2), (0, 0, 0), cv2.MARKER_CROSS, 20, 2)

            # 7. 顯示視窗
            cv2.imshow(f'RealSense Depth - {TARGET_SERIAL}', depth_image)

            # 按 'q' 離開
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

    finally:
        # 8. 關閉並釋放資源
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()