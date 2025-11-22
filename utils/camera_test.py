import cv2
import time
from threading import Thread

# 测试配置
CAM_IDS = [4, 6]  # 两个摄像头索引
TEST_DUR = 10     # 测试时长(秒)
MAX_RETRY = 3     # 单帧重试次数
RETRY_DELAY = 0.05

def test_cam(idx):
    """单摄像头测试线程"""
    win_name = f"Cam {idx}"
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"❌ Cam{idx}: 无法打开")
        return

    # 基础配置
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -6)  # 自动曝光
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    start = time.time()
    frame_cnt = 0
    err_cnt = 0

    # 读取+显示循环
    while time.time() - start < TEST_DUR:
        ret, frame = False, None
        for _ in range(MAX_RETRY):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
            time.sleep(RETRY_DELAY)

        if ret:
            frame_cnt += 1
            # 叠加状态文字
            fps = frame_cnt / (time.time() - start)
            cv2.putText(frame, f"Cam{idx} | FPS:{fps:.1f} | Err:{err_cnt}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow(win_name, frame)
        else:
            err_cnt += 1
            print(f"❌ Cam{idx}: 第{err_cnt}次读取失败")

        # 按q退出所有
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理+结果
    cap.release()
    cv2.destroyWindow(win_name)
    print(f"\n✅ Cam{idx}: 帧数{frame_cnt} | 错误{err_cnt} | 平均FPS:{frame_cnt/(time.time()-start):.1f}")

if __name__ == "__main__":
    print(f"=== 双摄像头测试 (索引{CAM_IDS}) ===")
    print(f"测试{TEST_DUR}秒 | 按q退出")
    
    # 启动双线程
    threads = [Thread(target=test_cam, args=(idx,)) for idx in CAM_IDS]
    [t.start() for t in threads]
    [t.join() for t in threads]
    
    cv2.destroyAllWindows()
    print("\n=== 测试结束 ===")
