import time
import os
from anomaly_detector import load_config, is_gpu_abnormal, check_tensorboard_abnormal
from message_handler import send_error_to_port

import pynvml

def monitor_gpu(config):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    low_util_start = None

    while True:
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory_usage = mem_info.used / 1024 / 1024  # MB
            compute_usage = util.gpu  # %

            if is_gpu_abnormal(memory_usage, compute_usage, config):
                if low_util_start is None:
                    low_util_start = time.time()
                elif time.time() - low_util_start > config['gpu']['low_util_duration']:
                    send_error_to_port(f"GPU {i} 长时间（{config['gpu']['low_util_duration']}秒）低利用率，显存:{memory_usage}MB，计算:{compute_usage}%")
            else:
                low_util_start = None

        time.sleep(config['gpu']['check_interval'])

def monitor_tensorboard_logs(config):
    while True:
        for log_dir in config['tensorboard']['log_dirs']:
            if not os.path.exists(log_dir):
                continue
            for root, _, files in os.walk(log_dir):
                for file in files:
                    if file.endswith(".log") or file.endswith(".txt"):
                        with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                            abnormal_info = check_tensorboard_abnormal(content, config)
                            if abnormal_info:
                                send_error_to_port(f"TensorBoard 日志异常：{file}，详情：{abnormal_info}")
        time.sleep(config['gpu']['check_interval'])

if __name__ == "__main__":
    config = load_config()
    # 简化为串行轮询，减少线程资源占用
    while True:
        monitor_gpu(config)
        monitor_tensorboard_logs(config)
        time.sleep(60)