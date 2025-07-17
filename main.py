import time
import os
from anomaly_detector import load_config, is_gpu_abnormal, check_tensorboard_abnormal
from message_handler import send_error_to_port
from tensorboard.backend.event_processing import event_accumulator

import pynvml

def monitor_gpu(config):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    low_util_start = None


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


def monitor_tensorboard_logs(config):

    for log_dir in config['tensorboard']['log_dirs']:
        if not os.path.exists(log_dir):
            continue
        # 自动识别TensorBoard事件文件
        for root, _, files in os.walk(log_dir):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    event_file = os.path.join(root, file)
                    try:
                        ea = event_accumulator.EventAccumulator(event_file)
                        ea.Reload()
                        # 获取所有loss和val_loss的scalar数据
                        losses = ea.Scalars('loss') if 'loss' in ea.Tags()['scalars'] else []
                        val_losses = ea.Scalars('val_loss') if 'val_loss' in ea.Tags()['scalars'] else []
                        # 构造log_content字符串，便于后续异常检测
                        
                        log_lines = []
                        for l in losses:
                            log_lines.append(f"step {l.step}: loss={l.value}")
                        for vl in val_losses:
                            log_lines.append(f"step {vl.step}: val_loss={vl.value}")
                        log_content = "\n".join(log_lines)
                        abnormal_info = check_tensorboard_abnormal(log_content)
                        if abnormal_info:
                            send_error_to_port(f"TensorBoard 日志异常：{file}，详情：{abnormal_info}")
                    except Exception as e:
                        send_error_to_port(f"TensorBoard 日志解析失败：{file}，错误：{e}")
if __name__ == "__main__":
    config = load_config()
    # 简化为串行轮询，减少线程资源占用
    while True:
        monitor_gpu(config)
        monitor_tensorboard_logs(config)
        time.sleep(config['check_interval'])