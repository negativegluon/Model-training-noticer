import yaml

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def is_gpu_abnormal(memory_usage, compute_usage, config):
    return (memory_usage < config['gpu']['memory_threshold'] and
            compute_usage < config['gpu']['compute_threshold'])



#下面是重要的需要自定义的函数，这里提供一个检测训练收敛的示例


def check_tensorboard_abnormal(log_content, config=None):
    """
    仅判断val_loss收敛或灾难性梯度爆炸。
    如果step数小于阈值或未检测到val_loss则直接报错。
    否则检测判断窗口长度中所有可用val_loss值。
    """
    import re

    # 内部预设参数
    exploding_loss_threshold = 1e6  # 梯度爆炸阈值
    converge_window = 5             # 收敛判断窗口长度
    converge_eps = 1e-3             # 收敛判断变化阈值
    min_steps_required = 5          # 最小step数阈值

    # 提取val_loss数值
    val_loss_pattern = re.compile(r"val_loss[:=]\s*([0-9\.eE+-]+|nan)", re.IGNORECASE)
    val_losses = []

    for line in log_content.splitlines():
        val_loss_match = val_loss_pattern.search(line)
        if val_loss_match:
            try:
                val_losses.append(float(val_loss_match.group(1)))
            except ValueError:
                val_losses.append(float('nan'))

    # step数不足或没有val_loss直接报错
    if len(val_losses) < min_steps_required:
        return "错误：step数不足或未检测到val_loss"

    # 判断灾难性梯度爆炸
    for l in val_losses:
        if l != l or abs(l) > exploding_loss_threshold:
            return "检测到梯度爆炸或val_loss为nan"

    # 判断收敛（最近N次变化幅度很小）
    if len(val_losses) >= converge_window:
        recent_val_losses = val_losses[-converge_window:]
        val_loss_var = max(recent_val_losses) - min(recent_val_losses)
        if val_loss_var < converge_eps:
            return f"val_loss已收敛，{len(recent_val_losses)}个最近值变化幅度：{val_loss_var:.6f}"

    return None     # 无异常则返回None


if __name__ == "__main__":
    # 示例日志内容
    test_log = """
    step 1: loss=0.5 val_loss=0.6
    step 2: loss=0.49 val_loss=0.59
    step 3: loss=0.48 val_loss=0.58
    step 4: loss=0.47 val_loss=0.57
    step 5: loss=0.46 val_loss=0.56
    """

    result = check_tensorboard_abnormal(test_log)
    print("检测结果:", result)