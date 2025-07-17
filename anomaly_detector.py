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
    检查日志内容中的loss和val_loss是否收敛，或是否出现灾难性梯度爆炸。
    收敛判断：loss和val_loss在最近N次变化幅度很小（如小于阈值）。
    梯度爆炸判断：loss或val_loss突然变为极大值或nan。
    所有阈值均在函数内部预设。
    """
    import re

    # 内部预设参数
    exploding_loss_threshold = 1e6  # 梯度爆炸阈值
    converge_window = 10000         # 收敛判断窗口长度（step数）
    converge_eps = 5e-4             # 收敛判断变化阈值

    # 提取loss和val_loss数值
    loss_pattern = re.compile(r"loss[:=]\s*([0-9\.eE+-]+|nan)", re.IGNORECASE)
    val_loss_pattern = re.compile(r"val_loss[:=]\s*([0-9\.eE+-]+|nan)", re.IGNORECASE)

    losses = []
    val_losses = []

    for line in log_content.splitlines():
        loss_match = loss_pattern.search(line)
        val_loss_match = val_loss_pattern.search(line)
        if loss_match:
            try:
                losses.append(float(loss_match.group(1)))
            except ValueError:
                losses.append(float('nan'))
        if val_loss_match:
            try:
                val_losses.append(float(val_loss_match.group(1)))
            except ValueError:
                val_losses.append(float('nan'))

    # 判断灾难性梯度爆炸
    for l in losses + val_losses:
        if l != l or abs(l) > exploding_loss_threshold:
            return "检测到梯度爆炸或loss为nan"

    # 判断收敛（最近N次变化幅度很小）
    if len(losses) >= converge_window and len(val_losses) >= converge_window:
        recent_losses = losses[-converge_window:]
        recent_val_losses = val_losses[-converge_window:]
        loss_var = max(recent_losses) - min(recent_losses)
        val_loss_var = max(recent_val_losses) - min(recent_val_losses)
        if loss_var < converge_eps and val_loss_var < converge_eps:
            return "loss与val_loss均已收敛"

    return None     # 无异常则返回None