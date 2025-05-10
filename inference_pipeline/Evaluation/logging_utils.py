import os
import psutil
import time
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Setup basic logging
logging.basicConfig(level=logging.INFO)


def log_cpu_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    logging.info(f"CPU Usage: {cpu_percent}%")
    return cpu_percent


def log_memory_usage():
    mem = psutil.virtual_memory()
    logging.info(f"Memory Usage: {mem.percent}% ({mem.used / 1e9:.2f}GB/{mem.total / 1e9:.2f}GB)")
    return mem.percent, mem.used, mem.total


def log_disk_usage(path="/"):
    disk = psutil.disk_usage(path)
    logging.info(f"Disk Usage ({path}): {disk.percent}% ({disk.used / 1e9:.2f}GB/{disk.total / 1e9:.2f}GB)")
    return disk.percent, disk.used, disk.total


def log_gpu_usage():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1e9
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
            util = mem_alloc / mem_total * 100 if mem_total > 0 else 0
            logging.info(f"GPU {i}: {util:.2f}% ({mem_alloc:.2f}GB/{mem_total:.2f}GB)")
            gpu_stats.append((util, mem_alloc, mem_total))
        return gpu_stats
    else:
        logging.info("No GPU available or torch not installed.")
        return None


def log_all_system_metrics(disk_path="/"):
    """Log all system metrics and return as a dict."""
    metrics = {}
    metrics['cpu'] = log_cpu_usage()
    mem_percent, mem_used, mem_total = log_memory_usage()
    metrics['memory'] = {'percent': mem_percent, 'used': mem_used, 'total': mem_total}
    disk_percent, disk_used, disk_total = log_disk_usage(disk_path)
    metrics['disk'] = {'percent': disk_percent, 'used': disk_used, 'total': disk_total}
    gpu_stats = log_gpu_usage()
    metrics['gpu'] = gpu_stats
    return metrics


def log_model_info(model, model_name=None, model_path=None, device=None):
    """
    Log information about a loaded model.
    Args:
        model: The model object (can be any type).
        model_name (str, optional): Name or type of the model.
        model_path (str, optional): Path to the model file.
        device (str, optional): Device on which the model is loaded.
    """
    info = f"Model info: "
    if model_name is not None:
        info += f"name={model_name}, "
    info += f"class={type(model).__name__}, "
    if model_path is not None:
        info += f"path={model_path}, "
    if device is not None:
        info += f"device={device}, "
    logging.info(info.rstrip(', ')) 