import mlflow
import psutil
import GPUtil
import torch

def log_all_system_metrics():
    # CPU & RAM
    cpu_usage = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    mlflow.log_metric("cpu_usage_percent", cpu_usage)
    mlflow.log_metric("ram_usage_percent", ram.percent)
    mlflow.log_metric("ram_used_gb", ram.used / (1024 ** 3))
    mlflow.log_metric("ram_total_gb", ram.total / (1024 ** 3))

    # GPU (if available)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assumes single-GPU setup
        mlflow.log_metric("gpu_usage_percent", gpu.load * 100)
        mlflow.log_metric("gpu_mem_used_mb", gpu.memoryUsed)
        mlflow.log_metric("gpu_mem_total_mb", gpu.memoryTotal)
        mlflow.log_metric("gpu_mem_util_percent", gpu.memoryUtil * 100)
        mlflow.log_param("gpu_name", gpu.name)

def log_model_info(model, model_name: str, model_path: str = None, device: str = "cpu"):
    mlflow.log_param(f"{model_name}_device", device)
    if model_path:
        mlflow.log_param(f"{model_name}_path", model_path)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_metric(f"{model_name}_num_params", num_params)
import mlflow
import psutil
import GPUtil
import torch

def log_all_system_metrics():
    # CPU & RAM
    cpu_usage = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    mlflow.log_metric("cpu_usage_percent", cpu_usage)
    mlflow.log_metric("ram_usage_percent", ram.percent)
    mlflow.log_metric("ram_used_gb", ram.used / (1024 ** 3))
    mlflow.log_metric("ram_total_gb", ram.total / (1024 ** 3))

    # GPU (if available)
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assumes single-GPU setup
        mlflow.log_metric("gpu_usage_percent", gpu.load * 100)
        mlflow.log_metric("gpu_mem_used_mb", gpu.memoryUsed)
        mlflow.log_metric("gpu_mem_total_mb", gpu.memoryTotal)
        mlflow.log_metric("gpu_mem_util_percent", gpu.memoryUtil * 100)
        mlflow.log_param("gpu_name", gpu.name)

def log_model_info(model, model_name: str, model_path: str = None, device: str = "cpu"):
    mlflow.log_param(f"{model_name}_device", device)
    if model_path:
        mlflow.log_param(f"{model_name}_path", model_path)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_metric(f"{model_name}_num_params", num_params)
