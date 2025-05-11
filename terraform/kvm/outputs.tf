output "kvm_node_floating_ip" {
  description = "Floating IP address of the KVM node (MLflow/MinIO/Postgres)"
  value       = var.create_kvm_node ? openstack_networking_floatingip_v2.fip_kvm_group15_project[0].address : "KVM node not created"
}

output "ssh_command_kvm" {
  description = "Command to SSH into the KVM node"
  value       = var.create_kvm_node ? "ssh cc@${openstack_networking_floatingip_v2.fip_kvm_group15_project[0].address}" : "KVM node not created"
}

output "mlflow_ui_url" {
  description = "URL for the MLflow UI"
  value       = var.create_kvm_node ? "http://${openstack_networking_floatingip_v2.fip_kvm_group15_project[0].address}:5000" : "KVM node not created"
}

output "minio_console_url" {
  description = "URL for the MinIO Console"
  value       = var.create_kvm_node ? "http://${openstack_networking_floatingip_v2.fip_kvm_group15_project[0].address}:9001" : "KVM node not created"
}

output "minio_api_endpoint" {
  description = "API Endpoint for MinIO (S3 compatible)"
  value       = var.create_kvm_node ? "http://${openstack_networking_floatingip_v2.fip_kvm_group15_project[0].address}:9000" : "KVM node not created"
}

output "minio_access_key" {
  description = "MinIO Root Access Key"
  value       = var.create_kvm_node ? var.minio_root_user : "KVM node not created"
  sensitive   = true
}

output "minio_secret_key" {
  description = "MinIO Root Secret Key"
  value       = var.create_kvm_node ? var.minio_root_password : "KVM node not created"
  sensitive   = true
}

output "postgres_connection_string_internal" {
  description = "PostgreSQL connection string for internal use (e.g., from MLflow container)"
  value       = var.create_kvm_node ? "postgresql://${var.postgres_user}:${var.postgres_password}@postgres:5432/${var.postgres_db}" : "KVM node not created"
  sensitive   = true
}

output "postgres_connection_string_external" {
  description = "PostgreSQL connection string for external use (if needed, requires SG rule update)"
  value       = var.create_kvm_node ? "postgresql://${var.postgres_user}:${var.postgres_password}@${openstack_networking_floatingip_v2.fip_kvm_group15_project[0].address}:5432/${var.postgres_db}" : "KVM node not created"
  sensitive   = true
}

output "gpu_node_floating_ip" {
  description = "Floating IP address of the GPU node (CHI@TACC)"
  value       = var.create_gpu_node ? openstack_networking_floatingip_v2.fip_gpu_group15_project[0].address : "GPU node not created"
}

output "ssh_command_gpu" {
  description = "Command to SSH into the GPU node (CHI@TACC)"
  value       = var.create_gpu_node ? "ssh cc@${openstack_networking_floatingip_v2.fip_gpu_group15_project[0].address}" : "GPU node not created"
}

output "jupyter_lab_url" {
  description = "URL for Jupyter Lab"
  # Use try() in case the FIP resource wasn't created (if create_gpu_node=false)
  value       = try("http://${openstack_networking_floatingip_v2.fip_gpu_group15_project[0].address}:8888", "GPU node not created or FIP unavailable")
}

output "ray_dashboard_url" {
  description = "URL for the Ray Dashboard"
  value       = try("http://${openstack_networking_floatingip_v2.fip_gpu_group15_project[0].address}:8265", "GPU node not created or FIP unavailable")
}