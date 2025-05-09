variable "create_kvm_node" {
  description = "Set to true to provision the KVM node at KVM@TACC"
  type        = bool
  default     = true
}

variable "create_gpu_node" {
  description = "Set to true to provision the node at CHI@TACC"
  type        = bool
  default     = true
}

variable "ssh_key_pair_name" {
  description = "Name of the SSH key pair registered in OpenStack (must exist in both regions)"
  type        = string
  default     = "mlops-key-15"
}

variable "kvm_image_name" {
  description = "Image name for the KVM node (KVM@TACC)"
  type        = string
  default     = "CC-Ubuntu22.04"
}

variable "kvm_flavor_name" {
  description = "Flavor for the KVM node (KVM@TACC)"
  type        = string
  default     = "m1.large" 
}

variable "gpu_image_name" {
  description = "Image name for the node at CHI@TACC"
  type        = string
  default     = "CC-Ubuntu22.04"
}

variable "gpu_flavor_name" {
  description = "Flavor for the node at CHI@TACC (Use m1.large for now, will change to GPU bare metal later)"
  type        = string
  default     = "baremetal"
}

variable "gpu_lease_id" {
  description = "Lease ID for a baremetal at CHI site"
  type        = string
  default     = "5626cb8b-3e92-4fbc-9594-3808d1809e30"
}

variable "network_name" {
  description = "Name of the network to attach instances to"
  type        = string
  default     = "sharednet1"
}

variable "external_network_name" {
  description = "Name of the external network for Floating IPs"
  type        = string
  default     = "public"
}

variable "ssh_private_key_content" {
  description = "Content of the private SSH key for connecting to instances."
  type        = string
  sensitive   = true
}


variable "kvm_volume_size" {
  description = "Size of the persistent block storage volume in GB for the KVM node"
  type        = number
  default     = 50 # Adjust size as needed
}

# --- KVM Node Service Credentials (MLflow/MinIO) ---
variable "minio_root_user" {
  description = "Root username for MinIO"
  type        = string
  sensitive   = true
}

variable "minio_root_password" {
  description = "Root password for MinIO"
  type        = string
  sensitive   = true
}