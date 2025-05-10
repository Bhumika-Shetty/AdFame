variable "auth_url" {
  default = "https://kvm.tacc.chameleoncloud.org:5000/v3"
}

variable "region" {
  default = "KVM@TACC"
}

variable "application_credential_id" {
  description = "OpenStack application credential ID"
  default     = "d37008c7f5b446c29e03c490790d0e9d"
}

variable "application_credential_secret" {
  description = "OpenStack application credential secret"
  default     = "53Hr7TYYlywzzcXnOWBAKFB394V61zEVfXH_SAsXhThZGzyinuDkHHbM8-ypystvIOPIJdRK90DVfrvLLqsTeg"
}

variable "keypair_name" {}
variable "public_key_path" {}
variable "network_name" {
  default = "private_net_project15"
}
variable "image_name" {
  default = "CC-Ubuntu24.04"
}
variable "flavor_name" {
  default = "m1.large"
}