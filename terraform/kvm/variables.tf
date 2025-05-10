variable "auth_url" {
  default = "https://kvm.tacc.chameleoncloud.org:5000/v3"
}

variable "region" {
  default = "KVM@TACC"
}

variable "application_credential_id" {}
variable "application_credential_secret" {}

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