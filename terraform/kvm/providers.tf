# Configure the OpenStack Provider for KVM@TACC with alias
provider "openstack" {
  alias = "kvm" 
  cloud = "kvm-tacc"
}

# Configure the OpenStack Provider for CHI@TACC with alias
provider "openstack" {
  alias = "tacc"
  cloud = "tacc"
}

# Terraform settings
terraform {
  required_version = ">= 1.0.0"
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.48.0"
    }
  }
}
