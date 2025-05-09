# Configure the OpenStack Provider for KVM@TACC (Default)
provider "openstack" {
  cloud = "kvm-tacc"
}

# Configure the OpenStack Provider for CHI@TACC
provider "openstack" {
  alias = "tacc" # Alias for the CHI@TACC provider
  cloud = "tacc"
}

# Terraform settings
terraform {

  cloud {
    organization = "MLOps-Project" # Replace with your actual organization name
    workspaces {
      name = "mlops-chameleon-prod" # Replace with your actual workspace name
    }
  }

  required_version = ">= 1.0.0"
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "~> 1.48.0"
    }
  }
}   