terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = ">= 3.0.0"
    }
  }
}

provider "openstack" {
  auth_url                    = var.auth_url
  region                      = var.region
  application_credential_id   = var.application_credential_id
  application_credential_secret = var.application_credential_secret
}

# Pull network info
data "openstack_networking_network_v2" "network" {
  name = var.network_name
}

# Keypair
resource "openstack_compute_keypair_v2" "keypair" {
  name       = var.keypair_name
  public_key = file(var.public_key_path)
}

# Create dedicated port for controller
resource "openstack_networking_port_v2" "controller_port" {
  name       = "controller-port-group15"
  network_id = data.openstack_networking_network_v2.network.id
}

# Floating IP associated with port
resource "openstack_networking_floatingip_v2" "controller_fip" {
  pool    = "public"
  port_id = openstack_networking_port_v2.controller_port.id
}

# Controller node
resource "openstack_compute_instance_v2" "controller" {
  name            = "controller-group15"
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  key_pair        = openstack_compute_keypair_v2.keypair.name
  security_groups = ["default"]

  network {
    port = openstack_networking_port_v2.controller_port.id
  }
}

# Worker nodes (no floating IP)
resource "openstack_compute_instance_v2" "worker" {
  count           = 2
  name            = "worker${count.index}-group15"
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  key_pair        = openstack_compute_keypair_v2.keypair.name
  security_groups = ["default"]

  network {
    name = var.network_name
  }
}