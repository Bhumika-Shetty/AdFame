# Create private network
resource "openstack_networking_network_v2" "private_network" {
  name           = var.network_name
  admin_state_up = true
}

# Create subnet
resource "openstack_networking_subnet_v2" "private_subnet" {
  name            = "private_subnet_project15"
  network_id      = openstack_networking_network_v2.private_network.id
  cidr            = "192.168.17.0/24"
  ip_version      = 4
  gateway_ip      = "192.168.17.1"
  dns_nameservers = ["8.8.8.8", "1.1.1.1"]
}

# Keypair

# Dedicated port for controller
resource "openstack_networking_port_v2" "controller_port" {
  name       = "controller-port-project15"
  network_id = openstack_networking_network_v2.private_network.id
}

# Floating IP for controller
resource "openstack_networking_floatingip_v2" "controller_fip" {
  pool    = "public"
  port_id = openstack_networking_port_v2.controller_port.id
}

# Controller node
resource "openstack_compute_instance_v2" "controller" {
  name            = "controller-project15"
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  security_groups = ["default"]

  network {
    port = openstack_networking_port_v2.controller_port.id
  }
}

# Worker nodes (on private network)
resource "openstack_compute_instance_v2" "worker" {
  count           = 2
  name            = "worker${count.index}-project15"
  image_name      = var.image_name
  flavor_name     = var.flavor_name
  security_groups = ["default"]

  network {
    uuid = openstack_networking_network_v2.private_network.id
  }
}