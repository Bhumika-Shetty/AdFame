# KVM@TACC Ubuntu image
data "openstack_images_image_v2" "ubuntu" {
  provider     = openstack.kvm
  name         = "CC-Ubuntu22.04"
  most_recent  = true
}

# Shared network and subnet
data "openstack_networking_network_v2" "sharednet1" {
  provider = openstack.kvm
  name     = "sharednet1"
}

data "openstack_networking_subnet_v2" "sharednet1_subnet" {
  provider   = openstack.kvm
  name       = "sharednet1-subnet"
  network_id = data.openstack_networking_network_v2.sharednet1.id
}

data "openstack_networking_secgroup_v2" "default" {
  provider = openstack.kvm
  name     = "default"
}