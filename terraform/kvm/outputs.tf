output "serving_floating_ip" {
  value = openstack_networking_floatingip_v2.serving_fip.address
}

output "training_node_name" {
  value = openstack_compute_instance_v2.training_node.name
}

output "serving_node_name" {
  value = openstack_compute_instance_v2.serving_node.name
}