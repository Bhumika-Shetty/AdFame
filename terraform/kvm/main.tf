# --- KVM Node Security Group (MLflow, SSH) ---

resource "openstack_networking_secgroup_v2" "sg_kvm_project15_project" {
  count       = var.create_kvm_node ? 1 : 0
  name        = "sg-kvm-project15-project"
  description = "Allow SSH, MLflow, MinIO, Postgres, Prometheus, Grafana for KVM node project15 project"
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_kvm_ssh" {
  count             = var.create_kvm_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_kvm_project15_project[0].id
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_kvm_mlflow" {
  count             = var.create_kvm_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 5000 # MLflow UI
  port_range_max    = 5000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_kvm_project15_project[0].id
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_kvm_prometheus" {
  count             = var.create_kvm_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 9090 # Prometheus
  port_range_max    = 9090
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_kvm_project15_project[0].id
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_kvm_grafana" {
  count             = var.create_kvm_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 3000 # Grafana
  port_range_max    = 3000
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_kvm_project15_project[0].id
}

# --- KVM Node Resources (KVM@TACC) ---

# Persistent Block Storage Volume
resource "openstack_blockstorage_volume_v3" "volume_kvm_project15_project" {
  count = var.create_kvm_node ? 1 : 0
  name  = "volume-kvm-project15-project"
  size  = var.kvm_volume_size
  description = "Persistent data volume for KVM node project15 project"
}

# KVM Instance
resource "openstack_compute_instance_v2" "node_vm_project15_project" {
  count           = var.create_kvm_node ? 1 : 0
  name            = "node-vm-project15-project"
  image_name      = var.kvm_image_name
  flavor_name     = var.kvm_flavor_name
  key_pair        = var.ssh_key_pair_name
  security_groups = [openstack_networking_secgroup_v2.sg_kvm_project15_project[0].name]

  network {
    name = var.network_name
  }

  depends_on = [openstack_networking_secgroup_v2.sg_kvm_project15_project]
}

# Attach Volume to KVM Instance
resource "openstack_compute_volume_attach_v2" "attach_volume_kvm_project15_project" {
  count       = var.create_kvm_node ? 1 : 0
  instance_id = openstack_compute_instance_v2.node_vm_project15_project[0].id
  volume_id   = var.existing_volume_id

  depends_on = [openstack_compute_instance_v2.node_vm_project15_project]
}

# Floating IP for KVM Node
resource "openstack_networking_floatingip_v2" "fip_kvm_project15_project" {
  count = var.create_kvm_node ? 1 : 0
  pool  = var.external_network_name
}

# Associate Floating IP with KVM Instance
resource "openstack_compute_floatingip_associate_v2" "assoc_fip_kvm_project15_project" {
  count       = var.create_kvm_node ? 1 : 0
  floating_ip = openstack_networking_floatingip_v2.fip_kvm_project15_project[0].address
  instance_id = openstack_compute_instance_v2.node_vm_project15_project[0].id

  # Ensure instance and FIP exist before associating
  depends_on = [
    openstack_compute_instance_v2.node_vm_project15_project,
    openstack_networking_floatingip_v2.fip_kvm_project15_project
  ]
}

# --- Provision KVM Node ---
resource "null_resource" "provision_kvm_project15_project" {
  count = var.create_kvm_node ? 1 : 0

  triggers = {
    instance_id = openstack_compute_instance_v2.node_vm_project15_project[0].id
  }

  connection {
    type        = "ssh"
    user        = "cc"
    private_key = var.ssh_private_key_content 
    host        = openstack_networking_floatingip_v2.fip_kvm_project15_project[0].address
    timeout     = "20m"
  }

  depends_on = [
    openstack_compute_volume_attach_v2.attach_volume_kvm_project15_project,
    openstack_compute_floatingip_associate_v2.assoc_fip_kvm_project15_project
  ]

  # Step 1: Copy setup scripts
  provisioner "file" {
    source      = "${path.module}/scripts/install_docker.sh"
    destination = "/tmp/install_docker.sh"
  }
  provisioner "file" {
    source      = "${path.module}/scripts/mount_volume.sh"
    destination = "/tmp/mount_volume.sh"
  }
  # Copy the MLflow entrypoint script
  provisioner "file" {
    source      = "${path.module}/scripts/mlflow_entrypoint.sh"
    destination = "/tmp/mlflow_entrypoint.sh" # Copy to /tmp first
  }

  # Step 2: Copy Docker Compose template file, rendering variables into it
  provisioner "file" {
    content = templatefile("${path.module}/templates/kvm_docker-compose.yml.tftpl", {
      postgres_db         = var.postgres_db
      postgres_user       = var.postgres_user
      postgres_password   = var.postgres_password
      minio_root_user     = var.minio_root_user
      minio_root_password = var.minio_root_password
      grafana_admin_user  = var.grafana_admin_user
      grafana_admin_password = var.grafana_admin_password
      airflow_username    = var.airflow_username
      airflow_password    = var.airflow_password
    })
    destination = "/tmp/docker-compose.yml"
  }

  # Step 3: Execute setup scripts and run Docker Compose with enhanced logging
  provisioner "remote-exec" {
    inline = [
      "exec &> /tmp/provision_kvm_node_detailed.log",
      "set -ex",

      "echo '*** Starting KVM provisioning script execution at $(date) ***'",

      "echo 'Waiting for cloud-init to complete...'",
      "cloud-init status --wait",
      "echo 'Cloud-init finished at $(date)'",

      "echo 'Applying system updates (can take time)...'",
      "sudo apt-get update",
      "sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y",
      "echo 'System updates finished at $(date)'",

      "echo 'Checking block storage volume...'",
      "if [ -b /dev/vdb ]; then",
      "  echo 'Block device /dev/vdb exists'",
      "  # Check if device is already mounted",
      "  if mount | grep -q '/dev/vdb'; then",
      "    echo 'Device is already mounted'",
      "  else",
      "    # Check if device has a filesystem",
      "    if blkid /dev/vdb; then",
      "      echo 'Device has existing filesystem'",
      "    else",
      "      echo 'Device has no filesystem, formatting...'",
      "      sudo mkfs -t ext4 /dev/vdb",
      "    fi",
      "    echo 'Creating mount point and mounting...'",
      "    sudo mkdir -p /mnt/block",
      "    sudo mount /dev/vdb /mnt/block",
      "    # Only add to fstab if not already there",
      "    if ! grep -q '/dev/vdb /mnt/block' /etc/fstab; then",
      "      echo '/dev/vdb /mnt/block ext4 defaults 0 0' | sudo tee -a /etc/fstab",
      "    fi",
      "  fi",
      "  echo 'Verifying mount:'",
      "  df -h /mnt/block",
      "  mount | grep /mnt/block",
      "else",
      "  echo 'ERROR: Block device /dev/vdb not found!'",
      "  exit 1",
      "fi",

      "echo 'Creating service directories on mounted volume...'",
      "sudo mkdir -p /mnt/block/mlflow/postgres-db",
      "sudo mkdir -p /mnt/block/mlflow/mlflow-artifacts",
      "sudo mkdir -p /mnt/block/minio/data",
      "sudo mkdir -p /mnt/block/prometheus/data",
      "sudo mkdir -p /mnt/block/prometheus/config",
      "sudo mkdir -p /mnt/block/grafana/data",
      "sudo mkdir -p /mnt/block/airflow/dags",
      "sudo mkdir -p /mnt/block/airflow/logs",
      "sudo mkdir -p /mnt/block/airflow/plugins",
      
      "echo 'Setting up SSH key for git operations...'",
      "mkdir -p /home/cc/.ssh",
      "echo '${var.ssh_private_key_content}' > /home/cc/.ssh/id_rsa",
      "chmod 600 /home/cc/.ssh/id_rsa",
      "ssh-keyscan github.com >> /home/cc/.ssh/known_hosts",
      
      "echo 'Cloning repository...'",
      "cd /home/cc",
      "rm -rf mlops-project",
      "git clone git@github.com:meghrathod/mlops-project.git",
      "cp -r /home/cc/mlops-project/terraform/dags/* /mnt/block/airflow/dags/",
      
      # Only set permissions if directories are empty
      "if [ -z \"$(ls -A /mnt/block/mlflow/postgres-db)\" ]; then",
      "  echo 'Setting initial permissions for postgres-db'",
      "  sudo chown -R 999:999 /mnt/block/mlflow/postgres-db",  # 999 is postgres user in container
      "fi",
      
      "if [ -z \"$(ls -A /mnt/block/minio/data)\" ]; then",
      "  echo 'Setting initial permissions for minio data'",
      "  sudo chown -R 1000:1000 /mnt/block/minio/data",  # 1000 is minio user in container
      "fi",
      
      "if [ -z \"$(ls -A /mnt/block/mlflow/mlflow-artifacts)\" ]; then",
      "  echo 'Setting initial permissions for mlflow artifacts'",
      "  sudo chown -R $USER:$USER /mnt/block/mlflow/mlflow-artifacts",
      "fi",

      "if [ -z \"$(ls -A /mnt/block/prometheus/data)\" ]; then",
      "  echo 'Setting initial permissions for prometheus data'",
      "  sudo chown -R 65534:65534 /mnt/block/prometheus/data",  # 65534 is nobody user in container
      "fi",

      "if [ -z \"$(ls -A /mnt/block/grafana/data)\" ]; then",
      "  echo 'Setting initial permissions for grafana data'",
      "  sudo chown -R 472:472 /mnt/block/grafana/data",  # 472 is grafana user in container
      "fi",

      "if [ -z \"$(ls -A /mnt/block/airflow/dags)\" ]; then",
      "  echo 'Setting initial permissions for airflow dags'",
      "  sudo chown -R $USER:$USER /mnt/block/airflow/dags",
      "fi",

      "sudo chmod +x /tmp/install_docker.sh",
      "sudo /tmp/install_docker.sh",
      "echo 'Docker installation finished at $(date)'",

      "echo 'Adding user cc to docker group...' ",
      "sudo usermod -aG docker $USER",
      "echo 'User added to docker group at $(date)'",

      "echo '--- Checking /tmp/docker-compose.yml (copied by file provisioner) BEFORE move ---'",
      "if [ -f /tmp/docker-compose.yml ]; then",
      "  echo '/tmp/docker-compose.yml exists. Size: $(stat -c %s /tmp/docker-compose.yml) bytes.'",
      "  echo 'Head:'",
      "  head -n 10 /tmp/docker-compose.yml",
      "  echo 'Tail:'",
      "  tail -n 10 /tmp/docker-compose.yml",
      "else",
      "  echo 'ERROR: /tmp/docker-compose.yml NOT FOUND!'",
      "  exit 1",
      "fi",
      "echo '--- Finished checking /tmp/docker-compose.yml ---'",

      "echo 'Copying Docker Compose file and entrypoint script...'",
      "mkdir -p /home/cc/mlops_kvm", # Ensure directory exists
      # Move Compose file
      "mv /tmp/docker-compose.yml /home/cc/mlops_kvm/docker-compose.yml",
      # Move the entrypoint script and make it executable
      "mv /tmp/mlflow_entrypoint.sh /home/cc/mlops_kvm/mlflow_entrypoint.sh",
      "chmod +x /home/cc/mlops_kvm/mlflow_entrypoint.sh",

      "cd /home/cc/mlops_kvm",
      "echo 'Files moved at $(date). Current dir: $(pwd)'",

      "echo '--- Checking docker-compose.yml in $(pwd) AFTER move ---'",
      "if [ -f docker-compose.yml ]; then",
      "  echo 'docker-compose.yml exists. Size: $(stat -c %s docker-compose.yml) bytes.'",
      "  echo 'Head:'",
      "  head -n 10 docker-compose.yml",
      "else",
      "  echo 'ERROR: docker-compose.yml NOT FOUND after move!'",
      "  exit 1",
      "fi",
      "echo '--- Finished checking docker-compose.yml after move ---'",

      "echo '*** Starting Docker Compose execution within newgrp docker... at $(date) ***'",
      "newgrp docker << END",
      "  set -ex",
      "  echo '--- [newgrp] Current user/group: $(id) ---'",
      "  echo '--- [newgrp] Docker socket permissions: $(ls -l /var/run/docker.sock) ---'",
      "  echo '--- [newgrp] Current dir: $(pwd) ---'",
      "  echo '--- [newgrp] Running docker compose pull at $(date) ---'",
      "  docker compose pull",
      "  echo '--- [newgrp] Finished docker compose pull at $(date) ---'",
      "  echo '--- [newgrp] Running docker compose up -d at $(date) ---'",
      "  docker compose up -d",
      "  echo '--- [newgrp] Finished docker compose up -d command execution at $(date) ---'",
      "  echo '--- [newgrp] Listing containers (docker ps -a) after compose up ---'",
      "  docker ps -a",
      "END",

      "echo '*** Finished Docker Compose execution block at $(date) ***'",

      "echo 'Sleeping for 30 seconds before final checks...'",
      "sleep 30",

      "echo '--- Final container status check (docker ps -a) at $(date) ---'",
      "sudo docker ps -a",

      "echo '--- Checking logs for postgres (last 50 lines) ---'",
      "docker logs postgres_project15_project --tail 50 || echo '[Error getting postgres logs]'",
      "echo '--- Checking logs for minio (last 50 lines) ---'",
      "docker logs minio_project15_project --tail 50 || echo '[Error getting minio logs]'",
      "echo '--- Checking logs for mlflow (last 50 lines) ---'",
      "docker logs mlflow_project15_project --tail 50 || echo '[Error getting mlflow logs]'",
      "echo '--- Checking logs for airflow-webserver (last 50 lines) ---'",
      "docker logs airflow-webserver --tail 50 || echo '[Error getting airflow-webserver logs]'",
      "echo '--- Checking logs for airflow-scheduler (last 50 lines) ---'",
      "docker logs airflow-scheduler --tail 50 || echo '[Error getting airflow-scheduler logs]'",

      "echo '*** KVM node provisioning script seemingly complete at $(date). ***'",
    ]
  }
}

# ================================================================
# --- GPU Node Resources (CHI@TACC) - Conditionally Created ---
# ================================================================

# --- Security Group for GPU Node (Ray, Jupyter, SSH) ---
resource "openstack_networking_secgroup_v2" "sg_gpu_project15_project" {
  provider    = openstack.tacc # Use CHI@TACC provider
  count       = var.create_gpu_node ? 1 : 0
  name        = "sg-gpu-project15-project"
  description = "Allow SSH, Ray, Jupyter for project15 project GPU node"
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_gpu_ssh" {
  provider          = openstack.tacc
  count             = var.create_gpu_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_gpu_project15_project[0].id
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_gpu_jupyter" {
  provider          = openstack.tacc
  count             = var.create_gpu_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8888 # Jupyter Lab
  port_range_max    = 8888
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_gpu_project15_project[0].id
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_gpu_ray_dash" {
  provider          = openstack.tacc
  count             = var.create_gpu_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 8265 # Ray Dashboard
  port_range_max    = 8265
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_gpu_project15_project[0].id
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_gpu_ray_client" {
  provider          = openstack.tacc
  count             = var.create_gpu_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 10001 # Ray Client
  port_range_max    = 10001
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_gpu_project15_project[0].id
}

resource "openstack_networking_secgroup_rule_v2" "sg_rule_gpu_ray_redis" {
  provider          = openstack.tacc
  count             = var.create_gpu_node ? 1 : 0
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 6379 # Ray GCS (Redis)
  port_range_max    = 6379
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.sg_gpu_project15_project[0].id
}

# --- GPU Node Instance ---
resource "openstack_compute_instance_v2" "node_gpu_project15_project" {
  provider        = openstack.tacc # Use CHI@TACC provider
  count           = var.create_gpu_node ? 1 : 0
  name            = "node-gpu-project15-project"
  image_name      = var.gpu_image_name       # Ensure this is a CUDA-enabled image
  flavor_name     = var.gpu_flavor_name      # Ensure this is a baremetal GPU flavor
  key_pair        = var.ssh_key_pair_name
  security_groups = [openstack_networking_secgroup_v2.sg_gpu_project15_project[0].name]

  network {
    name = var.network_name
  }

  scheduler_hints {
      additional_properties = {
      "reservation" = "4118d33f-ac3d-4870-80a9-19c44b52e6ba"
    }
  }

  # Bare metal nodes might take longer to provision
  timeouts {
    create = "45m" # Increased timeout for bare metal
    delete = "15m"
  }

  depends_on = [openstack_networking_secgroup_v2.sg_gpu_project15_project]
}

# --- Floating IP for GPU Node ---
resource "openstack_networking_floatingip_v2" "fip_gpu_project15_project" {
  provider = openstack.tacc # Use CHI@TACC provider
  count    = var.create_gpu_node ? 1 : 0
  pool     = var.external_network_name
}

# --- Associate Floating IP with GPU Instance ---
resource "openstack_compute_floatingip_associate_v2" "assoc_gpu_project15_project" {
  provider    = openstack.tacc # Use CHI@TACC provider
  count       = var.create_gpu_node ? 1 : 0
  floating_ip = openstack_networking_floatingip_v2.fip_gpu_project15_project[0].address
  instance_id = openstack_compute_instance_v2.node_gpu_project15_project[0].id

  depends_on = [
     openstack_compute_instance_v2.node_gpu_project15_project,
     openstack_networking_floatingip_v2.fip_gpu_project15_project
  ]
}

# --- Provision GPU Node (Install Docker + NVIDIA Toolkit) ---
# Note: We are NOT deploying Ray/Jupyter yet in this step.
resource "null_resource" "provision_gpu_base_project15_project" {
  # Rename to avoid conflict if we add a later provisioner for compose
  count    = var.create_gpu_node ? 1 : 0

  triggers = {
    instance_id = openstack_compute_instance_v2.node_gpu_project15_project[0].id
  }

  connection {
    type        = "ssh"
    user        = "cc"
    private_key = var.ssh_private_key_content
    host        = openstack_networking_floatingip_v2.fip_gpu_project15_project[0].address
    timeout     = "20m" # Increased timeout for installation
  }

  depends_on = [
    openstack_compute_floatingip_associate_v2.assoc_gpu_project15_project
  ]

  # Step 1: Copy GPU Docker setup script
  provisioner "file" {
    source      = "${path.module}/scripts/install_docker_gpu.sh"
    destination = "/tmp/install_docker_gpu.sh"
  }

  # Step 2: Execute base setup
  provisioner "remote-exec" {
    inline = [
      "exec &> /tmp/provision_gpu_base_detailed.log", # Log output
      "set -ex",

      "echo '*** Starting GPU Base provisioning script execution at $(date) ***'",
      "echo 'Waiting for cloud-init to complete...'",
      "cloud-init status --wait",

      "echo 'Applying system updates...'",
      "sudo apt-get update",
      "sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y",

      "echo 'Making install script executable...'",
      "sudo chmod +x /tmp/install_docker_gpu.sh",

      "echo 'Running Docker + NVIDIA Toolkit installation script...'",
      "sudo /tmp/install_docker_gpu.sh",

      "echo 'Adding cc user to docker group...'",
      "sudo usermod -aG docker $USER",

      "echo '*** GPU Base provisioning complete at $(date). ***'",
      "echo 'Run docker info | grep -i runtime manually to verify nvidia runtime.'",
    ]
  }
}

# --- Provision GPU Node (Deploy Ray/Jupyter Compose) ---
resource "null_resource" "provision_gpu_compose_project15_project" {
  count    = var.create_gpu_node ? 1 : 0 # Only run if GPU node is created

  triggers = {
    base_provisioner_id = null_resource.provision_gpu_base_project15_project[0].id
    kvm_fip             = var.create_kvm_node ? openstack_networking_floatingip_v2.fip_kvm_project15_project[0].address : "kvm-node-disabled"
  }

  # Connection block for remote-exec
  connection {
    type        = "ssh"
    user        = "cc"
    private_key = var.ssh_private_key_content # Use the content directly
    host        = openstack_networking_floatingip_v2.fip_gpu_project15_project[0].address
    timeout     = "15m"
  }

  depends_on = [
    null_resource.provision_gpu_base_project15_project,
  ]

  # Step 1 (was Step 2): Copy Docker Compose template file, rendering variables
  provisioner "file" {
    # Pass the HASHED password variable directly to the template
    content = templatefile("${path.module}/templates/gpu_docker-compose.yml.tftpl", {
      # Pass KVM node details if KVM node is enabled
      mlflow_tracking_uri = var.create_kvm_node ? "http://${openstack_networking_floatingip_v2.fip_kvm_project15_project[0].address}:5000" : ""
      minio_endpoint_url  = var.create_kvm_node ? "http://${openstack_networking_floatingip_v2.fip_kvm_project15_project[0].address}:9000" : ""
      minio_access_key    = var.create_kvm_node ? var.minio_root_user : ""
      minio_secret_key    = var.create_kvm_node ? var.minio_root_password : ""
    })
    destination = "/tmp/gpu-docker-compose.yml"
  }


  # Step 2 (was Step 4): Execute setup and run Docker Compose on GPU node
  provisioner "remote-exec" {
    # This uses the 'connection' block defined above
    inline = [
      "exec &> /tmp/provision_gpu_compose_detailed.log", # Log output
      "set -ex",

      "echo '*** Starting GPU Compose provisioning script execution at $(date) ***'",

      "echo 'Ensuring directories exist for services...'",
      "mkdir -p /home/cc/mlops_gpu/jupyter_work",
      "sudo chown -R $USER:$USER /home/cc/mlops_gpu",
      "sudo chmod -R 775 /home/cc/mlops_gpu",

      "echo 'Copying Docker Compose file (overwrites existing)...'",
      "mv /tmp/gpu-docker-compose.yml /home/cc/mlops_gpu/docker-compose.yml",
      "cd /home/cc/mlops_gpu",
      "echo 'Compose file placed at $(date). Current dir: $(pwd)'",
      "echo 'Compose file content check (head):'",
      "head -n 20 docker-compose.yml",


      "echo '*** Starting Docker Compose execution (pull + up -d) within newgrp docker... at $(date) ***'",
      "newgrp docker << END",
      "  set -ex",
      "  echo '--- [newgrp] Current dir: $(pwd) ---'",
      "  echo '--- [newgrp] Running docker compose pull at $(date) ---'",
      "  docker compose pull",
      "  echo '--- [newgrp] Finished docker compose pull at $(date) ---'",
      "  echo '--- [newgrp] Running docker compose up -d at $(date) ---'",
      "  docker compose up -d",
      "  echo '--- [newgrp] Finished docker compose up -d command execution at $(date) ---'",
      "  echo '--- [newgrp] Listing containers (docker ps -a) after compose up ---'",
      "  docker ps -a",
      "END",

      "echo '*** Finished Docker Compose execution block at $(date) ***'",

      "echo 'Sleeping for 15 seconds before final checks...'",
      "sleep 15",
      "echo '--- Final container status check (docker ps -a) at $(date) ---'",
      "docker ps -a",

      "echo '--- Checking logs for ray-head (last 50 lines) ---'",
      "docker logs ray_head_gpu_project15_project --tail 50 || echo '[Skipping ray-head logs - container likely stopped/not found]'",
      "echo '--- Checking logs for jupyter (last 50 lines) ---'",
      "docker logs jupyter_gpu_project15_project --tail 50 || echo '[Skipping jupyter logs - container likely stopped/not found]'",

      "echo '*** GPU Compose provisioning script seemingly complete at $(date). ***'",
    ]
  }


}