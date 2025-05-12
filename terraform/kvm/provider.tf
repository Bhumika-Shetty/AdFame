provider "openstack" {
  cloud = "chi"
}

provider "openstack" {
  alias = "chi"
  cloud = "chi"
}

provider "openstack" {
  alias = "kvm"
  cloud = "kvm"
}

provider "openstack" {
  alias = "uc"
  cloud = "uc"
}
