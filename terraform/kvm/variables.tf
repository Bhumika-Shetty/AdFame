variable "key" {
  description = "SSH key pair name"
  type        = string
}

variable "suffix" {
  description = "Suffix for naming resources"
  type        = string
}

variable "nodes" {
  description = "Map of node roles to flavor types"
  type = map(object({
    flavor = string
  }))
}

variable "training_volume_id" {
  description = "Volume ID to attach to training node"
  type        = string
}