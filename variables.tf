variable "project_id" {
  description = "Google Cloud Project ID."
  type        = string
  default     = "world-learning-400909"
}

variable "region" {
  description = "The region where resources will be deployed."
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "The name of the Google Cloud Storage bucket."
  type        = string
  default     = "test-az-test"
}

variable "app_version" {
  description = "The version of the application."
  type        = string
  default     = "v4"
}

variable "service_name" {
  description = "The name of the App Engine service."
  type        = string
  default     = "wb-citations-main-service"  
}

variable "bucket_object_name" {
  description = "The name of the Google Cloud Storage bucket object."
  type        = string
  default     = "wb-citations-main.zip"
}

variable "source_path" {
  description = "The local path to the source code."
  type        = string
  default     = "./wb-citations-main.zip"
}

variable "entrypoint_shell" {
  description = "The shell command for the App Engine entrypoint."
  type        = string
  default     = "gunicorn -w 2 -b 0.0.0.0:8080 main:app"
}

variable "environment_port" {
  description = "The port to be used in the environment variables."
  type        = string
  default     = "8080"
}

variable "runtime" {
  description = "The runtime for the App Engine."
  type        = string
  default     = "python39"
}

variable "automatic_scaling_min_instances" {
  description = "The minimum number of instances for automatic scaling."
  type        = number
  default     = 1
}

variable "automatic_scaling_max_instances" {
  description = "The maximum number of instances for automatic scaling."
  type        = number
  default     = 10
}
