terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "5.14.0"
    }
  }

  backend "gcs" {
    bucket = "test-az-test"
     prefix = "ts-appengine-terraform/state"
     }
}

provider "google" {
  project = var.project_id
  region  = var.region
}


resource "google_storage_bucket" "terraform_state" {
  name          = var.bucket_name
  force_destroy = false
  location      = "US"
  storage_class = "STANDARD"
  versioning {
    enabled = true
  }
}

resource "google_storage_bucket_object" "object" {
  name   = var.bucket_object_name
   bucket = google_storage_bucket.terraform_state.name
   source = var.source_path
 }

resource "google_app_engine_standard_app_version" "myapp_v1" {
  version_id = var.app_version
  service    = var.service_name
  runtime    = var.runtime

  deployment {
    zip {
      source_url = "https://storage.googleapis.com/test-az-test/wb-citations-main.zip"  # Replace with your GitHub repository URL
    }
  }

  entrypoint {
    
      shell  = var.entrypoint_shell
  }

  env_variables = {
    PORT = var.environment_port
  }

  automatic_scaling {
    max_concurrent_requests = 10
    min_idle_instances = 1
    max_idle_instances = 3
    min_pending_latency = "1s"
    max_pending_latency = "5s"
    standard_scheduler_settings {
      target_cpu_utilization = 0.5
      target_throughput_utilization = 0.75
      min_instances = var.automatic_scaling_min_instances
      max_instances = var.automatic_scaling_max_instances
    }
  }

  delete_service_on_destroy = true
}
