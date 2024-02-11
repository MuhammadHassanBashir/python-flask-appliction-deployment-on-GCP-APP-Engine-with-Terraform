#!/bin/bash
echo "Step 1: Installing Required Python Packages"
pip install -r ./wb-citations-main/requirements.txt

echo "Step 2: Creating Deployment Archive"
cd wb-citations-main

zip -r wb-citations-main.zip ./*

echo "Archive Created: wb-citations-main.zip"

echo "Step 4: Moving Archive to the Root Directory"
mv wb-citations-main.zip ../.

cd ..

echo "Step 5: Initializing Terraform"
terraform init 

echo "Step 6: Planning the Infrastructure"
terraform plan

echo "Step 7: Applying Terraform Script"
terraform apply -auto-approve

#echo "Cleaning Up"
#rm -rf venv wb-citations-main"


































