#!/bin/sh
URL="https://drive.google.com/uc?id=1ongIYCtX7qmknp1_ykWlGzZjJnsWjSUS"
ZIP_FILE="AFHQv2.zip"
gdown "$URL"
echo "Extracting..."
unzip -qq "$ZIP_FILE" -d "./data"
rm "$ZIP_FILE"
echo "Dataset is ready."