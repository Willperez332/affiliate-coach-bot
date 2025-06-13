#!/bin/bash

# This script will install dependencies and then start the bot.

echo "--- Checking and installing dependencies ---"
sudo apt-get update
sudo apt-get install -y ffmpeg

echo "--- Starting the bot ---"
python3 bot.py