#!/bin/bash
# Save progress from fast local SSD (/content/frontier) to persistent Drive.
# Usage: bash save_to_drive.sh [--delete]
set -e
SRC="/content/frontier/"
DST="/content/drive/MyDrive/frontier/"
mkdir -p "$DST"
if [ "$1" == "--delete" ]; then
  rsync -a --delete --exclude='__pycache__' --exclude='*.pyc' "$SRC" "$DST"
else
  rsync -a --exclude='__pycache__' --exclude='*.pyc' "$SRC" "$DST"
fi
echo "saved -> $DST ($(du -sh "$DST" | cut -f1))"
