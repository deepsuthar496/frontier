#!/bin/bash
# Restore work from Drive to a fresh Colab session.
# Usage (fresh runtime): bash "/content/drive/MyDrive/frontier/restore_from_drive.sh"
# Clones/copies Drive backup -> /content/frontier
set -e
DST="/content/frontier/"
SRC="/content/drive/MyDrive/frontier/"
mkdir -p "$DST"
rsync -a --exclude='__pycache__' --exclude='*.pyc' "$SRC" "$DST"
echo "restored -> $DST ($(du -sh "$DST" | cut -f1))"
cd "$DST" && git status --short | head -n 20 || true
