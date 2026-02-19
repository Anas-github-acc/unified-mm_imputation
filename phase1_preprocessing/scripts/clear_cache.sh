#!/bin/bash
# Helper script to clear Linux system cache
# Run with: sudo ./clear_cache.sh
# or: watch -n 60 sudo ./clear_cache.sh (clears cache every 60 seconds)

echo "=== Clearing System Cache ==="
echo "Before:"
free -h | grep -E "Mem|Swap"

echo ""
echo "Syncing and dropping caches (1=pagecache, 2=dentries/inodes, 3=both)..."
sync
echo 3 > /proc/sys/vm/drop_caches

echo ""
echo "After:"
free -h | grep -E "Mem|Swap"
echo "=== Done ==="
