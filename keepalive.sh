#!/bin/bash
while true; do
    echo "Last updated: $(date)" > keepalive.txt
    sleep 60
done
