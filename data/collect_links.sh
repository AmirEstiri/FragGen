#!/bin/sh
# Run this bash to collect links for fragrances

while true
do
    python data/scraper.py
    echo "Run finished!"
    sleep 600
done