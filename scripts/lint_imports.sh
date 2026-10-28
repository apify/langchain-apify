#!/bin/bash

set -eu

# Initialize a variable to keep track of errors
errors=0

# make sure the package does not import from langchain, langchain_experimental, or langchain_community;
# examples and docs may, since they build agents on top of it
git --no-pager grep '^from langchain\.' -- langchain_apify/ && errors=$((errors+1))
git --no-pager grep '^from langchain_experimental\.' -- langchain_apify/ && errors=$((errors+1))
git --no-pager grep '^from langchain_community\.' -- langchain_apify/ && errors=$((errors+1))

# Decide on an exit status based on the errors
if [ "$errors" -gt 0 ]; then
    exit 1
else
    exit 0
fi
