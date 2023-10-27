#!/bin/bash

# Function to format sizes
format_size() {
  local size=$1
  if [[ $size -lt 1024 ]]; then
    echo "${size}b"
  elif [[ $size -lt 1048576 ]]; then
    echo $((size / 1024))"k"
  elif [[ $size -lt 1073741824 ]]; then
    echo $((size / 1048576))"M"
  else
    echo $((size / 1073741824))"G"
  fi
}

# Main loop to display sizes
for item in *; do
  if [[ -d $item ]]; then
    size=$(du -sb "$item" | cut -f1)
  else
    size=$(stat -c "%s" "$item")
  fi
  formatted_size=$(format_size $size)
  printf "%5s\t%s\n" "$formatted_size" "$item"
done
