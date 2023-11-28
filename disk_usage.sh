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

# Delete trash
trash_size=$(du -sb ".local/share/Trash/" | cut -f1)
formatted_trash_size=$(format_size $trash_size)
rm -rf ".local/share/Trash/"*
echo "Trash deleted ($formatted_trash_size)"

# Initialize total size variable
total_size=0

# Check for directory argument
dir="."
if [ "$#" -eq 1 ]; then
  dir="$1"
fi

# Main loop to display sizes
cd "$dir" || exit 1
for item in .[^.]* *; do
  if [[ -d $item ]]; then
    size=$(du -sb "$item" | cut -f1)
  else
    size=$(stat -c "%s" "$item")
  fi
  total_size=$((total_size + size))
  formatted_size=$(format_size $size)
  printf "%5s\t%s\n" "$formatted_size" "$item"
done

echo "-----"

# Display total size
formatted_total_size=$(format_size $total_size)
printf "%5s\t%s\n" "$formatted_total_size" "Total"
