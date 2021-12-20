# args: input file, output file, number of frames
awk_arg='$1+0 <= '$3
sed 's/$/,-1,-1,-1/' $1 | tr -d "[:blank:]" | awk -F, awk_arg > $2