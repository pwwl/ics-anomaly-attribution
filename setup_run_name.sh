# Simple script that prepares a new run_name, used across all python scripts
R=$'\r'
testVar=${testVar%$R}    # removes pesky $\r following arg1
mkdir models/$1