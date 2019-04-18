TOOL=pylint
CONFIG_FILE="pylint.rcfile"
OUTPUT_FILE="pylint_report.html"
DIR="samples/server/*.py"

pylint $DIR --rcfile=$CONFIG_FILE > $OUTPUT_FILE
