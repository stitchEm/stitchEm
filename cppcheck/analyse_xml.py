#!/usr/bin/env python

import os
import sys
import xml.etree.ElementTree as ET

OUTPUT = "report.xml"

def main():
    tree = ET.parse(OUTPUT)
    root = tree.getroot()
    count = len(root.findall("errors")[0].findall("error"))
    if count:
        code = 1
        print "found {} errors".format(count)
    else:
        code = 0
    os.remove(OUTPUT)
    sys.exit(code)

if __name__ == '__main__':
    main()

