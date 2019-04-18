#!/usr/bin/env python

import os
from os import path as osp
import xml.etree.ElementTree as ET

DIR = "xml"
OUTPUT = "report.xml"

def main():
    if osp.isfile(OUTPUT):
        os.remove(OUTPUT)
    xml_files = os.listdir(DIR)
    xml_files = [osp.join(DIR, f) for f in xml_files if f.endswith(".xml")]
    # take the first file as base
    tree_base = ET.parse(xml_files[0])
    root_base = tree_base.getroot()
    os.remove(xml_files[0])
    # iterate over other files and extract errors
    for xml_file in xml_files[1:]:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for error in root.findall("errors")[0].findall("error"):
            root_base.findall("errors")[0].append(error)
        os.remove(xml_file)

    tree_base.write(OUTPUT)

if __name__ == '__main__':
    main()

