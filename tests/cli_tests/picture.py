#! /usr/bin/env python

from optparse import OptionParser
import os
from os import path as osp

BASE_DIR = osp.dirname(osp.realpath(__file__))
BRANCH_PICS = osp.join(BASE_DIR, "pictures", "branch")
TEMPLATE = osp.join(BASE_DIR, "template.tpl")
RES = osp.join(BASE_DIR, "res.html")

IMG = '<td><a href="{0}"><img src="{0}"></a></td>'

def main():
    usage = "picture.py CURRENT_BRANCH"
    parser = OptionParser(usage)
    (_, args) = parser.parse_args()
    if len(args) != 1:
        parser.error(usage)
    branch = args[0]
    res = "<tr>"
    for pic in os.listdir(BRANCH_PICS):
        tmp = IMG.format("refs/" + pic) + IMG.format("branch/" + pic)
        res += tmp + "</tr><tr>"
    res += "</tr>"
    with open(TEMPLATE, "r") as f:
        output = f.read()
    output = output.replace("##BRANCH##", branch)
    output = output.replace("##PICTURES##", res)
    with open(RES, "w") as f:
        f.write(output)

if __name__ == "__main__":
    main()

