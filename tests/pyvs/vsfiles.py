import os
from os import path as osp
import shutil

WC = "*"
REPO_DIR = osp.dirname(osp.dirname(osp.realpath(__file__)))

def extend_path(path):
    """
    List files (not directory) matching the given wildcard (only one *
    supported)
    """
    nb = path.count(WC)
    if nb == 0:
        if osp.isfile(path):
            return [path]
        else:
            return []
    elif nb == 1:
        base, wildcard = osp.split(path)
        prefix, suffix = wildcard.split(WC)
        res = []
        for f in os.listdir(base):
            if f.startswith(prefix) and f.endswith(suffix) \
                                    and osp.isfile(osp.join(base, f)):
                res.append(osp.join(base, f))
        return res
    else:
        raise Exception("multiple wildcards not supported")

def remove_if_exist(path):
    """Only for files, not folders"""
    if osp.isfile(path):
        os.remove(path)

def clean_folder(path):
    if osp.isdir(path):
        if osp.islink(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
        create_folder(path)

def soft_clean_folder(path):
    if osp.isdir(path):
        for f in extend_path(osp.join(path, "*")):
            remove_if_exist(f)

def reset_to_empty_folder(path):
    if folder_exists(path):
        if osp.isdir(path):
            clean_folder(path)
        elif osp.isfile(path):
            os.remove(path)
            create_folder(path)
        else:
            raise Exception("I don't know what I'm doing here")
    else:
        create_folder(path)

def folder_exists(path):
    return osp.exists(path)

def create_folder(path):
    os.makedirs(path)

def remove_folder(path):
    if folder_exists(path):
        shutil.rmtree(path)

def tail_lines(filename,linesback=10,returnlist=0):
    """Does what "tail -10 filename" would have done
       Parameters:
            filename   file to read
            linesback  Number of lines to read from end of file
            returnlist Return a list containing the lines instead of a string

    """
    avgcharsperline = 75.

    with open(filename,'r') as f:
        while True:
            try:
                f.seek(-1 * avgcharsperline * linesback, 2)
            except IOError:
                f.seek(0)
            if f.tell() == 0:
                atstart = 1
            else:
                atstart = 0

            lines = f.read().split("\n")
            if (len(lines) > (linesback+1)) or atstart:
                break
            #The lines are bigger than we thought
            avgcharsperline = avgcharsperline * 1.3 #Inc avg for retry

    if len(lines) > linesback:
        start = len(lines) - linesback - 1
    else:
        start = 0
    if returnlist:
        return lines[start:len(lines) - 1]

    out = ""
    for l in lines[start:len(lines) - 1]:
        out=out + l + "\n"
    return out

