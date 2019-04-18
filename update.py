#! python

import re
import urllib2
import argparse
import os
import sys
import shutil
import json
import time
import getpass
import platform
from sys import stdin
import ssl

ADDR = "https://bb.video-stitch.com:8010"
DIRECTORY = os.path.dirname(os.path.realpath(__file__))
DEPS_FOLDER = os.path.join(DIRECTORY, "external_deps")
LIB_FOLDER = os.path.join(DEPS_FOLDER, "lib")
LIB64_FOLDER = os.path.join(DEPS_FOLDER, "lib64")
PASS_PATH = os.path.abspath(os.path.join(DIRECTORY, '..', 'pass.py'))
def os_name():
    if platform.system() == "Windows" or\
            "cygwin" in platform.system().lower():
        return "windows"
    elif platform.system() == "Darwin":
        return "mac"
    else:
        return "linux"

def target_name():
    if os_name() != "linux":
        return os_name()
    else:
        return ARGS.target

def stdin_readline_without_cr():
    return sys.stdin.readline()[:-len(os.linesep)]

def ask_for_credentials():
    global LOGIN
    global PASS
    print("Enter your buildbot login : ")
    LOGIN = stdin_readline_without_cr()
    PASS = getpass.getpass()

def ask_boolean(prompt):
    boolstr = ""
    while(boolstr not in ["yes","no"]):
        print(prompt + " (yes or no)")
        boolstr = stdin_readline_without_cr()
        if (boolstr == ""):
            boolstr = "no"
    return boolstr == "yes"

REPOS = ["deps", "visualizer-deps", "visualizer-data"]

PARSER = argparse.ArgumentParser(
    description="Retrieve deps for the application you want to build")
PARSER.add_argument(
    "repo",
    help="Repo you want to pull (" + ",".join(REPOS + ["all)"]),
    )
PARSER.add_argument(
    "-v",
    "--verbose",
    action="store_true",
    help="Verbose mode",
    )

TARGETS = ["linux", "armhf", "aarch64", "android-arm"]
PARSER.add_argument(
    "--target", default="linux",
    help="target platform in case of cross compilation on linux (" + ",".join(TARGETS) + ")",
    )

ARGS = PARSER.parse_args()

def get_credentials():
    if not "LOGIN" in globals() or not "PASS" in globals():
        global LOGIN
        global PASS
        try:
            with open(PASS_PATH,"r") as f:
                line=f.readline()
                while(line):
                    for var in ["LOGIN", "PASS", "PROMPT_LOGIN"]:
                        match=re.match(var + "=\"(.*)\"", line)
                        if match:
                            globals()[var]=match.group(1)
                    line=f.readline()
        except IOError:
            print("File pass.py not found.")
            print("The pass.py file is an helper file where your "
                  "buildbot credentials are stored")
            print("NB: for now those informations are stored clear, "
                  "so anyone can have your password from it")
            print("This script can create it for you right now, if "
                  "you don't want to you will not be prompt anymore")
            print("You can still add it later manually")
            create_pass_py = ask_boolean("Do you want to create the file?")
            ask_for_credentials()
            with open(PASS_PATH,"w") as f:
                if create_pass_py:
                    f.write("LOGIN=\"" + LOGIN + "\"\n")
                    f.write("PASS=\"" + PASS + "\"")
                else:
                    f.write("PROMPT_LOGIN=\"True\"")
        try:
            if (PROMPT_LOGIN):
                if stdin.isatty():
                    ask_for_credentials()
        except Exception:
            pass

BB_DL_URL = "/".join([ADDR, "downloads", target_name()])

def get_default_url(repo_name):
    return "/".join([BB_DL_URL, repo_name])

class DownloadableItem(object):
    def __init__(self, typeh, name, modiftime):
        self.type = typeh
        self.name = name
        self.modiftime = modiftime

def make_auth(url):
    get_credentials()
    passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, ADDR + "/downloads", LOGIN, PASS)
    authhandler = urllib2.HTTPBasicAuthHandler(passman)
    opener = urllib2.build_opener(authhandler)
    urllib2.install_opener(opener)

def open_url(url):
    try:
        make_auth(url)
        if hasattr(ssl, '_create_unverified_context'):
            ssl._create_default_https_context = ssl._create_unverified_context
        urlret = urllib2.urlopen(url, timeout=4)
    except urllib2.HTTPError as err:
        if err.code == 401:
            raise Exception("The login and password you provided "
                            "are not correct")
        else:
            raise
    except urllib2.URLError:
        try:
            urlret = urllib2.urlopen(url)
        except urllib2.URLError:
            raise Exception("An error occured : looks like the "
                            "buildbot is unreachable.")
    return urlret.read()

def download_file(url, file_name="", verbose=False, **kwargs):
    if file_name == "":
        file_name = url.split("/")[-1]
    if verbose:
        print("Starting download of " + file_name)
    file_content = open_url(url)
    if os.path.dirname(file_name) and\
            not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))
    with open(file_name,"wb") as f:
        f.write(file_content)
    os.chmod(file_name, 0o755)
    if verbose:
        print(file_name + " downloaded successfully")

def get_package_url(repo_name, package):
    return "/".join([get_default_url(repo_name), package["name"],
                     package["version"]])

def get_file_and_dir_list(html):
    dilist = []
    for line in html.split("\n"):
        # TODO: rewrite this shit
        #date_pattern = "[0-9]{2}-[A-Z][a-z]{2}-[0-9]{4}"
        date_pattern = "[0-9]{4}-[0-9]{2}-[0-9]{2}"
        match = re.match(".*<a href=\"(.*)\">\\1</a>\\s+(" + date_pattern +\
            " [0-9]{2}:[0-9]{2}).*", line)
        if match:
            item_name = match.group(1)
            #modiftime = time.strptime(match.group(2),"%d-%b-%Y %H:%M")
            modiftime = time.strptime(match.group(2),"%Y-%m-%d %H:%M")
            if (item_name[-1] == "/"):
                t = "dir"
                item_name = item_name[:-1]
            else:
                t = "file"
            dilist.append(DownloadableItem(t,item_name,modiftime))
    return dilist

def string_matches_array(string, arr):
    for s in arr:
        if s in string:
            return True
    return False

def download_dir(url, dest_folder, verbose):
    if dest_folder == "":
        dest_folder = url.split("/")[-1]
    try:
        os.makedirs(dest_folder)
    except Exception:
        pass
    if verbose:
        print("Downloading " + url + " to " + dest_folder)
    try:
        html = open_url(url)
    except TypeError:
        html = open_url(url)
    except Exception as e:
        print("A problem occured while opening " + url)
        print(e)
        html = ""
    dlist = get_file_and_dir_list(html)
    # Download file and dir list
    for ditem in dlist:
        dest = "/".join([dest_folder,ditem.name])
        if ditem.type == "file":
            # Ignore if file hasn't changed
            if os.path.exists(dest):
                if os.stat(dest).st_mtime > time.mktime(ditem.modiftime):
                    continue
        # Call the right download function (file or dir)
        globals()["_".join(["download", ditem.type])]("/".join(
            [url, ditem.name]), dest, verbose)

def download_package(repo_name, package, verbose):
    print("Downloading package " + package["name"] + " from repo " +\
            repo_name)
    url = get_package_url(repo_name, package)
    open_url(url)
    not_static = not "static" in package or not package["static"]
    # on linux put the .so in external_deps/lib, on windows/mac in the build
    # directory
    if os_name() is "linux":
        if not_static:
            download_dir("/".join([url, "bin"]), LIB_FOLDER, verbose)
        download_dir("/".join([url, "lib"]), LIB_FOLDER, verbose)
        if target_name() != "linux":
            download_dir("/".join([url, "lib64"]), LIB64_FOLDER, verbose)
    else:
        if not_static:
            download_dir("/".join([url, "bin"]), "", verbose)
        download_dir("/".join([url, "lib"]), os.path.join(
            DEPS_FOLDER, "lib", package["name"]), verbose)
    download_dir("/".join([url,"include"]), os.path.join(
        DEPS_FOLDER, "include", package["name"]), verbose)
    print("Download of " + package["name"] + " complete.\n")

def download_visualizer_package(repo_name, package, verbose):
    print("Downloading package " + package["name"] + " from repo " +\
            repo_name)
    url = get_package_url(repo_name, package)
    open_url(url)
    if not "static" in package or not package["static"]:
        # on linux put the .so in external_deps/lib, on windows/mac in the
        # build directory
        if os_name() is "linux":
            download_dir("/".join([url,"bin"]), LIB_FOLDER, verbose)
        else:
            download_dir("/".join([url,"bin"]), "", verbose)
    download_dir("/".join([url,"lib"]), os.path.join(
        DEPS_FOLDER, "lib", package["name"]), verbose)
    download_dir("/".join([url,"include"]), os.path.join(
        DEPS_FOLDER, "include", package["name"]), verbose)
    print("Download of " + package["name"] + " complete.\n")

def download_visualizer_data(repo_name, package, verbose):
    print("Downloading package " + package["name"] + " from repo " +\
            repo_name)
    url = get_package_url(repo_name, package)
    open_url(url)
    download_dir(url, os.path.join("apps", "src",
                                   "videostitch-visualizer-projects",
                                   package["name"]), verbose)
    print("Download of " + package["name"] + " complete.\n")

def update_visualizer_deps(verbose):
    with open(os.path.join(DIRECTORY, "visualizer-deps.json"),"r") as f:
        deps = json.loads(f.read())
    if (deps["repo"] != "deps"):
        print("Looks like virualizer-deps for the repo " + deps["repo"] +\
                " are not yet implemented.")
    for package in deps["packages"]:
        try:
            download_visualizer_package(deps["repo"], package, verbose)
        except Exception as e:
            print("A problem occured while downloading " + package["name"] +\
                    " version " + package["version"])
            print(e)

def update_visualizer_data(verbose):
    with open(os.path.join(DIRECTORY, "visualizer-data.json"),"r") as f:
        deps = json.loads(f.read())
    if deps["repo"] != "deps":
        print("Looks like virualizer-deps for the repo " + deps["repo"] +\
                " are not yet implemented.")
    for package in deps["packages"]:
        try:
            download_visualizer_data(deps["repo"], package, verbose)
        except Exception as e:
            print("A problem occured while downloading " + package["name"] +\
                    " version " + package["version"])
            print(e)

def update_deps(verbose):
    try:
        shutil.rmtree(DEPS_FOLDER)
    except OSError:
        pass
    with open(os.path.join(DIRECTORY, "deps.json"),"r") as f:
        deps = json.loads(f.read())
    if (deps["repo"] != "deps"):
        print("Looks like deps for the repo " + deps["repo"] +\
                " are not yet implemented.")
    for package in deps["packages"]:
        try:
            download_package(deps["repo"], package, verbose)
        except TypeError:
            download_package(deps["repo"], package, verbose)
        except Exception as e:
            print("A problem occured while downloading " + package["name"] +\
                    " version " + package["version"])
            print(e)

if ARGS.target not in TARGETS:
    print("--target " + ARGS.target + " is an invalid target option.")
    print("target should be one of : " + ", ".join(TARGETS) + ".")
    exit(1)

if ARGS.repo in REPOS:
    REPOS = [ARGS.repo]
elif ARGS.repo != "all":
    print("Repo " + ARGS.repo + " is an invalid repo option.")
    print("Repo should be one of : " + ", ".join(["all"] + REPOS) + ".")
    exit(1)

for repo in REPOS:
    if repo == "deps":
        update_deps(ARGS.verbose)
    elif repo == "visualizer-deps":
        update_visualizer_deps(ARGS.verbose)
    elif repo == "visualizer-data":
        update_visualizer_data(ARGS.verbose)
    else:
        raise Exception("unknown command")
