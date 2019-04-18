def url_join(scheme, hostname, port, *path):
    scheme = "{}://".format(scheme) if scheme is not None else ""
    netloc = "{}:{}".format(hostname, port) if port is not None else hostname
    path = "/{}".format("/".join(path)) if len(path) > 0 else ""
    return "{}{}{}".format(scheme, netloc, path)