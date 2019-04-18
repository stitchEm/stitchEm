from HTMLParser import HTMLParser

class HTMLValidator(HTMLParser):
    """
    super simple html validator : check that each opening tag is closed
        with respect to tag hierarchy
    """

    def __init__(self):
        HTMLParser.__init__(self)

    def handle_starttag(self, tag, attrs):
        self.tag_stack.append(tag)

    def handle_endtag(self, tag):
        try:
            open_tag = self.tag_stack.pop()
            assert open_tag == tag
        except IndexError:
            raise Exception(
                "found an end tag but there was no more opened ones")
        except AssertionError:
            raise Exception(
                "mismatch between opened tag {} and closing tag {}".format(
                    open_tag, tag))

    def feed(self, data):
        self.tag_stack = []
        HTMLParser.feed(self, data)
