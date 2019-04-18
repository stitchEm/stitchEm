import vs
import json
import errors
import UserDict


class PTV(UserDict.UserDict):
    """ Helper class to manipulate PTV files
    """

    def _load(self, file_path):
        try:
            with open(file_path, 'r') as input_file:
                self.data = json.load(input_file)
        except IOError as error:
            if error.errno == 2:
                raise errors.ParserFileNotFound('Cannot load the configuration')
            else:
                raise errors.ParserError('Cannot load the configuration')
        except Exception as error:
            raise errors.ParserError('Cannot load the configuration')

    def _loads(self, string):
        try:
            self.data = json.loads(string)
        except:
            raise errors.ParserError('Cannot parse the configuration')

    def _merge(self, source, destination):
        """
        merge recursively source into destination
            for dict => if the key in source exists in destination
                we replace simple values and merge dicts and lists
            for list => we browse source
                if there is an element at the same index in destination,
                    we replace simple values and merge dicts and lists
                otherwise
                    we add the source list element at the end of destination list
        """
        if isinstance(source, dict):
            for k, v in source.iteritems():
                if isinstance(v, dict):
                    node = destination.setdefault(k, {})
                    self._merge(v, node)
                elif isinstance(v, list):
                    if k in destination:
                        self._merge(v, destination[k])
                    else:
                        destination[k] = v
                else:
                    destination[k] = v
        elif isinstance(source, list):
            for i in range(len(source)):
                if i < len(destination):
                    if isinstance(source[i], dict) or isinstance(source[i], list):
                        self._merge(source[i], destination[i])
                    else:
                        destination[i] = source[i]
                else:
                    destination.append(source[i])

    def _filter(self, source, filter):
        """
        deletes recursively from source all the entries not found in the filter
        """
        if isinstance(source, dict):
            for k in source.keys():
                if isinstance(source[k], dict) or isinstance(source[k], list):
                    if filter and k in filter:
                        self._filter(source[k], filter[k])
                    else:
                        del source[k]
                else:
                    if not filter or not k in filter:
                        del source[k]
        elif isinstance(source, list):
            for i in range(len(source)):
                if i < len(filter):
                    if isinstance(source[i], dict) or isinstance(source[i], list):
                        self._filter(source[i], filter[i])
                else:
                    source.pop()

    def merge(self, source, destination=None):
        destination = destination if destination is not None else self.data
        source = source.data if isinstance(source, PTV) else source
        self._merge(source, destination)

    def filter(self, source, filter):
        source = source.data if isinstance(source, PTV) else source
        self._filter(source, filter)

    def to_config(self, own=True):
        parser = vs.Parser_create()
        json_string = json.dumps(self.data)
        if not parser.parseData(json_string):
            raise errors.ParserError('Cannot parse the configuration')
        result = parser.getRoot().clone()
        result.thisown = int(own)  # as we receive pointer, swig doesn't know that we own the result
        return result

    @classmethod
    def from_ptv_value(cls, ptv_value):
        return cls(json.loads(ptv_value.getJsonStr()))

    @classmethod
    def from_file(cls, file_path):
        ptv = cls()
        ptv._load(file_path)
        return ptv

    @classmethod
    def from_string(cls, string):
        ptv = cls()
        ptv._loads(string)
        return ptv
