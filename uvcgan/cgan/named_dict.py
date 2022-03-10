
class NamedDict:
    # pylint: disable=too-many-instance-attributes

    def __init__(self, keys = None):
        if keys is not None:
            for key in keys:
                setattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def keys(self):
        return self.__dict__.keys()

