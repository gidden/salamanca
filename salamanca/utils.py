from copy import deepcopy


class AttrObject(object):
    """Simple base class to have dictionary-like attributes attached to an 
    object
    """

    def __init__(self, **kwargs):
        self.update(copy=False, **kwargs)

    def update(self, copy=True, override=True, **kwargs):
        """Update attributes.

        Parameters
        ----------
        copy : bool, optional, default: True
            operate on a copy of this object
        override : bool, optional, default: True
            overrides attributes if they already exist on the object
        """
        x = deepcopy(self) if copy else self
        for k, v in kwargs.items():
            if override or getattr(x, k, None) is None:
                setattr(x, k, v)
        return x
