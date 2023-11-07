import logging
from typing import Union

import pkg_resources

logger = logging.getLogger("cxplain")


def get_resource_string(path: str, decode=True) -> Union[str, bytes]:
    """
    Load a package resource, such as a file within the package.

    Args:
        path (str): The path to the resource file, starting at the root of the current module
            (e.g., 'res/default.conf'). Must be a string, not a Path object.
        decode (bool): If True, decode the file contents as a string; otherwise, return the contents as bytes.

    Returns:
        Union[str, bytes]: The contents of the resource file, either as a string or bytes.
    """
    s = pkg_resources.resource_string(__name__.split(".")[0], path)
    return s.decode(errors="ignore") if decode else s
