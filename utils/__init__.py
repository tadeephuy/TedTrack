ifnone = lambda x, y: y if x is None else x

get_name_from_path = lambda x: os.path.splitext(os.path.basename(x))[0]

def assert_in_list(x, x_values, name=''):
    assert x in x_values, f"{name} should be one of {x_values}"

from .tracking import *
from .detection import *
from .models import *