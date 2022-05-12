from warnings import warn, catch_warnings, simplefilter
from . import gidr_dun_

# Workaround: https://github.com/numba/numba/issues/3341
import numba
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("umap-learn").version
except pkg_resources.DistributionNotFound:
    __version__ = "0.5-dev"
