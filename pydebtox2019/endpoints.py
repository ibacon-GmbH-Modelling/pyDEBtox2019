# Shared endpoint registry.
#
# This is a deliberately dependency-free leaf module: both readin.py (which
# builds the data containers this registry names) and models.py (which
# consumes them during ODE solving/fitting) import from here, instead of
# one of those two importing the other. Keeping it separate avoids the
# readin.py <-> models.py dependency direction being backwards either way
# (see review item A3).
from dataclasses import dataclass


@dataclass(frozen=True)
class EndpointSpec:
    """
    Everything the rest of the package needs to know about one observable
    endpoint, in one place, so it doesn't have to be re-declared (and kept
    in sync by hand) everywhere an endpoint is referenced by code.

    Attributes:
    - code: the integer endpoint code used throughout the package (in
      active_endpoints, calc_ecx/calc_epx's `endpoints=` argument, etc.)
    - name: display/lookup name, e.g. 'length'
    - state_idx: row in the array returned by calc_model that this
      endpoint reads. NOTE this is *not* the same numbering as `code` -
      e.g. survival is code 0 but state row 3 - which is exactly why this
      needs to be an explicit, named mapping rather than an assumption.
    - dataset_attr: attribute name on a `completedataset` instance that
      signals this endpoint is present (e.g. 'lengthdata'); also the key
      used in `completedataset.time_indices` for this endpoint.
    - struct_list_attr: attribute name on `DEBtox2019models` holding the
      per-dataset list of this endpoint's data structure (e.g.
      'lengthstruct_list').
    - indexcommon_attr: attribute name on `DEBtox2019models` holding the
      per-dataset list of this endpoint's common-time indices (e.g.
      'indexcommon_length').
    - is_survival: True only for the survival endpoint - used where
      survival needs different handling from the other (continuous)
      endpoints, e.g. LCx/LPx vs ECx/EPx labeling.
    """
    code: int
    name: str
    state_idx: int
    dataset_attr: str
    struct_list_attr: str
    indexcommon_attr: str
    is_survival: bool = False


# Declared in the same order the pre-registry code used to append to
# active_endpoints (length, reproduction, survival) so that ordering -
# e.g. the order endpoints are summed in log_likelihood, or printed in
# efsa_criteria - is unchanged by this being a loop over a dict now.
ENDPOINTS = {
    1: EndpointSpec(1, 'length', state_idx=1, dataset_attr='lengthdata',
                     struct_list_attr='lengthstruct_list', indexcommon_attr='indexcommon_length'),
    2: EndpointSpec(2, 'reproduction', state_idx=2, dataset_attr='reprodata',
                     struct_list_attr='reprostruct_list', indexcommon_attr='indexcommon_repro'),
    0: EndpointSpec(0, 'survival', state_idx=3, dataset_attr='survdata',
                     struct_list_attr='survstruct_list', indexcommon_attr='indexcommon_surv',
                     is_survival=True),
}
NAME_TO_CODE = {spec.name: code for code, spec in ENDPOINTS.items()}
