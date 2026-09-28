"""Module containing methods for validation of input parameters."""

from numbers import Integral
import warnings

from .._warnings import QmeqWarning

ITYPE_OPTIONS = {
    0: ("finite", "quad"),
    1: ("infinite", "digamma"),
    2: ("finite", "omit"),
    3: ("infinite", "omit"),
}
TRANSPORT_OPTIONS = {options: itype for itype, options in ITYPE_OPTIONS.items()}


APPROACHES = ('Pauli', 'Lindblad', 'Redfield', '1vN', '2vN', 'RTD', 'RTDnoise')
KERNTYPES = APPROACHES + tuple('py' + approach for approach in APPROACHES)


def validate_kerntype(kerntype):
    """Refuse an unknown kerntype string rather than substitute another approach.

    A misspelled name used to fall back to Pauli with a warning, so a sweep
    meant for a coherent approach silently ran the Pauli one instead.
    """
    if isinstance(kerntype, str) and kerntype not in KERNTYPES:
        raise ValueError(
            f"Unknown kerntype {kerntype!r}. Allowed values are "
            + ", ".join(repr(name) for name in KERNTYPES) + "."
        )
    return kerntype


def resolve_transport_options(itype, bandwidth, principal_part, kerntype):
    """Resolve descriptive transport options and the legacy ``itype`` shorthand."""

    if bandwidth not in {None, "finite", "infinite"}:
        raise ValueError("bandwidth must be 'finite' or 'infinite'.")
    if principal_part not in {None, "quad", "digamma", "omit"}:
        raise ValueError(
            "principal_part must be 'quad', 'digamma', or 'omit'."
        )

    legacy_explicit = itype is not None
    descriptive_explicit = bandwidth is not None or principal_part is not None
    approach = kerntype[2:] if isinstance(kerntype, str) and kerntype.startswith("py") else kerntype

    if itype is None:
        itype = 0
    elif itype not in ITYPE_OPTIONS:
        raise ValueError(f"itype must be 0, 1, 2, or 3, not {itype!r}.")

    legacy_bandwidth, legacy_principal_part = ITYPE_OPTIONS[itype]

    if approach in ("Lindblad", "Pauli"):
        if legacy_explicit:
            expected_bandwidth = (
                "finite" if itype in (0, 2) else "infinite"
            )
            if bandwidth is not None and bandwidth != expected_bandwidth:
                raise ValueError(
                    f"itype={itype} implies bandwidth={expected_bandwidth!r} "
                    f"for the {approach} approach, not {bandwidth!r}."
                )
            bandwidth = expected_bandwidth
        else:
            bandwidth = bandwidth or (
                "infinite" if approach == "Lindblad" else "finite"
            )

        # The Lindblad principal_part, its Lamb shift, has no default: QmeQ 1.1
        # had no Lamb shift, so any default would silently change what a 1.1
        # script computes. It stays None (unset), and constructing or solving
        # an unset Lindblad system raises.
        if approach == "Pauli":
            if principal_part not in (None, "omit"):
                raise ValueError(
                    "The Pauli approach has no principal-value contribution; "
                    "use principal_part='omit'."
                )
            principal_part = "omit"

        if not legacy_explicit:
            itype = 0 if bandwidth == "finite" else 1

    elif approach == "2vN":
        if descriptive_explicit:
            raise ValueError(
                "The 2vN approach does not use bandwidth or principal_part."
            )
        bandwidth = None
        principal_part = None

    else:
        if not legacy_explicit and descriptive_explicit:
            if bandwidth is None:
                bandwidth = "infinite" if principal_part == "digamma" else "finite"
            if principal_part is None:
                principal_part = "digamma" if bandwidth == "infinite" else "quad"
            try:
                itype = TRANSPORT_OPTIONS[(bandwidth, principal_part)]
            except KeyError:
                raise ValueError(
                    f"The combination bandwidth={bandwidth!r}, "
                    f"principal_part={principal_part!r} is not supported."
                ) from None
        else:
            if bandwidth is not None and bandwidth != legacy_bandwidth:
                raise ValueError(
                    f"itype={itype} implies bandwidth={legacy_bandwidth!r}, "
                    f"not {bandwidth!r}."
                )
            if principal_part is not None and principal_part != legacy_principal_part:
                raise ValueError(
                    f"itype={itype} implies principal_part="
                    f"{legacy_principal_part!r}, not {principal_part!r}."
                )
            bandwidth = legacy_bandwidth
            principal_part = legacy_principal_part

    if approach in {"RTD", "RTDnoise"} and itype != 1:
        if descriptive_explicit:
            raise ValueError(
                "The RTD approach requires bandwidth='infinite' for its "
                "sequential rates and principal_part='digamma'. Its "
                "second-order integrals still use dband as a finite "
                "wide-band regulator."
            )
        if legacy_explicit:
            warnings.warn(
                "Only itype=1 is supported by the RTD approach. Using itype=1.",
                QmeqWarning,
                stacklevel=2,
            )
        itype = 1
        bandwidth, principal_part = ITYPE_OPTIONS[itype]

    return itype, bandwidth, principal_part


def validate_itype(itype, kerntype):
    """Validate the legacy integral selector."""

    return resolve_transport_options(
        itype, None, None, kerntype
    )[0]


def validate_itype_ph(itype_ph):
    if itype_ph not in {0, 2}:
        raise ValueError(f"itype_ph must be 0 or 2, not {itype_ph!r}.")
    return itype_ph

def validate_mfreeq(kerntype, mfreeq):
    if mfreeq and kerntype in {'RTD', 'pyRTD', 'RTDnoise', 'pyRTDnoise'}:
        raise ValueError(
            "mfreeq=True is not supported by the RTD approach; use the default "
            "mfreeq=False."
        )
    return mfreeq

def validate_indexing(indexing, symmetry, kerntype):
    # Unknown values are refused: a misspelled symmetry used to be ignored
    # and a misspelled indexing replaced, both changing the calculation.
    # Valid choices an approach cannot honour are still substituted with a
    # warning below, as in QmeQ 1.1.
    if symmetry not in {None, 'spin'}:
        raise ValueError(
            f"symmetry must be None or 'spin', not {symmetry!r}."
        )
    if indexing is not None and indexing not in {'Lin', 'charge', 'sz', 'ssq'}:
        raise ValueError(
            "indexing must be 'Lin', 'charge', 'sz', or 'ssq', not "
            f"{indexing!r}."
        )
    if indexing is None:
        if symmetry == 'spin' and kerntype in {'pyRTD', 'RTD', 'pyRTDnoise', 'RTDnoise'}:
            warnings.warn(
                "symmetry='spin' is not supported by the RTD approach. "
                "Using default indexing='charge'.",
                QmeqWarning,
                stacklevel=2,
            )
            indexing = 'charge'
            symmetry = None
        elif symmetry == 'spin' and kerntype not in {'py2vN', '2vN'}:
            indexing = 'ssq'
        else:
            indexing = 'charge'

    if indexing not in {'Lin', 'charge'} and kerntype in {'py2vN', '2vN'}:
        warnings.warn(
            "For the 2vN approach indexing needs to be 'Lin' or 'charge'. "
            "Using indexing='charge' as a default.",
            QmeqWarning,
            stacklevel=2,
        )
        indexing = 'charge'

    if indexing != 'charge' and kerntype in {'pyRTD', 'RTD', 'pyRTDnoise', 'RTDnoise'}:
        warnings.warn(
            "For the RTD approach indexing needs to be 'charge'. "
            "Using indexing='charge' as a default.",
            QmeqWarning,
            stacklevel=2,
        )
        indexing = 'charge'
    return indexing, symmetry

def validate_countingleads(countingleads, nleads=None):
    """Validate and freeze the lead order used for particle counting."""
    if countingleads is None:
        return None
    if isinstance(countingleads, (str, bytes)):
        raise TypeError("countingleads must be an iterable of lead indices.")
    try:
        leads = tuple(countingleads)
    except TypeError as exc:
        raise TypeError(
            "countingleads must be an iterable of lead indices."
        ) from exc
    if not leads:
        raise ValueError("countingleads must contain at least one lead index.")
    if any(isinstance(lead, bool) or not isinstance(lead, Integral)
           for lead in leads):
        raise TypeError("countingleads must contain only integer lead indices.")
    leads = tuple(int(lead) for lead in leads)
    if len(set(leads)) != len(leads):
        raise ValueError("countingleads must not contain duplicate lead indices.")
    if nleads is not None and any(lead < 0 or lead >= nleads for lead in leads):
        raise ValueError(
            f"countingleads entries must be between 0 and {nleads - 1}."
        )
    return leads
