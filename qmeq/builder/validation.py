"""Module containing methods for validation of input parameters."""

ITYPE_OPTIONS = {
    0: ("finite", "quad"),
    1: ("infinite", "digamma"),
    2: ("finite", "omit"),
    3: ("infinite", "omit"),
}
TRANSPORT_OPTIONS = {options: itype for itype, options in ITYPE_OPTIONS.items()}


def validate_kerntype(kerntype):
    if isinstance(kerntype, str):
        if kerntype not in {'Pauli', 'Lindblad', 'Redfield', '1vN', '2vN', 'pyPauli',
                    'pyLindblad', 'pyRedfield', 'py1vN', 'py2vN', 'pyRTD', 'RTD'}:
            print("WARNING: Allowed kerntype values are: " +
                  "\'Pauli\', \'Lindblad\', \'Redfield\', \'1vN\', \'2vN\', " +
                  "\'pyPauli\', \'pyLindblad\', \'pyRedfield\', \'py1vN\', \'py2vN\', \'RTD\'. " +
                  "Using default kerntype=\'Pauli\'.")
            kerntype = 'Pauli'
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
        if descriptive_explicit:
            raise ValueError("itype must be 0, 1, 2, or 3.")
        print("WARNING: itype needs to be 0, 1, 2, or 3. Using default itype=0.")
        itype = 0

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
            bandwidth = bandwidth or "finite"

        if approach == "Lindblad":
            if principal_part == "quad":
                raise ValueError(
                    "The Lindblad approach supports principal_part='omit' "
                    "or 'digamma'; quadrature is not implemented."
                )
            principal_part = principal_part or "omit"
        else:
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

    if approach == "RTD" and itype != 1:
        if descriptive_explicit:
            raise ValueError(
                "The RTD approach requires bandwidth='infinite' and "
                "principal_part='digamma'."
            )
        if legacy_explicit:
            print("WARNING: only itype=1 is supported by the RTD approach. Using itype=1.")
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
        print("WARNING: itype_ph needs to be 0, or 2. Using default itype=0.")
        itype_ph = 0
    return itype_ph

def validate_mfreeq(kerntype, mfreeq):
    if mfreeq and kerntype in {'RTD', 'pyRTD'}:
        print("WARNING: mfreeq=True is not supported by the RTD approach. Using default mfreeq=False.")
        mfreeq = False
    return mfreeq

def validate_indexing(indexing, symmetry, kerntype):
    if indexing is None:
        if symmetry == 'spin' and kerntype in {'pyRTD', 'RTD'}:
            print("WARNING: symmetry=\'spin\' is not supported by the RTD approach. " +
                  "Using default indexing=\'charge\'.")
            indexing = 'charge'
            symmetry = None
        elif symmetry == 'spin' and kerntype not in {'py2vN', '2vN'}:
            indexing = 'ssq'
        else:
            indexing = 'charge'

    if indexing not in {'Lin', 'charge', 'sz', 'ssq'}:
        print("WARNING: Allowed indexing values are: \'Lin\', \'charge\', \'sz\', \'ssq\'. " +
              "Using default indexing=\'charge\'.")
        indexing = 'charge'

    if indexing not in {'Lin', 'charge'} and kerntype in {'py2vN', '2vN'}:
        print("WARNING: For the 2vN approach indexing needs to be \'Lin\' or \'charge\'. " +
              "Using indexing=\'charge\' as a default.")
        indexing = 'charge'

    if indexing != 'charge' and kerntype in {'pyRTD', 'RTD'}:
        print("WARNING: For the RTD approach indexing needs to be \'charge\'. " +
              "Using indexing=\'charge\' as a default.")
        indexing = 'charge'

    return indexing, symmetry
