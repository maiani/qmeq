"""Warning categories emitted by QmeQ."""


class QmeqWarning(UserWarning):
    """Base category for all warnings emitted by QmeQ."""


class QmeqRuntimeWarning(QmeqWarning, RuntimeWarning):
    """Warning for numerical or runtime conditions that may affect results."""
