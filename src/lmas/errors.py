class LMASError(Exception):
    """Base exception for user-facing LMAS errors."""


class DependencyError(LMASError):
    """Raised when an optional runtime dependency is unavailable."""


class DatasetError(LMASError):
    """Raised when input data do not satisfy the LMAS source schema."""


class ArchiveMemberSelectionRequired(DatasetError):
    """Raised when a compressed bundle contains multiple plausible datasets."""

    def __init__(self, archive, members):
        self.archive = archive
        self.members = tuple(str(value) for value in members)
        super().__init__(
            f"Archive {archive} contains multiple LMA datasets; select one or load all: "
            + ", ".join(self.members)
        )


class ConfigurationError(LMASError):
    """Raised when a project or plotting configuration is invalid."""
