class DomainException(Exception):
    """Base exception for domain errors."""

    pass


class ModelNotFoundException(DomainException):
    """Raised when a DRL model is not found."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' not found")


class ModelLoadException(DomainException):
    """Raised when a model fails to load."""

    def __init__(self, model_id: str, reason: str):
        self.model_id = model_id
        self.reason = reason
        super().__init__(f"Failed to load model '{model_id}': {reason}")


class InstanceParseException(DomainException):
    """Raised when an instance file cannot be parsed."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"Failed to parse instance '{filename}': {reason}")


class UnsupportedFileFormatException(DomainException):
    """Raised when an uploaded file has an unsupported format."""

    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(
            f"Unsupported file format for '{filename}'. Use .json or .vrp files."
        )
