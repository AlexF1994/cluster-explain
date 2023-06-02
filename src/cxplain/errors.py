class XkmError(Exception):
    """Generic error for the cxplain package, please don't raise this error directly as
    it is too generic"""


class NotFittedError(XkmError):
    """Raised if an input of a calculation step hasn't been calculated before."""


class NonExistingRelevanceError(XkmError):
    """Raise if a special kind of relevance isn't available for specified explainer method."""


class NonExsitingXkmFlavourError(XkmError):
    """Raise if a xkm flavour is chosen, which doesn't exist."""


class InconsistentNamingError(XkmError):
    """Raise if number of feature names is not equal to number of features."""


class MetricError(XkmError):
    """Generic error for everything that has to do with metrics."""


class MetricNotImplementedError(MetricError):
    """Raise if a metric that is not yet impelemented should be used."""
