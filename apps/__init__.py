"""Local AI Foundation – Application Composition Layer.

Provides the AppContext composition root for bootstrapping the full
Foundation stack (FoundationCore + InferenceConnector) and constructing
ready-to-use domain workflow instances.
"""

from apps.context import AppContext

__all__ = ["AppContext"]
