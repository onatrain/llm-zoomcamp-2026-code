import weakref
from typing import Any, Callable, Generic, Self, TypeVar

_T = TypeVar("_T")


class ManagedResource(Generic[_T]):
    """Mixin implementing the open/close lifecycle of a resource owner.

    Subclasses must define ``closed_message`` (raised when the resource is used
    while closed) and are responsible for implementing ``close()``, which must
    call ``_release_resource()`` once the underlying resource has been closed.
    """

    closed_message: str = "Resource closed. You must open it first."

    def __init__(self) -> None:
        self._resource: _T | None = None
        self._finalizer: weakref.finalize[Any, Any] | None = None

    def _register_resource(self, resource: _T, closer: Callable[[], None]) -> None:
        """Register a resource and the callable that closes it.

        Any previously registered resource is released first.

        Args:
            resource: The resource to manage.
            closer: Zero-argument callable that closes the resource, typically
                a bound method such as ``resource.close``.
        """
        self._release_resource()

        self._resource = resource
        self._finalizer = weakref.finalize(self, closer)

    def _require_resource(self) -> _T:
        """Return the managed resource or raise if it is closed.

        Returns:
            The resource currently held by this instance.

        Raises:
            RuntimeError: If the resource has been closed or never opened.
        """
        resource = self._resource
        if resource is None:
            raise RuntimeError(self.closed_message)

        return resource

    def _release_resource(self) -> None:
        """Drop the reference to the managed resource and detach its finalizer."""
        self._resource = None

        if self._finalizer is not None and self._finalizer.alive:
            self._finalizer.detach()
            self._finalizer = None

    @property
    def is_open(self) -> bool:
        """Whether the managed resource is currently held.

        Returns:
            True if a resource is registered, False otherwise.
        """
        return self._resource is not None

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns:
            This instance.
        """
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Leave the context manager, closing the managed resource."""
        close = getattr(self, "close", None)
        if callable(close):
            close()
