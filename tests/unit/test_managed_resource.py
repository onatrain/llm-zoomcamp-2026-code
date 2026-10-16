"""Unit tests for the ManagedResource open/close lifecycle mixin."""

import gc

import pytest

from classes.managed_resource import ManagedResource


class _FakeResource:
    """Resource double that records how many times it is closed."""

    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _FakeOwner(ManagedResource[_FakeResource]):
    """Minimal owner following the mixin contract."""

    closed_message = "Fake resource closed."

    def open(self) -> _FakeResource:
        resource = _FakeResource()
        self._register_resource(resource, resource.close)
        return resource

    def close(self) -> None:
        resource = self._resource
        if resource is None:
            return
        resource.close()
        self._release_resource()


def test_a_fresh_owner_is_closed() -> None:
    owner = _FakeOwner()

    assert owner.is_open is False
    assert owner._resource is None


def test_require_resource_raises_the_closed_message() -> None:
    owner = _FakeOwner()

    with pytest.raises(RuntimeError, match="Fake resource closed."):
        owner._require_resource()


def test_registering_a_resource_opens_the_owner_and_arms_a_finalizer() -> None:
    owner = _FakeOwner()

    resource = owner.open()

    assert owner.is_open is True
    assert owner._require_resource() is resource
    assert owner._finalizer is not None
    assert owner._finalizer.alive is True


def test_releasing_a_resource_closes_the_owner_and_detaches_the_finalizer() -> None:
    owner = _FakeOwner()
    owner.open()
    finalizer = owner._finalizer

    owner._release_resource()

    assert owner.is_open is False
    assert finalizer is not None
    assert finalizer.alive is False
    assert owner._finalizer is None


def test_registering_a_new_resource_detaches_the_previous_finalizer() -> None:
    owner = _FakeOwner()
    first = owner.open()
    first_finalizer = owner._finalizer

    second = owner.open()

    assert first_finalizer is not None
    assert first_finalizer.alive is False
    assert owner._require_resource() is second

    owner.close()
    assert second.close_calls == 1
    assert first.close_calls == 0


def test_the_finalizer_closes_the_resource_on_garbage_collection() -> None:
    owner = _FakeOwner()
    resource = owner.open()

    del owner
    gc.collect()

    assert resource.close_calls == 1


def test_the_context_manager_closes_the_owner() -> None:
    owner = _FakeOwner()

    with owner as handle:
        assert handle is owner
        handle.open()
        assert owner.is_open is True

    assert owner.is_open is False


def test_exiting_the_context_manager_is_a_no_op_when_closed() -> None:
    owner = _FakeOwner()

    with owner:
        pass

    assert owner.is_open is False
