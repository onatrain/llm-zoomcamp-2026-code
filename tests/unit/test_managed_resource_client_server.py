"""Unit tests for the ManagedResource mixin using a Client/Server composition."""

import gc

import pytest

from classes.managed_resource import ManagedResource


class Server:
    """Resource double that records how many times it is closed."""

    def __init__(self) -> None:
        self.close_calls = 0
        self.is_closed = False

    def close(self) -> None:
        self.close_calls += 1
        self.is_closed = True


class Client(ManagedResource[Server]):
    """Owner that composes a Server and manages its lifecycle."""

    closed_message = "Client closed. You must open() it first."

    def __init__(self) -> None:
        super().__init__()
        self._server: Server | None = None

    @property
    def is_open(self) -> bool:
        return self._server is not None

    def open(self) -> Server:
        if self.is_open:
            self.close()
        server = Server()
        self._server = server
        self._register_resource(server, server.close)
        return server

    def ping(self) -> str:
        self._require_resource()
        return "pong"

    def close(self) -> None:
        if self._server is not None:
            self._server.close()
        self._server = None
        self._release_resource()


def test_a_fresh_client_is_closed() -> None:
    client = Client()

    assert client.is_open is False
    assert client._resource is None


def test_require_resource_raises_the_closed_message() -> None:
    client = Client()

    with pytest.raises(RuntimeError, match="Client closed."):
        client._require_resource()


def test_opening_registers_the_server_and_arms_a_finalizer() -> None:
    client = Client()

    server = client.open()

    assert client.is_open is True
    assert client._require_resource() is server
    assert client._finalizer is not None
    assert client._finalizer.alive is True


def test_ping_requires_an_open_client() -> None:
    client = Client()

    with pytest.raises(RuntimeError, match="Client closed."):
        client.ping()

    server = client.open()
    assert client.ping() == "pong"
    assert server.is_closed is False


def test_closing_releases_the_server_and_detaches_the_finalizer() -> None:
    client = Client()
    client.open()
    finalizer = client._finalizer

    client.close()

    assert client.is_open is False
    assert finalizer is not None
    assert finalizer.alive is False
    assert client._finalizer is None


def test_reopening_detaches_the_previous_finalizer_without_closing_it_twice() -> None:
    client = Client()
    first = client.open()
    first_finalizer = client._finalizer

    second = client.open()

    assert first_finalizer is not None
    assert first_finalizer.alive is False
    assert client._require_resource() is second
    assert first.close_calls == 1

    client.close()
    assert second.close_calls == 1
    assert first.close_calls == 1


def test_finalizer_closes_the_server_on_garbage_collection() -> None:
    client = Client()
    server = client.open()

    del client
    gc.collect()

    assert server.close_calls == 1
    assert server.is_closed is True


def test_context_manager_closes_the_client() -> None:
    client = Client()

    with client as handle:
        assert handle is client
        handle.open()
        assert client.is_open is True

    assert client.is_open is False


def test_exiting_context_manager_is_a_no_op_when_closed() -> None:
    client = Client()

    with client:
        pass

        assert client.is_open is False
