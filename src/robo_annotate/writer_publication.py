"""Version-neutral, descriptor-anchored writer publication primitives."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
import secrets
import stat
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .secure_tree import SecureTree, rename_noreplace_at


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_OUTPUT_FILE_FLAGS = (
    os.O_RDWR
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
)
_PRIVATE_DIRECTORY_PREFIX = ".writer-dir-"
_RETIREMENT_NAMESPACE_PREFIX = ".writer-quarantine-"
_RETIREMENT_PREFIX = ".writer-retire-"
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_UNMOUNT = 0x00002000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_IN_ISDIR = 0x40000000
_IN_ONLYDIR = 0x01000000
_INOTIFY_EVENT = struct.Struct("iIII")
_BIRTH_WATCH_MASK = (
    _IN_CREATE
    | _IN_DELETE
    | _IN_MOVED_FROM
    | _IN_MOVED_TO
    | _IN_DELETE_SELF
    | _IN_MOVE_SELF
    | _IN_UNMOUNT
    | _IN_ONLYDIR
)


class _DirectoryBirthWitness:
    """Use an inode-bound inotify watch to prove one private mkdir birth."""

    _WATCH_MASK = _BIRTH_WATCH_MASK

    def __init__(self, descriptor: int, watch: int, context: str) -> None:
        self.descriptor = descriptor
        self.watch = watch
        self.context = context
        self._events: list[tuple[int, int, bytes]] = []

    @classmethod
    def open(cls, parent_fd: int, context: str) -> "_DirectoryBirthWitness":
        libc = ctypes.CDLL(None, use_errno=True)
        initialize = getattr(libc, "inotify_init1", None)
        add_watch = getattr(libc, "inotify_add_watch", None)
        if initialize is None or add_watch is None:
            raise ValueError(
                "private directory creation requires Linux inotify"
            )
        initialize.argtypes = [ctypes.c_int]
        initialize.restype = ctypes.c_int
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        descriptor = initialize(os.O_NONBLOCK | os.O_CLOEXEC)
        if descriptor < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), context)
        watch = add_watch(
            descriptor,
            os.fsencode(f"/proc/self/fd/{parent_fd}"),
            cls._WATCH_MASK,
        )
        if watch >= 0:
            return cls(descriptor, watch, context)
        error = ctypes.get_errno()
        try:
            os.close(descriptor)
        except OSError as close_error:
            failure = ValueError(
                f"unable to witness {context} private directory creation"
            )
            failure.add_note(
                f"Secondary inotify close failure: {close_error}"
            )
            raise failure from OSError(error, os.strerror(error), context)
        raise OSError(error, os.strerror(error), context)

    def verify_birth(self, name: str) -> None:
        """Require one mkdir event and reject every mutation of its name."""
        self._events.extend(self._read_events())
        expected_name = os.fsencode(name)
        relevant = [
            mask
            for mask, _, event_name in self._events
            if event_name == expected_name
        ]
        global_failure = self._has_global_failure(self._events)
        if global_failure or relevant != [_IN_CREATE | _IN_ISDIR]:
            raise ValueError(
                f"{self.context} changed while establishing ownership"
            )

    @staticmethod
    def _has_global_failure(
        events: Sequence[tuple[int, int, bytes]],
    ) -> bool:
        failures = (
            _IN_Q_OVERFLOW
            | _IN_IGNORED
            | _IN_DELETE_SELF
            | _IN_MOVE_SELF
            | _IN_UNMOUNT
        )
        return any(mask & failures for mask, _, _ in events)

    def _read_events(self) -> list[tuple[int, int, bytes]]:
        events: list[tuple[int, int, bytes]] = []
        while True:
            try:
                payload = os.read(self.descriptor, 64 * 1024)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    break
                raise ValueError(
                    f"unable to witness {self.context} private directory creation"
                ) from exc
            if not payload:
                break
            offset = 0
            while offset < len(payload):
                if len(payload) - offset < _INOTIFY_EVENT.size:
                    raise ValueError("truncated private directory birth event")
                _, mask, cookie, length = _INOTIFY_EVENT.unpack_from(
                    payload,
                    offset,
                )
                offset += _INOTIFY_EVENT.size
                if length > len(payload) - offset:
                    raise ValueError("truncated private directory birth event")
                raw_name = payload[offset : offset + length]
                offset += length
                events.append((mask, cookie, raw_name.split(b"\0", 1)[0]))
        return events

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            os.close(descriptor)


@dataclass(frozen=True)
class _EntryIdentity:
    device: int
    inode: int
    mode: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_EntryIdentity":
        return cls(value.st_dev, value.st_ino, value.st_mode)

    @property
    def directory_key(self) -> tuple[int, int]:
        return self.device, self.inode


class _OwnershipRegistry:
    """Track active cleanup authority without allowing inode-reuse matches."""

    def __init__(self) -> None:
        self._active: set[_EntryIdentity] = set()
        self._retiring: set[_EntryIdentity] = set()
        self._retirement_namespace: _RetirementNamespace | None = None

    def bind_retirement_namespace(
        self,
        namespace: "_RetirementNamespace",
    ) -> None:
        if self._retirement_namespace is not None:
            raise ValueError("writer retirement namespace is already bound")
        self._retirement_namespace = namespace

    def replace_retirement_namespace(
        self,
        namespace: "_RetirementNamespace",
    ) -> None:
        if self._retirement_namespace is None:
            raise ValueError("writer retirement namespace is unavailable")
        self._retirement_namespace = namespace

    @property
    def retirement_namespace(self) -> "_RetirementNamespace":
        if self._retirement_namespace is None:
            raise ValueError("writer retirement namespace is unavailable")
        return self._retirement_namespace

    def register(self, identity: _EntryIdentity) -> None:
        if identity in self._active or identity in self._retiring:
            raise ValueError("duplicate task-owned filesystem identity")
        self._active.add(identity)

    def is_owned(self, identity: _EntryIdentity) -> bool:
        return identity in self._active

    def begin_retirement(self, identity: _EntryIdentity) -> bool:
        if identity not in self._active:
            return False
        self._active.remove(identity)
        self._retiring.add(identity)
        return True

    def restore(self, identity: _EntryIdentity) -> None:
        if identity in self._retiring:
            self._retiring.remove(identity)
            self._active.add(identity)

    def finish_retirement(self, identity: _EntryIdentity) -> None:
        self._retiring.discard(identity)

    def release_all(self) -> None:
        self._active.clear()
        self._retiring.clear()

    def has_active(self) -> bool:
        return bool(self._active or self._retiring)


class _CleanupFailures:
    """Attempt every cleanup and preserve an already-active primary error."""

    def __init__(self) -> None:
        self.errors: list[tuple[str, BaseException]] = []

    def attempt(self, context: str, action: Callable[[], None]) -> None:
        try:
            action()
        except BaseException as exc:
            self.errors.append((context, exc))

    def finish(self, primary: BaseException | None, context: str) -> None:
        if not self.errors:
            return
        details = "; ".join(
            f"{label}: {type(error).__name__}: {error}"
            for label, error in self.errors
        )
        if primary is not None:
            primary.add_note(f"Secondary {context} failures: {details}")
            return
        first_error = self.errors[0][1]
        error = ValueError(f"{context} cleanup failed: {details}")
        error.add_note(f"Cleanup failures: {details}")
        raise error from first_error


class _AnchoredDirectoryPath:
    """Hold every component of an absolute directory path open and verified."""

    def __init__(
        self,
        path: Path,
        label: str,
        descriptors: list[int],
        names: list[str],
        identities: list[_EntryIdentity],
        created_final: bool,
    ) -> None:
        self.path = path
        self.label = label
        self._descriptors = descriptors
        self._names = names
        self._identities = identities
        self.created_final = created_final

    @classmethod
    def open(
        cls,
        path: Path,
        label: str,
        *,
        create_final: bool,
        registry: _OwnershipRegistry | None = None,
    ) -> "_AnchoredDirectoryPath":
        absolute = Path(os.path.abspath(path))
        components = absolute.parts[1:]
        if not components:
            raise ValueError(f"{label} must not be the filesystem root")
        root_fd = os.open("/", _DIRECTORY_FLAGS)
        try:
            root_identity = _directory_identity(os.fstat(root_fd), label)
        except BaseException as exc:
            failures = _CleanupFailures()
            failures.attempt("root descriptor close", lambda: os.close(root_fd))
            primary = (
                exc
                if isinstance(exc, ValueError)
                else ValueError(f"unable to anchor {label}")
            )
            failures.finish(primary, label)
            if primary is exc:
                raise
            raise primary from exc
        descriptors = [root_fd]
        names: list[str] = []
        identities = [root_identity]
        created_final_component = False
        try:
            for position, component in enumerate(components):
                parent_fd = descriptors[-1]
                try:
                    before = os.stat(
                        component,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if not create_final or position != len(components) - 1:
                        raise ValueError(
                            f"{label} parent components must already exist"
                        ) from None
                    if registry is None:
                        raise ValueError(f"ownership registry required for {label}")
                    parent_is_attached = lambda: _anchored_chain_is_attached(
                        descriptors,
                        names,
                        identities,
                        label,
                    )
                    try:
                        child = _create_published_child_directory(
                            parent_fd,
                            component,
                            label,
                            parent_is_attached,
                            registry,
                        )
                    except (OSError, ValueError) as exc:
                        raise ValueError(f"unable to anchor {label}") from exc
                    created_final_component = True
                    descriptors.append(child.descriptor)
                    child.descriptor = -1
                    names.append(component)
                    identities.append(child.identity)
                    continue
                expected = _directory_identity(before, label)
                child_fd = -1
                try:
                    child_fd = os.open(
                        component,
                        _DIRECTORY_FLAGS,
                        dir_fd=parent_fd,
                    )
                    opened = _directory_identity(os.fstat(child_fd), label)
                    current = _directory_identity(
                        os.stat(
                            component,
                            dir_fd=parent_fd,
                            follow_symlinks=False,
                        ),
                        label,
                    )
                    if expected != opened or opened != current:
                        raise ValueError(f"{label} changed while opening")
                except BaseException as exc:
                    failures = _CleanupFailures()
                    if child_fd >= 0:
                        failures.attempt(
                            "untracked directory close",
                            lambda descriptor=child_fd: os.close(descriptor),
                        )
                    failures.finish(exc, label)
                    raise
                descriptors.append(child_fd)
                names.append(component)
                identities.append(opened)
            return cls(
                absolute,
                label,
                descriptors,
                names,
                identities,
                created_final_component,
            )
        except BaseException as exc:
            primary = (
                exc
                if not isinstance(exc, OSError)
                else ValueError(f"unable to anchor {label}")
            )
            failures = _CleanupFailures()
            for descriptor in reversed(descriptors):
                failures.attempt(
                    "anchored descriptor close",
                    lambda current=descriptor: os.close(current),
                )
            descriptors.clear()
            failures.finish(primary, label)
            if primary is exc:
                raise
            raise primary from exc

    @property
    def descriptor(self) -> int:
        return self._descriptors[-1]

    @property
    def identity(self) -> _EntryIdentity:
        return self._identities[-1]

    @property
    def ancestry(self) -> tuple[tuple[int, int], ...]:
        return tuple(identity.directory_key for identity in self._identities)

    def verify(self, context: str) -> None:
        self._verify_components(len(self._names), context)

    def is_attached(self) -> bool:
        try:
            self._verify_components(len(self._names), "cleanup")
        except (OSError, ValueError):
            return False
        return True

    def _parent_chain_is_attached(self) -> bool:
        try:
            self._verify_components(
                max(0, len(self._names) - 1),
                "cleanup",
            )
        except (OSError, ValueError):
            return False
        return True

    def _verify_components(self, count: int, context: str) -> None:
        if len(self._descriptors) != len(self._identities):
            raise ValueError(f"closed {self.label}")
        if _directory_identity(
            os.fstat(self._descriptors[0]), self.label
        ) != self._identities[0]:
            raise ValueError(f"{self.label} changed during {context}")
        for index, name in enumerate(self._names[:count]):
            expected = self._identities[index + 1]
            try:
                entry = _directory_identity(
                    os.stat(
                        name,
                        dir_fd=self._descriptors[index],
                        follow_symlinks=False,
                    ),
                    self.label,
                )
                opened = _directory_identity(
                    os.fstat(self._descriptors[index + 1]), self.label
                )
            except (OSError, ValueError) as exc:
                raise ValueError(f"{self.label} changed during {context}") from exc
            if entry != expected or opened != expected:
                raise ValueError(f"{self.label} changed during {context}")

    def remove_created_final(self, registry: _OwnershipRegistry) -> None:
        if not self.created_final:
            return
        self.remove_final_if_owned(registry)

    def remove_final_if_owned(self, registry: _OwnershipRegistry) -> None:
        """Remove the anchored final entry only when its exact identity is owned."""
        parent_fd = self._descriptors[-2]
        name = self._names[-1]
        _remove_owned_empty_directory_at(
            parent_fd,
            name,
            self.identity,
            registry,
            self._parent_chain_is_attached,
        )

    def close(self) -> None:
        failures = _CleanupFailures()
        for descriptor in reversed(self._descriptors):
            failures.attempt(
                "anchored descriptor close",
                lambda current=descriptor: os.close(current),
            )
        self._descriptors.clear()
        failures.finish(None, self.label)


class _WriterLock:
    """Serialize cooperating Robo-Annotate writers on one staging parent.

    Linux ``flock`` is advisory.  This lock is the supported concurrency
    boundary for Robo-Annotate writers; it is not an atomic compare-delete
    defense against an uncooperative same-credential process.
    """

    def __init__(
        self,
        parent: _AnchoredDirectoryPath,
        descriptor: int,
        identity: _EntryIdentity,
    ) -> None:
        self.parent = parent
        self.descriptor = descriptor
        self.identity = identity
        self.held = True

    @classmethod
    def acquire(cls, parent: _AnchoredDirectoryPath) -> "_WriterLock":
        parent.verify("writer lock acquisition")
        descriptor = os.dup(parent.descriptor)
        try:
            identity = _directory_identity(
                os.fstat(descriptor),
                "staging parent writer lock",
            )
            if identity != parent.identity:
                raise ValueError("staging parent changed before writer lock")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            parent.verify("writer lock acquisition")
            if (
                _directory_identity(
                    os.fstat(descriptor),
                    "staging parent writer lock",
                )
                != identity
            ):
                raise ValueError("staging parent changed during writer lock")
            return cls(parent, descriptor, identity)
        except BaseException as exc:
            failures = _CleanupFailures()
            failures.attempt(
                "writer lock descriptor close",
                lambda: os.close(descriptor),
            )
            primary = (
                exc
                if isinstance(exc, ValueError)
                else ValueError("writer lock acquisition failed")
            )
            failures.finish(primary, "writer lock acquisition")
            if primary is exc:
                raise
            raise primary from exc

    def verify(self, context: str) -> None:
        if not self.held or self.descriptor < 0:
            raise ValueError("staging parent writer lock is not held")
        self.parent.verify(context)
        if (
            _directory_identity(
                os.fstat(self.descriptor),
                "staging parent writer lock",
            )
            != self.identity
        ):
            raise ValueError(f"staging parent changed during {context}")

    def release(self) -> None:
        if self.descriptor >= 0 and self.held:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            self.held = False

    def reacquire(self) -> None:
        if self.descriptor < 0:
            raise ValueError("staging parent writer lock descriptor is closed")
        fcntl.flock(self.descriptor, fcntl.LOCK_EX)
        self.held = True
        self.verify("writer lock recovery")

    def rebind_parent(
        self,
        parent: _AnchoredDirectoryPath,
        context: str,
    ) -> None:
        """Replace a failed path anchor while retaining the lock descriptor."""
        if self.descriptor < 0:
            raise ValueError("staging parent writer lock descriptor is closed")
        parent.verify(context)
        if (
            parent.identity != self.identity
            or _directory_identity(
                os.fstat(self.descriptor),
                "staging parent writer lock",
            )
            != self.identity
        ):
            raise ValueError(f"staging parent changed during {context}")
        self.parent = parent
        if self.held:
            self.verify(context)

    def close(self) -> None:
        failures = _CleanupFailures()
        failures.attempt("writer lock release", self.release)
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            failures.attempt(
                "writer lock descriptor close",
                lambda: os.close(descriptor),
            )
        failures.finish(None, "writer lock")


@dataclass
class _ChildDirectory:
    parent_fd: int
    name: str
    descriptor: int
    identity: _EntryIdentity
    context: str
    created: bool

    def verify(self, phase: str) -> None:
        try:
            entry = _directory_identity(
                os.stat(
                    self.name,
                    dir_fd=self.parent_fd,
                    follow_symlinks=False,
                ),
                self.context,
            )
            opened = _directory_identity(os.fstat(self.descriptor), self.context)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{self.context} changed during {phase}") from exc
        if entry != self.identity or opened != self.identity:
            raise ValueError(f"{self.context} changed during {phase}")

    def is_attached(self) -> bool:
        try:
            self.verify("cleanup")
        except (OSError, ValueError):
            return False
        return True

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            os.close(descriptor)


class _RetirementNamespace:
    """Hold a never-published quarantine for one locked writer transaction."""

    def __init__(
        self,
        directory: _ChildDirectory,
        writer_lock: _WriterLock,
    ) -> None:
        self.directory = directory
        self.writer_lock = writer_lock
        self._unexpected: set[str] = set()

    @property
    def descriptor(self) -> int:
        return self.directory.descriptor

    def verify(self, context: str) -> None:
        self.writer_lock.verify(context)
        self.directory.verify(context)

    def is_attached(self) -> bool:
        try:
            self.verify("retirement quarantine cleanup")
        except (OSError, ValueError):
            return False
        return True

    def inspect(self, allowed: Sequence[str] = ()) -> None:
        self.verify("retirement quarantine inspection")
        permitted = set(allowed)
        self._unexpected.update(
            name for name in os.listdir(self.descriptor) if name not in permitted
        )

    def verify_clean(self) -> None:
        self.inspect()
        if self._unexpected:
            names = ", ".join(sorted(self._unexpected))
            raise ValueError(
                f"retirement quarantine contains unexpected entries: {names}"
            )

    def remove(self) -> None:
        self.verify_clean()
        removed = _remove_locked_owned_empty_directory_at(
            self.directory.parent_fd,
            self.directory.name,
            self.directory.identity,
            self.is_attached,
        )
        if not removed:
            raise ValueError("retirement quarantine removal was incomplete")

    def close(self) -> None:
        self.directory.close()


@dataclass
class OwnedFile:
    """Hold one writer-owned regular file by descriptor and exact identity."""

    parent_fd: int
    name: str
    descriptor: int
    identity: _EntryIdentity
    context: str

    @property
    def proc_path(self) -> Path:
        if self.descriptor < 0:
            raise ValueError(f"closed {self.context}")
        return Path(f"/proc/self/fd/{self.descriptor}")

    def verify(self, phase: str) -> None:
        try:
            descriptor = _EntryIdentity.from_stat(os.fstat(self.descriptor))
            current = _entry_at(self.parent_fd, self.name)
        except OSError as exc:
            raise ValueError(f"{self.context} changed during {phase}") from exc
        if (
            descriptor != self.identity
            or current != self.identity
            or not stat.S_ISREG(self.identity.mode)
        ):
            raise ValueError(f"{self.context} changed during {phase}")

    def close(self) -> None:
        if self.descriptor >= 0:
            descriptor = self.descriptor
            self.descriptor = -1
            os.close(descriptor)


def _directory_identity(value: os.stat_result, context: str) -> _EntryIdentity:
    if stat.S_ISLNK(value.st_mode):
        raise ValueError(f"{context} path contains a symbolic link")
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"{context} path must contain only real directories")
    return _EntryIdentity.from_stat(value)


def _anchored_chain_is_attached(
    descriptors: Sequence[int],
    names: Sequence[str],
    identities: Sequence[_EntryIdentity],
    context: str,
) -> bool:
    """Prove that every held descriptor is still linked at its expected name."""
    try:
        if len(descriptors) != len(identities):
            return False
        if _directory_identity(os.fstat(descriptors[0]), context) != identities[0]:
            return False
        for index, name in enumerate(names):
            expected = identities[index + 1]
            if (
                _directory_identity(
                    os.stat(
                        name,
                        dir_fd=descriptors[index],
                        follow_symlinks=False,
                    ),
                    context,
                )
                != expected
                or _directory_identity(
                    os.fstat(descriptors[index + 1]), context
                )
                != expected
            ):
                return False
    except (OSError, ValueError):
        return False
    return True


def _open_or_create_child_directory(
    parent_fd: int,
    name: str,
    context: str,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry,
) -> _ChildDirectory:
    try:
        try:
            before = os.stat(
                name,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return _create_published_child_directory(
                parent_fd,
                name,
                context,
                parent_is_attached,
                registry,
            )
        expected = _directory_identity(before, context)
        return _open_child_directory(
            parent_fd,
            name,
            expected,
            context,
            created=False,
        )
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"unable to open {context}") from exc


def _open_child_directory(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    context: str,
    *,
    created: bool,
) -> _ChildDirectory:
    descriptor = -1
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        opened = _directory_identity(os.fstat(descriptor), context)
        current = _directory_identity(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False),
            context,
        )
        if expected != opened or opened != current:
            raise ValueError(f"{context} changed while opening")
        return _ChildDirectory(
            parent_fd,
            name,
            descriptor,
            expected,
            context,
            created,
        )
    except BaseException as exc:
        failures = _CleanupFailures()
        if descriptor >= 0:
            failures.attempt(
                "untracked child descriptor close",
                lambda: os.close(descriptor),
            )
        failures.finish(exc, context)
        raise


def _create_private_child_directory(
    parent_fd: int,
    prefix: str,
    context: str,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry | None,
) -> _ChildDirectory:
    """Create and witness one private child before accepting its pinned identity."""
    for _ in range(100):
        name = prefix + secrets.token_hex(12)
        if not parent_is_attached():
            raise ValueError(f"{context} parent changed before creation")
        witness: _DirectoryBirthWitness | None = None
        try:
            witness = _DirectoryBirthWitness.open(parent_fd, context)
            if not parent_is_attached():
                raise ValueError(f"{context} parent changed before creation")
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            if witness is not None:
                witness.close()
            continue
        except BaseException as exc:
            failures = _CleanupFailures()
            if witness is not None:
                failures.attempt("private birth witness close", witness.close)
            primary = (
                exc
                if isinstance(exc, ValueError)
                else ValueError(f"unable to create {context}")
            )
            failures.finish(primary, context)
            if primary is exc:
                raise
            raise primary from exc
        expected: _EntryIdentity | None = None
        descriptor = -1
        validation_descriptor = -1
        registered = False
        accepted = False
        try:
            # Pin the first object reachable after mkdir, derive authority only
            # from fstat, and let the event witness reject a replaced pathname.
            descriptor = os.open(
                os.fsencode(name),
                _DIRECTORY_FLAGS,
                dir_fd=parent_fd,
            )
            expected = _directory_identity(os.fstat(descriptor), context)
            assert witness is not None
            witness.verify_birth(name)
            if registry is not None:
                registry.register(expected)
                registered = True
            accepted = True
            current = _directory_identity(
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False), context
            )
            if not parent_is_attached() or expected != current:
                raise ValueError(f"{context} changed while establishing ownership")
            validation_descriptor = os.open(
                name,
                _DIRECTORY_FLAGS,
                dir_fd=parent_fd,
            )
            validation_identity = _directory_identity(
                os.fstat(validation_descriptor), context
            )
            if validation_identity != expected:
                raise ValueError(f"{context} changed while establishing ownership")
            os.close(validation_descriptor)
            validation_descriptor = -1
            witness.verify_birth(name)
            witness.close()
            witness = None
            if (
                not parent_is_attached()
                or _directory_identity(
                    os.stat(
                        name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    ),
                    context,
                )
                != expected
            ):
                raise ValueError(f"{context} changed while establishing ownership")
            return _ChildDirectory(
                parent_fd,
                name,
                descriptor,
                expected,
                context,
                True,
            )
        except BaseException as exc:
            failures = _CleanupFailures()
            if witness is not None:
                failures.attempt("private birth witness close", witness.close)
            if validation_descriptor >= 0:
                failures.attempt(
                    "private validation descriptor close",
                    lambda current=validation_descriptor: os.close(current),
                )
            if descriptor >= 0:
                failures.attempt(
                    "private directory descriptor close",
                    lambda current=descriptor: os.close(current),
                )
            if expected is not None and accepted:
                if registry is not None and registered:
                    failures.attempt(
                        "private directory removal",
                        lambda: _remove_owned_empty_directory_at(
                            parent_fd,
                            name,
                            expected,
                            registry,
                            parent_is_attached,
                        ),
                    )
                elif registry is None:
                    failures.attempt(
                        "private directory removal",
                        lambda: _remove_locked_owned_empty_directory_at(
                            parent_fd,
                            name,
                            expected,
                            parent_is_attached,
                        ),
                    )
            primary = (
                exc
                if not isinstance(exc, OSError)
                else ValueError(f"unable to open {context}")
            )
            failures.finish(primary, context)
            if primary is exc:
                raise
            raise primary from exc
    raise ValueError(f"unable to create {context}")


def _publish_private_directory(
    directory: _ChildDirectory,
    destination_parent_fd: int,
    destination_name: str,
    context: str,
    parent_is_attached: Callable[[], bool],
) -> _ChildDirectory:
    if not parent_is_attached():
        raise ValueError(f"{context} parent changed before publication")
    directory.verify("private publication")
    source_name = directory.name
    rename_noreplace_at(
        directory.parent_fd,
        source_name,
        destination_parent_fd,
        destination_name,
    )
    directory.parent_fd = destination_parent_fd
    directory.name = destination_name
    directory.context = context
    if not parent_is_attached():
        raise ValueError(f"{context} changed while publishing")
    try:
        opened = _directory_identity(os.fstat(directory.descriptor), context)
        current = _directory_identity(
            os.stat(
                destination_name,
                dir_fd=destination_parent_fd,
                follow_symlinks=False,
            ),
            context,
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{context} changed while publishing") from exc
    if opened != directory.identity or current != directory.identity:
        raise ValueError(f"{context} changed while publishing")
    return directory


def _create_published_child_directory(
    parent_fd: int,
    name: str,
    context: str,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry,
) -> _ChildDirectory:
    private: _ChildDirectory | None = None
    try:
        private = _create_private_child_directory(
            parent_fd,
            _PRIVATE_DIRECTORY_PREFIX,
            context,
            parent_is_attached,
            registry,
        )
        return _publish_private_directory(
            private,
            parent_fd,
            name,
            context,
            parent_is_attached,
        )
    except BaseException as exc:
        failures = _CleanupFailures()
        if private is not None:
            failures.attempt("private directory close", private.close)
            failures.attempt(
                "private directory removal",
                lambda: _remove_owned_empty_directory_at(
                    parent_fd,
                    private.name,
                    private.identity,
                    registry,
                    parent_is_attached,
                ),
            )
        failures.finish(exc, context)
        raise


def _entry_at(parent_fd: int, name: str) -> _EntryIdentity | None:
    try:
        return _EntryIdentity.from_stat(
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        )
    except FileNotFoundError:
        return None


def _remove_locked_owned_empty_directory_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    lock_and_parent_are_attached: Callable[[], bool],
) -> bool:
    """Remove one exact private directory while its parent writer lock is held."""
    if (
        not stat.S_ISDIR(expected.mode)
        or not lock_and_parent_are_attached()
        or _entry_at(parent_fd, name) != expected
    ):
        return False
    descriptor = -1
    primary_error: BaseException | None = None
    removed = False
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        valid = not (
            _EntryIdentity.from_stat(os.fstat(descriptor)) != expected
            or not lock_and_parent_are_attached()
            or _entry_at(parent_fd, name) != expected
        )
        if valid:
            os.rmdir(name, dir_fd=parent_fd)
            after = os.fstat(descriptor)
            if _EntryIdentity.from_stat(after) != expected or after.st_nlink != 0:
                raise ValueError("locked private directory removal was incomplete")
            if _entry_at(parent_fd, name) is not None:
                raise ValueError("locked private directory name was unexpectedly reused")
            removed = True
    except BaseException as exc:
        primary_error = exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as close_error:
                if primary_error is None:
                    primary_error = close_error
                else:
                    primary_error.add_note(
                        "Secondary locked directory descriptor close failure: "
                        f"{type(close_error).__name__}: {close_error}"
                    )
    if primary_error is not None:
        raise primary_error
    return removed


def _create_owned_file_at(
    parent_fd: int,
    name: str,
    context: str,
    parent_is_attached: Callable[[], bool],
    registry: _OwnershipRegistry,
) -> OwnedFile:
    if not name or "/" in name or name in (".", ".."):
        raise ValueError(f"unsafe {context} filename")
    if not parent_is_attached():
        raise ValueError(f"{context} parent changed before creation")
    descriptor = -1
    identity: _EntryIdentity | None = None
    try:
        descriptor = os.open(
            name,
            _OUTPUT_FILE_FLAGS,
            0o600,
            dir_fd=parent_fd,
        )
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise ValueError(f"{context} is not a regular file")
        identity = _EntryIdentity.from_stat(created)
        registry.register(identity)
        if (
            not parent_is_attached()
            or _entry_at(parent_fd, name) != identity
        ):
            raise ValueError(f"{context} changed while opening")
        return OwnedFile(parent_fd, name, descriptor, identity, context)
    except BaseException as exc:
        failures = _CleanupFailures()
        if descriptor >= 0:
            failures.attempt(
                f"{context} descriptor close",
                lambda: os.close(descriptor),
            )
        if identity is not None:
            failures.attempt(
                f"{context} removal",
                lambda: _remove_owned_entry_at(
                    parent_fd,
                    name,
                    identity,
                    registry,
                    parent_is_attached,
                ),
            )
        failures.finish(exc, context)
        raise


def _retire_and_delete_owned_entry_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> bool:
    """Move one owned identity into the locked private retirement namespace."""
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return False
    namespace = registry.retirement_namespace
    namespace.verify("owned entry retirement")
    if not registry.begin_retirement(expected):
        return False
    quarantine: str | None = None
    retirement_descriptor = -1
    retired = False
    primary_error: BaseException | None = None
    try:
        if not still_attached() or _entry_at(parent_fd, name) != expected:
            registry.restore(expected)
            return False
        namespace.inspect()
        for _ in range(100):
            candidate = _RETIREMENT_PREFIX + secrets.token_hex(12)
            if not still_attached() or not namespace.is_attached():
                raise ValueError("owned entry parent changed during retirement")
            try:
                rename_noreplace_at(
                    parent_fd,
                    name,
                    namespace.descriptor,
                    candidate,
                )
            except FileExistsError:
                namespace.inspect()
                continue
            quarantine = candidate
            break
        if quarantine is None:
            raise ValueError("unable to reserve an owned entry quarantine")
        flags = _DIRECTORY_FLAGS if stat.S_ISDIR(expected.mode) else (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        retirement_descriptor = os.open(
            quarantine,
            flags,
            dir_fd=namespace.descriptor,
        )
        pinned = _EntryIdentity.from_stat(os.fstat(retirement_descriptor))
        quarantined = _entry_at(namespace.descriptor, quarantine)
        if pinned != expected or quarantined != expected:
            raise ValueError("owned entry changed inside retirement quarantine")
        namespace.inspect((quarantine,))
        before = os.fstat(retirement_descriptor)
        if _EntryIdentity.from_stat(before) != expected or before.st_nlink <= 0:
            raise ValueError("owned entry changed inside retirement quarantine")
        if stat.S_ISDIR(expected.mode):
            os.rmdir(quarantine, dir_fd=namespace.descriptor)
        else:
            if before.st_nlink != 1:
                raise ValueError("owned file has an ambiguous retirement link count")
            os.unlink(quarantine, dir_fd=namespace.descriptor)
        after = os.fstat(retirement_descriptor)
        if _EntryIdentity.from_stat(after) != expected or after.st_nlink != 0:
            raise ValueError("owned entry retirement was incomplete")
        if _entry_at(namespace.descriptor, quarantine) is not None:
            raise ValueError("owned retirement quarantine name was unexpectedly reused")
        namespace.inspect()
        retired = True
    except BaseException as exc:
        primary_error = exc

    if retirement_descriptor >= 0:
        try:
            os.close(retirement_descriptor)
        except BaseException as close_error:
            if primary_error is None:
                primary_error = close_error
            else:
                primary_error.add_note(
                    "Secondary retirement descriptor close failure: "
                    f"{type(close_error).__name__}: {close_error}"
                )

    if retired:
        registry.finish_retirement(expected)
    else:
        registry.restore(expected)
        quarantined_after_failure: _EntryIdentity | None = None
        if quarantine is not None:
            try:
                quarantined_after_failure = _entry_at(
                    namespace.descriptor,
                    quarantine,
                )
            except BaseException as inspection_error:
                if primary_error is None:
                    primary_error = inspection_error
                else:
                    primary_error.add_note(
                        "Secondary owned quarantine inspection failure: "
                        f"{type(inspection_error).__name__}: {inspection_error}"
                    )
        if quarantined_after_failure == expected:
            assert quarantine is not None
            try:
                if (
                    not still_attached()
                    or _entry_at(parent_fd, name) is not None
                    or not namespace.is_attached()
                ):
                    raise ValueError(
                        "owned entry parent changed before quarantine restore"
                    )
                rename_noreplace_at(
                    namespace.descriptor,
                    quarantine,
                    parent_fd,
                    name,
                )
            except BaseException as restore_error:
                assert primary_error is not None
                primary_error.add_note(
                    "Secondary owned quarantine restore failure: "
                    f"{type(restore_error).__name__}: {restore_error}"
                )
        elif quarantined_after_failure is not None:
            try:
                namespace.inspect()
            except BaseException as inspection_error:
                assert primary_error is not None
                primary_error.add_note(
                    "Secondary unexpected quarantine inspection failure: "
                    f"{type(inspection_error).__name__}: {inspection_error}"
                )
    if primary_error is not None:
        raise primary_error
    return True


def _remove_owned_empty_directory_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> None:
    if not stat.S_ISDIR(expected.mode) or not still_attached():
        return
    if _entry_at(parent_fd, name) != expected or not still_attached():
        return
    if _entry_at(parent_fd, name) != expected:
        return
    _retire_and_delete_owned_entry_at(
        parent_fd,
        name,
        expected,
        registry,
        still_attached,
    )


def _remove_owned_tree_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> None:
    """Delete only identities recorded as task-owned below an attached parent."""
    if not registry.is_owned(expected) or not still_attached():
        return
    if _entry_at(parent_fd, name) != expected:
        return
    if not stat.S_ISDIR(expected.mode):
        if not still_attached() or _entry_at(parent_fd, name) != expected:
            return
        _retire_and_delete_owned_entry_at(
            parent_fd,
            name,
            expected,
            registry,
            still_attached,
        )
        return
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return
    descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    failures = _CleanupFailures()
    traversal_error: BaseException | None = None
    opened_valid = False
    try:
        opened_valid = not (
            _EntryIdentity.from_stat(os.fstat(descriptor)) != expected
            or not still_attached()
            or _entry_at(parent_fd, name) != expected
        )

        def directory_is_attached() -> bool:
            try:
                return (
                    still_attached()
                    and _EntryIdentity.from_stat(os.fstat(descriptor)) == expected
                    and _entry_at(parent_fd, name) == expected
                )
            except OSError:
                return False

        if opened_valid:
            for child in sorted(os.listdir(descriptor)):
                if not directory_is_attached():
                    opened_valid = False
                    break
                child_identity = _entry_at(descriptor, child)
                if (
                    child_identity is not None
                    and registry.is_owned(child_identity)
                ):
                    failures.attempt(
                        f"owned child removal {child}",
                        lambda child_name=child, identity=child_identity: (
                            _remove_owned_tree_at(
                                descriptor,
                                child_name,
                                identity,
                                registry,
                                directory_is_attached,
                            )
                        ),
                    )
    except BaseException as exc:
        traversal_error = exc
    finally:
        failures.attempt("owned tree descriptor close", lambda: os.close(descriptor))
    failures.finish(traversal_error, "owned tree")
    if traversal_error is not None:
        raise traversal_error
    if not opened_valid:
        return
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return
    _remove_owned_empty_directory_at(
        parent_fd,
        name,
        expected,
        registry,
        still_attached,
    )


def _remove_owned_entry_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> None:
    _remove_owned_tree_at(
        parent_fd,
        name,
        expected,
        registry,
        still_attached,
    )


def _remove_owned_identities_below(
    parent_fd: int,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
    visited: set[tuple[int, int]] | None = None,
) -> None:
    """Find and remove recorded identities only below a live anchored root."""
    if visited is None:
        visited = set()
    failures = _CleanupFailures()
    for name in sorted(os.listdir(parent_fd)):
        if not still_attached():
            break
        current = _entry_at(parent_fd, name)
        if current is None:
            continue
        if registry.is_owned(current):
            failures.attempt(
                f"owned entry removal {name}",
                lambda current_name=name, current_identity=current: (
                    _remove_owned_tree_at(
                        parent_fd,
                        current_name,
                        current_identity,
                        registry,
                        still_attached,
                    )
                ),
            )
            continue
        if not stat.S_ISDIR(current.mode):
            continue
        key = current.directory_key
        if key in visited:
            continue
        descriptor = -1
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            opened = _EntryIdentity.from_stat(os.fstat(descriptor))
            linked = _entry_at(parent_fd, name)
            if opened != current or linked != current:
                continue
            visited.add(key)

            def directory_is_attached(
                parent_guard: Callable[[], bool] = still_attached,
                parent_descriptor: int = parent_fd,
                child_name: str = name,
                child_descriptor: int = descriptor,
                expected: _EntryIdentity = current,
            ) -> bool:
                try:
                    return (
                        parent_guard()
                        and _EntryIdentity.from_stat(
                            os.fstat(child_descriptor)
                        )
                        == expected
                        and _entry_at(parent_descriptor, child_name) == expected
                    )
                except OSError:
                    return False

            _remove_owned_identities_below(
                descriptor,
                registry,
                directory_is_attached,
                visited,
            )
        except BaseException as exc:
            failures.errors.append((f"cleanup traversal {name}", exc))
            continue
        finally:
            if descriptor >= 0:
                failures.attempt(
                    "cleanup traversal descriptor close",
                    lambda current_descriptor=descriptor: os.close(
                        current_descriptor
                    ),
                )
    failures.finish(None, "cleanup traversal")


def capture_owned_directory_identity(path: Path) -> tuple[int, int, int]:
    """Capture the exact stable identity used to authorize later cleanup."""
    descriptor = -1
    primary: BaseException | None = None
    result: tuple[int, int, int] | None = None
    try:
        descriptor = os.open(path, _DIRECTORY_FLAGS)
        identity = _directory_identity(
            os.fstat(descriptor),
            "owned staging directory",
        )
        current = _directory_identity(
            os.stat(path, follow_symlinks=False),
            "owned staging directory",
        )
        if identity != current:
            raise ValueError("owned staging directory changed while opening")
        result = identity.device, identity.inode, stat.S_IFMT(identity.mode)
    except BaseException as exc:
        primary = exc
    finally:
        failures = _CleanupFailures()
        if descriptor >= 0:
            failures.attempt(
                "owned staging descriptor close",
                lambda: os.close(descriptor),
            )
        failures.finish(primary, "owned staging identity capture")
    if primary is not None:
        raise primary
    assert result is not None
    return result


def remove_owned_staging_tree(
    staging: Path,
    parent: Path,
    output_name: str,
    expected_key: tuple[int, int, int] | None,
) -> None:
    """Retire one converter-owned staging tree under the writer lock.

    Cleanup authority follows the pinned directory identity, never the staging
    pathname.  The tree is deleted only after an atomic move into this
    transaction's private quarantine below the locked staging parent.
    """
    staging = Path(staging)
    parent = Path(parent)
    if (
        staging.parent != parent
        or not staging.name.startswith(f"{output_name}.staging-")
    ):
        raise ValueError("refusing to clean an unowned path")
    _remove_owned_tree_under_writer_lock(
        staging,
        parent,
        expected_key,
        "staging cleanup",
    )


def remove_owned_published_tree(
    output: Path,
    parent: Path,
    output_name: str,
    expected_key: tuple[int, int, int] | None,
) -> None:
    """Roll back one exactly identified publication with the writer contract."""
    output = Path(output)
    parent = Path(parent)
    if output.parent != parent or output.name != output_name:
        raise ValueError("refusing to clean an unowned published path")
    _remove_owned_tree_under_writer_lock(
        output,
        parent,
        expected_key,
        "published output rollback",
    )


def _remove_owned_tree_under_writer_lock(
    staging: Path,
    parent: Path,
    expected_key: tuple[int, int, int] | None,
    cleanup_context: str,
) -> None:
    if expected_key is None:
        raise ValueError("owned staging identity is unavailable")
    if not stat.S_ISDIR(expected_key[2]):
        raise ValueError("owned staging identity is not a directory")

    parent_anchor: _AnchoredDirectoryPath | None = None
    writer_lock: _WriterLock | None = None
    retirement: _RetirementNamespace | None = None
    staging_descriptor = -1
    retirement_name: str | None = None
    retirement_removed = False
    primary: BaseException | None = None
    try:
        parent_anchor = _AnchoredDirectoryPath.open(
            parent,
            "staging cleanup parent",
            create_final=False,
        )
        writer_lock = _WriterLock.acquire(parent_anchor)
        try:
            current = _directory_identity(
                os.stat(
                    staging.name,
                    dir_fd=parent_anchor.descriptor,
                    follow_symlinks=False,
                ),
                "owned staging directory",
            )
        except FileNotFoundError as exc:
            raise ValueError(
                "owned staging directory disappeared before cleanup"
            ) from exc
        if _stable_directory_key(current) != expected_key:
            raise ValueError("refusing to clean a replaced staging directory")
        staging_descriptor = os.open(
            staging.name,
            _DIRECTORY_FLAGS,
            dir_fd=parent_anchor.descriptor,
        )
        expected = _directory_identity(
            os.fstat(staging_descriptor),
            "owned staging directory",
        )
        if (
            _stable_directory_key(expected) != expected_key
            or _directory_identity(
                os.stat(
                    staging.name,
                    dir_fd=parent_anchor.descriptor,
                    follow_symlinks=False,
                ),
                "owned staging directory",
            )
            != expected
        ):
            raise ValueError(
                "owned staging directory changed before quarantine"
            )

        retirement_directory = _create_private_child_directory(
            parent_anchor.descriptor,
            _RETIREMENT_NAMESPACE_PREFIX,
            "staging cleanup quarantine",
            lambda: (
                writer_lock is not None
                and _lock_is_verified(writer_lock, "staging cleanup")
            ),
            None,
        )
        retirement = _RetirementNamespace(
            retirement_directory,
            writer_lock,
        )
        retirement.verify_clean()
        retirement_name = _RETIREMENT_PREFIX + secrets.token_hex(12)
        rename_noreplace_at(
            parent_anchor.descriptor,
            staging.name,
            retirement.descriptor,
            retirement_name,
        )
        moved = _entry_at(retirement.descriptor, retirement_name)
        if moved != expected:
            # The source name was swapped at the rename boundary. Restore
            # the unowned object without following or deleting it.
            rename_noreplace_at(
                retirement.descriptor,
                retirement_name,
                parent_anchor.descriptor,
                staging.name,
            )
            retirement_name = None
            raise ValueError(
                "owned staging directory changed while entering quarantine"
            )
        if (
            _directory_identity(
                os.fstat(staging_descriptor),
                "owned staging directory",
            )
            != expected
        ):
            raise ValueError(
                "owned staging descriptor changed while entering quarantine"
            )
        _remove_private_quarantined_entry(
            retirement.descriptor,
            retirement_name,
            expected,
        )
        retirement_name = None
        retirement.remove()
        retirement_removed = True
    except BaseException as exc:
        primary = exc
    finally:
        failures = _CleanupFailures()
        if staging_descriptor >= 0:
            failures.attempt(
                "staging cleanup descriptor close",
                lambda: os.close(staging_descriptor),
            )
        if retirement is not None and not retirement_removed:
            failures.attempt(
                "staging cleanup quarantine removal",
                retirement.remove,
            )
        if retirement is not None:
            failures.attempt(
                "staging cleanup quarantine close",
                retirement.close,
            )
        if writer_lock is not None:
            failures.attempt("staging cleanup lock release", writer_lock.release)
            failures.attempt("staging cleanup lock close", writer_lock.close)
        if parent_anchor is not None:
            failures.attempt(
                "staging cleanup parent close",
                parent_anchor.close,
            )
        failures.finish(primary, cleanup_context)
    if primary is not None:
        raise primary


def _lock_is_verified(lock: _WriterLock, context: str) -> bool:
    try:
        lock.verify(context)
    except (OSError, ValueError):
        return False
    return True


def _stable_directory_key(identity: _EntryIdentity) -> tuple[int, int, int]:
    return identity.device, identity.inode, stat.S_IFMT(identity.mode)


def _remove_private_quarantined_entry(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
) -> None:
    """Remove an identity entirely within a transaction-private namespace."""
    current = _entry_at(parent_fd, name)
    if current != expected:
        raise ValueError("quarantined staging entry changed before cleanup")
    if stat.S_ISDIR(expected.mode):
        descriptor = -1
        primary: BaseException | None = None
        try:
            descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
            if (
                _EntryIdentity.from_stat(os.fstat(descriptor)) != expected
                or _entry_at(parent_fd, name) != expected
            ):
                raise ValueError("quarantined staging directory changed")
            for child_name in sorted(os.listdir(descriptor)):
                child = _entry_at(descriptor, child_name)
                if child is None:
                    raise ValueError("quarantined staging child changed")
                _remove_private_quarantined_entry(
                    descriptor,
                    child_name,
                    child,
                )
            if (
                _EntryIdentity.from_stat(os.fstat(descriptor)) != expected
                or _entry_at(parent_fd, name) != expected
            ):
                raise ValueError("quarantined staging directory changed")
        except BaseException as exc:
            primary = exc
        finally:
            failures = _CleanupFailures()
            if descriptor >= 0:
                failures.attempt(
                    "quarantined directory close",
                    lambda: os.close(descriptor),
                )
            failures.finish(primary, "quarantined staging tree")
        if primary is not None:
            raise primary
        os.rmdir(name, dir_fd=parent_fd)
    else:
        os.unlink(name, dir_fd=parent_fd)
    if _entry_at(parent_fd, name) is not None:
        raise ValueError("quarantined staging entry removal was incomplete")


class WriterPublication:
    """Own one locked writer bundle and finalize every cleanup outcome.

    Every supported staging mutation is serialized by the anchored parent
    lock.  Owned entries retire only after a no-replace move into this
    transaction's private sibling quarantine.
    """

    def __init__(
        self,
        staging: Path,
        source_tree: SecureTree,
        bundle_prefix: str,
        context: str,
        bundle_context: str | None = None,
    ) -> None:
        self.staging_path = Path(staging)
        self.source_tree = source_tree
        self.bundle_prefix = bundle_prefix
        self.context = context
        self.bundle_context = bundle_context or f"{context} bundle"
        self.registry = _OwnershipRegistry()
        self.source_anchor: _AnchoredDirectoryPath | None = None
        self.staging_parent_anchor: _AnchoredDirectoryPath | None = None
        self.writer_lock: _WriterLock | None = None
        self.retirement_namespace: _RetirementNamespace | None = None
        self.staging_anchor: _AnchoredDirectoryPath | None = None
        self.bundle: _ChildDirectory | None = None
        self._children: list[_ChildDirectory] = []
        self._files: list[OwnedFile] = []
        self._close_actions: list[tuple[str, Callable[[], None]]] = []
        self._finished = False

    @classmethod
    def open(
        cls,
        staging: Path,
        source_tree: SecureTree,
        bundle_prefix: str,
        context: str,
        bundle_context: str | None = None,
    ) -> "WriterPublication":
        publication = cls(
            staging,
            source_tree,
            bundle_prefix,
            context,
            bundle_context,
        )
        try:
            publication.source_anchor = _AnchoredDirectoryPath.open(
                source_tree.path,
                "source path",
                create_final=False,
            )
            if (
                publication.source_anchor.identity.directory_key
                != source_tree.directory_identity
            ):
                raise ValueError("source path changed before staging publication")
            publication.staging_parent_anchor = _AnchoredDirectoryPath.open(
                publication.staging_path.parent,
                "staging parent path",
                create_final=False,
            )
            publication.writer_lock = _WriterLock.acquire(
                publication.staging_parent_anchor
            )
            retirement_directory = _create_private_child_directory(
                publication.staging_parent_anchor.descriptor,
                _RETIREMENT_NAMESPACE_PREFIX,
                "writer retirement quarantine",
                publication.staging_parent_is_locked,
                None,
            )
            publication.retirement_namespace = _RetirementNamespace(
                retirement_directory,
                publication.writer_lock,
            )
            publication.retirement_namespace.verify_clean()
            publication.registry.bind_retirement_namespace(
                publication.retirement_namespace
            )
            publication.writer_lock.verify("staging path creation")
            publication.staging_anchor = _AnchoredDirectoryPath.open(
                publication.staging_path,
                "staging path",
                create_final=True,
                registry=publication.registry,
            )
            publication.writer_lock.verify("staging path creation")
            if (
                publication.source_anchor.identity.directory_key
                in publication.staging_anchor.ancestry
                or publication.staging_anchor.identity.directory_key
                in publication.source_anchor.ancestry
            ):
                raise ValueError("staging and source directory identities overlap")
            publication.bundle = _create_private_child_directory(
                publication.staging_anchor.descriptor,
                bundle_prefix,
                publication.bundle_context,
                publication.staging_is_attached,
                publication.registry,
            )
            return publication
        except BaseException as exc:
            publication.finish(exc, committed=False)
            raise

    @property
    def staging(self) -> _AnchoredDirectoryPath:
        if self.staging_anchor is None:
            raise ValueError(f"closed {self.context} publication")
        return self.staging_anchor

    @property
    def private_bundle(self) -> _ChildDirectory:
        if self.bundle is None:
            raise ValueError(f"closed {self.context} publication")
        return self.bundle

    def staging_is_attached(self) -> bool:
        return (
            self.staging_parent_is_locked()
            and self.staging_anchor is not None
            and self.staging_anchor.is_attached()
        )

    def staging_parent_is_locked(self) -> bool:
        try:
            if self.writer_lock is None:
                return False
            self.writer_lock.verify("writer transaction")
        except (OSError, ValueError):
            return False
        return True

    def bundle_is_attached(self) -> bool:
        return (
            self.staging_is_attached()
            and self.bundle is not None
            and self.bundle.is_attached()
        )

    def track(self, directory: _ChildDirectory) -> _ChildDirectory:
        if directory is self.bundle or directory in self._children:
            raise ValueError("writer publication directory is already tracked")
        self._children.append(directory)
        return directory

    def open_or_create_staging_directory(
        self,
        name: str,
        context: str,
    ) -> _ChildDirectory:
        return self.track(
            _open_or_create_child_directory(
                self.staging.descriptor,
                name,
                context,
                self.staging_is_attached,
                self.registry,
            )
        )

    def create_directory(
        self,
        parent: _ChildDirectory,
        name: str,
        context: str,
        parent_is_attached: Callable[[], bool],
    ) -> _ChildDirectory:
        return self.track(
            _create_published_child_directory(
                parent.descriptor,
                name,
                context,
                parent_is_attached,
                self.registry,
            )
        )

    def publish(
        self,
        source_parent_fd: int,
        source_name: str,
        expected: _EntryIdentity,
        destination_parent_fd: int,
        destination_name: str,
        destination_is_attached: Callable[[], bool],
        context: str,
    ) -> None:
        if not destination_is_attached():
            raise ValueError(f"{context} parent changed before publication")
        rename_noreplace_at(
            source_parent_fd,
            source_name,
            destination_parent_fd,
            destination_name,
        )
        if (
            not destination_is_attached()
            or _entry_at(destination_parent_fd, destination_name) != expected
        ):
            raise ValueError(f"{context} changed during publication")

    def create_file(
        self,
        parent: _ChildDirectory,
        name: str,
        context: str,
        parent_is_attached: Callable[[], bool],
    ) -> OwnedFile:
        output = _create_owned_file_at(
            parent.descriptor,
            name,
            context,
            parent_is_attached,
            self.registry,
        )
        self._files.append(output)
        return output

    def add_close_action(
        self,
        context: str,
        action: Callable[[], None],
    ) -> None:
        """Make an external descriptor closure part of transaction success."""
        if self._finished:
            raise ValueError(f"{self.context} publication was already finalized")
        if not context or not callable(action):
            raise TypeError("close action requires a context and callable")
        self._close_actions.append((context, action))

    def finish(
        self,
        primary: BaseException | None,
        *,
        committed: bool,
    ) -> None:
        if self._finished:
            raise ValueError(f"{self.context} publication was already finalized")
        self._finished = True
        failures = _CleanupFailures()

        for output in reversed(self._files):
            failures.attempt(f"{output.context} close", output.close)
        self._files.clear()
        for child in reversed(self._children):
            failures.attempt(f"{child.context} close", child.close)
        self._children.clear()
        if self.bundle is not None:
            failures.attempt(f"{self.context} bundle close", self.bundle.close)
        for context, action in self._close_actions:
            failures.attempt(context, action)
        self._close_actions.clear()

        if (
            committed
            and not failures.errors
            and self.bundle is not None
            and self.staging_anchor is not None
        ):
            failures.attempt(
                f"{self.context} private bundle retirement",
                lambda: _remove_owned_entry_at(
                    self.staging_anchor.descriptor,
                    self.bundle.name,
                    self.bundle.identity,
                    self.registry,
                    self.staging_is_attached,
                ),
            )
            if (
                not failures.errors
                and self.registry.is_owned(self.bundle.identity)
            ):
                failures.errors.append(
                    (
                        f"{self.context} private bundle retirement",
                        ValueError("private writer bundle retirement was incomplete"),
                    )
                )

        if self.retirement_namespace is not None:
            failures.attempt(
                "retirement quarantine topology verification",
                self.retirement_namespace.verify_clean,
            )

        if self.source_anchor is not None:
            failures.attempt("source anchor close", self.source_anchor.close)

        if (
            self.retirement_namespace is not None
            and not self.retirement_namespace.is_attached()
        ):
            failures.attempt(
                "detached retirement quarantine descriptor close",
                self.retirement_namespace.close,
            )
            self._create_recovery_retirement_namespace(failures)

        rollback_required = not committed or bool(failures.errors)
        if (
            rollback_required
            and self.staging_anchor is not None
            and self.staging_is_attached()
        ):
            failures.attempt(
                f"{self.context} rollback",
                lambda: _remove_owned_identities_below(
                    self.staging_anchor.descriptor,
                    self.registry,
                    self.staging_is_attached,
                ),
            )
        if rollback_required and self.staging_anchor is not None:
            failures.attempt(
                "created staging root removal",
                lambda: self.staging_anchor.remove_created_final(self.registry),
            )
        if rollback_required and self.registry.has_active():
            failures.errors.append(
                (
                    f"{self.context} rollback",
                    ValueError("writer rollback left owned filesystem identities active"),
                )
            )

        staging_identity = (
            self.staging_anchor.identity
            if self.staging_anchor is not None
            else None
        )
        before_staging_close = len(failures.errors)
        if self.staging_anchor is not None:
            failures.attempt("staging anchor close", self.staging_anchor.close)
        if (
            committed
            and len(failures.errors) > before_staging_close
            and staging_identity is not None
        ):
            self._recover_after_final_anchor_close(
                staging_identity,
                failures,
            )

        before_quarantine_removal = len(failures.errors)
        if self.retirement_namespace is not None:
            failures.attempt(
                "retirement quarantine removal",
                self.retirement_namespace.remove,
            )
        if (
            committed
            and len(failures.errors) > before_quarantine_removal
            and self.registry.has_active()
            and staging_identity is not None
        ):
            if (
                self.retirement_namespace is not None
                and not self.retirement_namespace.is_attached()
            ):
                failures.attempt(
                    "removed retirement quarantine descriptor close",
                    self.retirement_namespace.close,
                )
                self._create_recovery_retirement_namespace(failures)
            self._recover_after_final_anchor_close(
                staging_identity,
                failures,
            )
            if self.retirement_namespace is not None:
                failures.attempt(
                    "retirement quarantine removal after rollback",
                    self.retirement_namespace.remove,
                )
        closing_retirement_namespace = self.retirement_namespace
        before_quarantine_close = len(failures.errors)
        if self.retirement_namespace is not None:
            failures.attempt(
                "retirement quarantine descriptor close",
                self.retirement_namespace.close,
            )
        if (
            committed
            and len(failures.errors) > before_quarantine_close
            and self.registry.has_active()
            and staging_identity is not None
        ):
            if closing_retirement_namespace is not None:
                failures.attempt(
                    "failed retirement quarantine descriptor retry close",
                    closing_retirement_namespace.close,
                )
            self._create_recovery_retirement_namespace(failures)
            self._recover_after_final_anchor_close(
                staging_identity,
                failures,
            )
            if self.retirement_namespace is not None:
                failures.attempt(
                    "quarantine-close recovery namespace removal",
                    self.retirement_namespace.remove,
                )
                failures.attempt(
                    "quarantine-close recovery descriptor close",
                    self.retirement_namespace.close,
                )

        staging_parent_identity = (
            self.staging_parent_anchor.identity
            if self.staging_parent_anchor is not None
            else None
        )
        before_parent_close = len(failures.errors)
        if self.staging_parent_anchor is not None:
            failures.attempt(
                "staging parent anchor close",
                self.staging_parent_anchor.close,
            )
        if (
            committed
            and len(failures.errors) > before_parent_close
            and self.registry.has_active()
            and staging_identity is not None
            and staging_parent_identity is not None
        ):
            self._recover_after_parent_anchor_close(
                staging_parent_identity,
                staging_identity,
                failures,
            )

        before_lock_release = len(failures.errors)
        if self.writer_lock is not None:
            failures.attempt("writer lock release", self.writer_lock.release)
        if (
            committed
            and len(failures.errors) > before_lock_release
            and self.registry.has_active()
            and staging_identity is not None
            and staging_parent_identity is not None
        ):
            self._recover_after_lock_release(
                staging_parent_identity,
                staging_identity,
                failures,
            )

        before_lock_close = len(failures.errors)
        if self.writer_lock is not None:
            failures.attempt("writer lock close", self.writer_lock.close)
        if (
            committed
            and len(failures.errors) > before_lock_close
            and self.registry.has_active()
            and staging_identity is not None
            and staging_parent_identity is not None
        ):
            self._recover_after_lock_close(
                staging_parent_identity,
                staging_identity,
                failures,
            )

        if primary is None and not failures.errors:
            self.registry.release_all()
        self.bundle = None
        self.source_anchor = None
        self.writer_lock = None
        self.retirement_namespace = None
        self.staging_parent_anchor = None
        self.staging_anchor = None
        failures.finish(primary, self.context)

    def _recover_after_lock_release(
        self,
        staging_parent_identity: _EntryIdentity,
        staging_identity: _EntryIdentity,
        failures: _CleanupFailures,
    ) -> None:
        failed_parent = self.staging_parent_anchor
        recovery_parent: _AnchoredDirectoryPath | None = None
        try:
            recovery_parent = _AnchoredDirectoryPath.open(
                self.staging_path.parent,
                "writer lock release recovery parent",
                create_final=False,
            )
            if recovery_parent.identity != staging_parent_identity:
                raise ValueError(
                    "staging parent changed before lock-release recovery"
                )
            if self.writer_lock is None:
                raise ValueError("writer lock is unavailable for release recovery")
            self.writer_lock.rebind_parent(
                recovery_parent,
                "writer lock release recovery",
            )
            self.staging_parent_anchor = recovery_parent
            self.writer_lock.reacquire()
            self._create_recovery_retirement_namespace(failures)
            self._recover_after_final_anchor_close(staging_identity, failures)
            if self.retirement_namespace is not None:
                failures.attempt(
                    "lock-release recovery retirement quarantine removal",
                    self.retirement_namespace.remove,
                )
                failures.attempt(
                    "lock-release recovery quarantine descriptor close",
                    self.retirement_namespace.close,
                )
        except BaseException as exc:
            failures.errors.append(("writer lock release recovery", exc))
        finally:
            if recovery_parent is not None:
                failures.attempt(
                    "writer lock release recovery parent close",
                    recovery_parent.close,
                )
            if failed_parent is not None and failed_parent is not recovery_parent:
                failures.attempt(
                    "released writer lock parent retry close",
                    failed_parent.close,
                )

    def _recover_after_lock_close(
        self,
        staging_parent_identity: _EntryIdentity,
        staging_identity: _EntryIdentity,
        failures: _CleanupFailures,
    ) -> None:
        failed_parent = self.staging_parent_anchor
        recovery_parent: _AnchoredDirectoryPath | None = None
        recovery_lock: _WriterLock | None = None
        try:
            recovery_parent = _AnchoredDirectoryPath.open(
                self.staging_path.parent,
                "writer lock close recovery parent",
                create_final=False,
            )
            if recovery_parent.identity != staging_parent_identity:
                raise ValueError(
                    "staging parent changed before lock-close recovery"
                )
            current = self.writer_lock
            if current is not None and current.descriptor >= 0:
                current.rebind_parent(
                    recovery_parent,
                    "writer lock close recovery",
                )
                current.reacquire()
                recovery_lock = current
            else:
                recovery_lock = _WriterLock.acquire(recovery_parent)
            self.staging_parent_anchor = recovery_parent
            self.writer_lock = recovery_lock
            self._create_recovery_retirement_namespace(failures)
            self._recover_after_final_anchor_close(staging_identity, failures)
            if self.retirement_namespace is not None:
                failures.attempt(
                    "lock-close recovery retirement quarantine removal",
                    self.retirement_namespace.remove,
                )
                failures.attempt(
                    "lock-close recovery quarantine descriptor close",
                    self.retirement_namespace.close,
                )
        except BaseException as exc:
            failures.errors.append(("writer lock close recovery", exc))
        finally:
            if recovery_lock is not None:
                failures.attempt(
                    "lock-close recovery writer lock close",
                    recovery_lock.close,
                )
            if recovery_parent is not None:
                failures.attempt(
                    "writer lock close recovery parent close",
                    recovery_parent.close,
                )
            if failed_parent is not None and failed_parent is not recovery_parent:
                failures.attempt(
                    "closed writer lock parent retry close",
                    failed_parent.close,
                )

    def _recover_after_parent_anchor_close(
        self,
        staging_parent_identity: _EntryIdentity,
        staging_identity: _EntryIdentity,
        failures: _CleanupFailures,
    ) -> None:
        failed_parent = self.staging_parent_anchor
        recovery_parent: _AnchoredDirectoryPath | None = None
        recovery_namespace: _RetirementNamespace | None = None
        try:
            recovery_parent = _AnchoredDirectoryPath.open(
                self.staging_path.parent,
                "staging parent rollback path",
                create_final=False,
            )
            if recovery_parent.identity != staging_parent_identity:
                raise ValueError(
                    "staging parent changed before close-failure recovery"
                )
            if self.writer_lock is None:
                raise ValueError("writer lock is unavailable for parent recovery")
            self.writer_lock.rebind_parent(
                recovery_parent,
                "staging parent close recovery",
            )
            self.staging_parent_anchor = recovery_parent
            directory = _create_private_child_directory(
                recovery_parent.descriptor,
                _RETIREMENT_NAMESPACE_PREFIX,
                "parent-close recovery retirement quarantine",
                self.staging_parent_is_locked,
                None,
            )
            recovery_namespace = _RetirementNamespace(
                directory,
                self.writer_lock,
            )
            recovery_namespace.verify_clean()
            self.registry.replace_retirement_namespace(recovery_namespace)
            self.retirement_namespace = recovery_namespace
            self._recover_after_final_anchor_close(staging_identity, failures)
            failures.attempt(
                "parent-close recovery retirement quarantine removal",
                recovery_namespace.remove,
            )
        except BaseException as exc:
            failures.errors.append(("staging parent close recovery", exc))
        finally:
            if recovery_namespace is not None:
                failures.attempt(
                    "parent-close recovery quarantine descriptor close",
                    recovery_namespace.close,
                )
            if recovery_parent is not None:
                failures.attempt(
                    "staging parent recovery anchor close",
                    recovery_parent.close,
                )
            if failed_parent is not None and failed_parent is not recovery_parent:
                failures.attempt(
                    "failed staging parent anchor retry close",
                    failed_parent.close,
                )

    def _create_recovery_retirement_namespace(
        self,
        failures: _CleanupFailures,
    ) -> None:
        if self.staging_parent_anchor is None or self.writer_lock is None:
            failures.errors.append(
                (
                    "recovery retirement quarantine creation",
                    ValueError("staging parent writer lock is unavailable"),
                )
            )
            return
        try:
            directory = _create_private_child_directory(
                self.staging_parent_anchor.descriptor,
                _RETIREMENT_NAMESPACE_PREFIX,
                "recovery retirement quarantine",
                self.staging_parent_is_locked,
                None,
            )
            namespace = _RetirementNamespace(directory, self.writer_lock)
            namespace.verify_clean()
            self.registry.replace_retirement_namespace(namespace)
            self.retirement_namespace = namespace
        except BaseException as exc:
            failures.errors.append(
                ("recovery retirement quarantine creation", exc)
            )

    def _recover_after_final_anchor_close(
        self,
        staging_identity: _EntryIdentity,
        failures: _CleanupFailures,
    ) -> None:
        recovery: _AnchoredDirectoryPath | None = None
        try:
            if not self.staging_parent_is_locked():
                raise ValueError("staging parent writer lock was lost before recovery")
            recovery = _AnchoredDirectoryPath.open(
                self.staging_path,
                "staging rollback path",
                create_final=False,
            )
            if recovery.identity != staging_identity:
                raise ValueError("staging path changed before rollback recovery")
            failures.attempt(
                "publication rollback after final anchor close failure",
                lambda: _remove_owned_identities_below(
                    recovery.descriptor,
                    self.registry,
                    recovery.is_attached,
                ),
            )
            failures.attempt(
                "created staging root removal after final close failure",
                lambda: recovery.remove_final_if_owned(self.registry),
            )
            if self.registry.has_active():
                failures.errors.append(
                    (
                        "staging rollback recovery",
                        ValueError(
                            "writer rollback left owned filesystem identities active"
                        ),
                    )
                )
        except BaseException as exc:
            failures.errors.append(("staging rollback recovery", exc))
        finally:
            if recovery is not None:
                failures.attempt("staging rollback anchor close", recovery.close)
