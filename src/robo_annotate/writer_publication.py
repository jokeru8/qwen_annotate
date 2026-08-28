"""Version-neutral, descriptor-anchored writer publication primitives."""

from __future__ import annotations

import ctypes
import errno
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
_RETIREMENT_PREFIX = ".writer-retire-"
_IN_ATTRIB = 0x00000004
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
    | _IN_ONLYDIR
)
_RETIREMENT_WATCH_MASK = _BIRTH_WATCH_MASK | _IN_ATTRIB | _IN_UNMOUNT
_AT_REMOVEDIR = 0x200


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


class _RetirementEventWitness(_DirectoryBirthWitness):
    """Prove one quarantine move and one unambiguous retirement delete."""

    _WATCH_MASK = _RETIREMENT_WATCH_MASK

    def verify_quarantine_move(
        self,
        source_name: str,
        quarantine_name: str,
        *,
        is_directory: bool,
    ) -> None:
        self._events.extend(self._read_events())
        source = os.fsencode(source_name)
        quarantine = os.fsencode(quarantine_name)
        relevant = [
            event
            for event in self._events
            if event[2] in (source, quarantine)
        ]
        kind = _IN_ISDIR if is_directory else 0
        valid = (
            len(relevant) == 2
            and relevant[0][0] == (_IN_MOVED_FROM | kind)
            and relevant[0][2] == source
            and relevant[0][1] != 0
            and relevant[1][0] == (_IN_MOVED_TO | kind)
            and relevant[1][2] == quarantine
            and relevant[1][1] == relevant[0][1]
        )
        if self._has_global_failure(self._events) or not valid:
            raise ValueError("owned entry quarantine event history is ambiguous")
        self._events.clear()

    def verify_unchanged(self, quarantine_name: str) -> None:
        self._events.extend(self._read_events())
        quarantine = os.fsencode(quarantine_name)
        if self._has_global_failure(self._events) or any(
            event_name == quarantine for _, _, event_name in self._events
        ):
            raise ValueError("owned entry changed before retirement removal")
        self._events.clear()

    def verify_deleted(
        self,
        quarantine_name: str,
        *,
        is_directory: bool,
    ) -> None:
        self._events.extend(self._read_events())
        quarantine = os.fsencode(quarantine_name)
        relevant = [
            mask
            for mask, _, event_name in self._events
            if event_name == quarantine
        ]
        kind = _IN_ISDIR if is_directory else 0
        if (
            self._has_global_failure(self._events)
            or relevant != [(_IN_DELETE | kind)]
        ):
            raise ValueError("owned entry retirement event history is ambiguous")
        self._events.clear()


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
    registry: _OwnershipRegistry,
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
            registry.register(expected)
            registered = True
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
            if expected is not None and registered:
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


def _raw_unlinkat(
    parent_fd: int,
    name: str,
    *,
    is_directory: bool,
    witness: _RetirementEventWitness,
) -> None:
    """Drain the boundary witness, then immediately call Linux unlinkat."""
    libc = ctypes.CDLL(None, use_errno=True)
    unlinkat = getattr(libc, "unlinkat", None)
    if unlinkat is None:
        raise ValueError("owned retirement requires Linux unlinkat")
    unlinkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    unlinkat.restype = ctypes.c_int
    encoded_name = os.fsencode(name)
    flags = _AT_REMOVEDIR if is_directory else 0
    witness.verify_unchanged(name)
    if unlinkat(parent_fd, encoded_name, flags) == 0:
        return
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), name)


def _remove_pinned_quarantine_entry_at(
    parent_fd: int,
    quarantine: str,
    expected: _EntryIdentity,
    retirement_descriptor: int,
    witness: _RetirementEventWitness,
    still_attached: Callable[[], bool],
) -> None:
    """Apply the Linux event invariant at the compare-to-unlink boundary.

    Linux has no compare-and-unlink syscall.  Safety therefore requires an
    inode-bound parent watch, a final no-follow name comparison, a clean event
    drain immediately followed by one raw unlinkat, exactly one matching
    delete event, and the pinned inode's own link-count transition to zero.
    A regular file must have exactly one link before removal, preventing an
    external unlink from helping produce that transition.  Any ambiguity is
    a transaction failure.
    """
    is_directory = stat.S_ISDIR(expected.mode)
    if not still_attached():
        raise ValueError("owned entry parent changed during retirement")
    current = _entry_at(parent_fd, quarantine)
    if current != expected or not still_attached():
        raise ValueError("owned entry changed before retirement removal")
    before = os.fstat(retirement_descriptor)
    if _EntryIdentity.from_stat(before) != expected or before.st_nlink <= 0:
        raise ValueError("owned entry changed before retirement removal")
    if not is_directory and before.st_nlink != 1:
        raise ValueError("owned file has an ambiguous retirement link count")
    _raw_unlinkat(
        parent_fd,
        quarantine,
        is_directory=is_directory,
        witness=witness,
    )
    witness.verify_deleted(quarantine, is_directory=is_directory)
    after = os.fstat(retirement_descriptor)
    if _EntryIdentity.from_stat(after) != expected or after.st_nlink != 0:
        raise ValueError("owned entry retirement was incomplete")
    if not still_attached():
        raise ValueError("owned entry parent changed during retirement")
    if _entry_at(parent_fd, quarantine) is not None:
        raise ValueError("owned entry quarantine was reused during retirement")


def _retire_and_delete_owned_entry_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> bool:
    """Quarantine one name atomically, then delete only its owned identity."""
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return False
    if not registry.begin_retirement(expected):
        return False
    quarantine: str | None = None
    retirement_descriptor = -1
    retirement_witness: _RetirementEventWitness | None = None
    retired = False
    primary_error: BaseException | None = None
    try:
        if not still_attached() or _entry_at(parent_fd, name) != expected:
            registry.restore(expected)
            return False
        retirement_witness = _RetirementEventWitness.open(
            parent_fd,
            "owned entry retirement",
        )
        if not still_attached():
            raise ValueError("owned entry parent changed during retirement")
        for _ in range(100):
            candidate = _RETIREMENT_PREFIX + secrets.token_hex(12)
            if not still_attached():
                raise ValueError("owned entry parent changed during retirement")
            try:
                rename_noreplace_at(
                    parent_fd,
                    name,
                    parent_fd,
                    candidate,
                )
            except FileExistsError:
                continue
            quarantine = candidate
            break
        if quarantine is None:
            raise ValueError("unable to reserve an owned entry quarantine")
        if not still_attached():
            raise ValueError("owned entry parent changed during retirement")
        flags = _DIRECTORY_FLAGS if stat.S_ISDIR(expected.mode) else (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        retirement_descriptor = os.open(
            quarantine,
            flags,
            dir_fd=parent_fd,
        )
        pinned = _EntryIdentity.from_stat(os.fstat(retirement_descriptor))
        quarantined = _entry_at(parent_fd, quarantine)
        if pinned != expected or quarantined != expected:
            error = ValueError("owned entry changed during retirement")
            try:
                _restore_quarantined_entry_at(
                    parent_fd,
                    quarantine,
                    name,
                    still_attached,
                )
            except BaseException as restore_error:
                error.add_note(
                    "Secondary quarantine restore failure: "
                    f"{type(restore_error).__name__}: {restore_error}"
                )
            raise error
        retirement_witness.verify_quarantine_move(
            name,
            quarantine,
            is_directory=stat.S_ISDIR(expected.mode),
        )
        _remove_pinned_quarantine_entry_at(
            parent_fd,
            quarantine,
            expected,
            retirement_descriptor,
            retirement_witness,
            still_attached,
        )
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
    if retirement_witness is not None:
        try:
            retirement_witness.close()
        except BaseException as close_error:
            if primary_error is None:
                primary_error = close_error
            else:
                primary_error.add_note(
                    "Secondary retirement witness close failure: "
                    f"{type(close_error).__name__}: {close_error}"
                )

    if retired:
        registry.finish_retirement(expected)
    else:
        registry.restore(expected)
        quarantined_after_failure: _EntryIdentity | None = None
        if quarantine is not None:
            try:
                quarantined_after_failure = _entry_at(parent_fd, quarantine)
            except BaseException as inspection_error:
                if primary_error is None:
                    primary_error = inspection_error
                else:
                    primary_error.add_note(
                        "Secondary owned quarantine inspection failure: "
                        f"{type(inspection_error).__name__}: {inspection_error}"
                    )
        if quarantined_after_failure is not None:
            assert quarantine is not None
            try:
                _restore_quarantined_entry_at(
                    parent_fd,
                    quarantine,
                    name,
                    still_attached,
                )
            except BaseException as restore_error:
                assert primary_error is not None
                primary_error.add_note(
                    "Secondary owned quarantine restore failure: "
                    f"{type(restore_error).__name__}: {restore_error}"
                )
    if primary_error is not None:
        raise primary_error
    return True


def _restore_quarantined_entry_at(
    parent_fd: int,
    quarantine: str,
    original_name: str,
    still_attached: Callable[[], bool],
) -> None:
    """Restore a quarantined entry without replacing any current original."""
    if not still_attached():
        raise ValueError("owned entry parent changed before quarantine restore")
    if _entry_at(parent_fd, quarantine) is None:
        return
    if not still_attached():
        raise ValueError("owned entry parent changed before quarantine restore")
    rename_noreplace_at(
        parent_fd,
        quarantine,
        parent_fd,
        original_name,
    )


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


class WriterPublication:
    """Own one atomic writer bundle and finalize every cleanup outcome."""

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
            publication.staging_anchor = _AnchoredDirectoryPath.open(
                publication.staging_path,
                "staging path",
                create_final=True,
                registry=publication.registry,
            )
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
        return self.staging_anchor is not None and self.staging_anchor.is_attached()

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

        if self.source_anchor is not None:
            failures.attempt("source anchor close", self.source_anchor.close)

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

        if primary is None and not failures.errors:
            self.registry.release_all()
        self.bundle = None
        self.source_anchor = None
        self.staging_anchor = None
        failures.finish(primary, self.context)

    def _recover_after_final_anchor_close(
        self,
        staging_identity: _EntryIdentity,
        failures: _CleanupFailures,
    ) -> None:
        recovery: _AnchoredDirectoryPath | None = None
        try:
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
