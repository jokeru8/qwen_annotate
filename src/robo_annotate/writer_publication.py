"""Version-neutral, descriptor-anchored writer publication primitives."""

from __future__ import annotations

import os
import secrets
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .secure_tree import SecureTree, rename_noreplace_at


_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_PRIVATE_DIRECTORY_PREFIX = ".writer-dir-"


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
        error = ValueError(f"{context} cleanup failed: {first_error}")
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
    """Create an unpredictable child and pin it before observing its pathname."""
    for _ in range(100):
        name = prefix + secrets.token_hex(12)
        if not parent_is_attached():
            raise ValueError(f"{context} parent changed before creation")
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        expected: _EntryIdentity | None = None
        descriptor = -1
        validation_descriptor = -1
        try:
            # The bytes-path open is the first operation after mkdir and pins
            # the just-created object.  All ownership facts come from fstat;
            # the pathname is only compared with that authoritative identity.
            descriptor = os.open(
                os.fsencode(name),
                _DIRECTORY_FLAGS,
                dir_fd=parent_fd,
            )
            expected = _directory_identity(os.fstat(descriptor), context)
            registry.register(expected)
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
            if expected is not None:
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



def _retire_and_delete_owned_entry_at(
    parent_fd: int,
    name: str,
    expected: _EntryIdentity,
    registry: _OwnershipRegistry,
    still_attached: Callable[[], bool],
) -> bool:
    """Make an identity unmatchable before its deleting syscall can free it."""
    if not still_attached() or _entry_at(parent_fd, name) != expected:
        return False
    if not registry.begin_retirement(expected):
        return False
    try:
        if not still_attached() or _entry_at(parent_fd, name) != expected:
            registry.restore(expected)
            return False
        if stat.S_ISDIR(expected.mode):
            os.rmdir(name, dir_fd=parent_fd)
        else:
            os.unlink(name, dir_fd=parent_fd)
    except BaseException:
        registry.restore(expected)
        raise
    registry.finish_retirement(expected)
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

        for child in reversed(self._children):
            failures.attempt(f"{child.context} close", child.close)
        self._children.clear()
        if self.bundle is not None:
            failures.attempt(f"{self.context} bundle close", self.bundle.close)

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
        if rollback_required and self.registry.has_active():
            failures.errors.append(
                (
                    f"{self.context} rollback",
                    ValueError("writer rollback left owned filesystem identities active"),
                )
            )

        if rollback_required and self.staging_anchor is not None:
            failures.attempt(
                "created staging root removal",
                lambda: self.staging_anchor.remove_created_final(self.registry),
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
        except BaseException as exc:
            failures.errors.append(("staging rollback recovery", exc))
        finally:
            if recovery is not None:
                failures.attempt("staging rollback anchor close", recovery.close)
