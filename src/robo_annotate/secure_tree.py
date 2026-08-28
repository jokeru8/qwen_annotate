"""Descriptor-anchored, no-follow reads for untrusted dataset trees."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator


_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
_READ_BLOCK = 1024 * 1024


@dataclass(frozen=True)
class _Identity:
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int
    changed_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "_Identity":
        return cls(
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )


class SecureFile:
    """A stable regular-file descriptor tied to its scanned directory entry."""

    def __init__(
        self,
        tree: "SecureTree",
        relative: str,
        parent_fd: int,
        name: str,
        fd: int,
        identity: _Identity,
        max_bytes: int,
        context: str,
    ) -> None:
        self.tree = tree
        self.relative = relative
        self._parent_fd = parent_fd
        self._name = name
        self._fd = fd
        self._identity = identity
        self._max_bytes = max_bytes
        self._context = context

    @property
    def display_path(self) -> Path:
        return self.tree.path / self.relative

    @property
    def proc_path(self) -> Path:
        if self._fd < 0:
            raise ValueError(f"closed secure file: {self.relative}")
        return Path(f"/proc/self/fd/{self._fd}")

    @property
    def size(self) -> int:
        return self._identity.size

    def verify(self) -> None:
        if self._fd < 0:
            raise ValueError(f"closed secure file: {self.relative}")
        try:
            descriptor = _Identity.from_stat(os.fstat(self._fd))
            entry = _Identity.from_stat(
                os.stat(self._name, dir_fd=self._parent_fd, follow_symlinks=False)
            )
        except OSError as exc:
            raise ValueError(f"{self._context} changed during validation: {self.relative}") from exc
        if descriptor != self._identity or entry != self._identity:
            raise ValueError(f"{self._context} changed during validation: {self.relative}")

    def read_bytes(self) -> bytes:
        self.verify()
        if self._identity.size > self._max_bytes:
            raise ValueError(
                f"{self._context} exceeds the {self._max_bytes}-byte validation limit: "
                f"{self.relative}"
            )
        chunks: list[bytes] = []
        offset = 0
        while offset < self._identity.size:
            try:
                chunk = os.pread(
                    self._fd,
                    min(_READ_BLOCK, self._identity.size - offset),
                    offset,
                )
            except OSError as exc:
                raise ValueError(f"unable to read {self._context}: {self.relative}") from exc
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        self.verify()
        if offset != self._identity.size:
            raise ValueError(f"short read for {self._context}: {self.relative}")
        return b"".join(chunks)

    def sha256(self) -> str:
        self.verify()
        digest = hashlib.sha256()
        offset = 0
        while offset < self._identity.size:
            try:
                block = os.pread(
                    self._fd,
                    min(_READ_BLOCK, self._identity.size - offset),
                    offset,
                )
            except OSError as exc:
                raise ValueError(f"unable to hash {self._context}: {self.relative}") from exc
            if not block:
                break
            digest.update(block)
            offset += len(block)
        self.verify()
        if offset != self._identity.size:
            raise ValueError(f"short read while hashing {self._context}: {self.relative}")
        return digest.hexdigest()

    def close(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1
        if self._parent_fd >= 0:
            os.close(self._parent_fd)
            self._parent_fd = -1

    def __enter__(self) -> "SecureFile":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SecureTree:
    """A scanned tree rooted at one verified, non-symlink directory descriptor."""

    def __init__(self, path: Path, label: str) -> None:
        if not isinstance(path, Path):
            raise TypeError(f"{label} path must be a Path")
        absolute = Path(os.path.abspath(path))
        if absolute == Path("/"):
            raise ValueError(f"{label} must be a real directory")
        current_fd = os.open("/", _DIRECTORY_FLAGS)
        root_fd = -1
        try:
            for position, component in enumerate(absolute.parts[1:]):
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise ValueError(f"{label} must be a real directory")
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
                if _Identity.from_stat(before) != _Identity.from_stat(os.fstat(child_fd)):
                    os.close(child_fd)
                    raise ValueError(f"{label} changed while opening")
                if position == len(absolute.parts[1:]) - 1:
                    root_fd = child_fd
                    break
                os.close(current_fd)
                current_fd = child_fd
        except OSError as exc:
            os.close(current_fd)
            raise ValueError(f"{label} must be a real directory") from exc
        except ValueError:
            os.close(current_fd)
            raise
        if root_fd < 0:
            os.close(current_fd)
            raise ValueError(f"{label} must be a real directory")
        after = os.fstat(root_fd)
        self.path = absolute
        self.label = label
        self._root_fd = root_fd
        self._root_parent_fd = current_fd
        self._root_name = absolute.name
        self._root_identity = _Identity.from_stat(after)
        self._entries: dict[str, _Identity] | None = None
        self._directories: dict[str, _Identity] | None = None
        self._open_files: list[SecureFile] = []

    def close(self) -> None:
        for item in self._open_files:
            item.close()
        self._open_files.clear()
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1
        if self._root_parent_fd >= 0:
            os.close(self._root_parent_fd)
            self._root_parent_fd = -1

    def __enter__(self) -> "SecureTree":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def directory_identity(self) -> tuple[int, int]:
        """Return the stable device/inode identity of the open root directory."""
        if self._root_fd < 0:
            raise ValueError(f"closed secure tree: {self.label}")
        current = _Identity.from_stat(os.fstat(self._root_fd))
        if current != self._root_identity:
            raise ValueError(f"{self.label} root changed during validation")
        return current.device, current.inode

    def scan(self) -> tuple[str, ...]:
        if self._entries is not None:
            return tuple(sorted(self._entries))
        entries: dict[str, _Identity] = {}
        directories: dict[str, _Identity] = {"": self._root_identity}

        def visit(directory_fd: int, prefix: PurePosixPath) -> None:
            try:
                names = sorted(os.listdir(directory_fd))
            except OSError as exc:
                raise ValueError(f"unable to enumerate {self.label} tree") from exc
            for name in names:
                if not name or "/" in name or name in (".", ".."):
                    raise ValueError(f"unsafe entry in {self.label} tree")
                relative_path = prefix / name
                relative = relative_path.as_posix()
                try:
                    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError as exc:
                    raise ValueError(f"unable to inspect {self.label} entry: {relative}") from exc
                identity = _Identity.from_stat(before)
                if stat.S_ISLNK(before.st_mode):
                    raise ValueError(f"{self.label} contains a symbolic link: {relative}")
                if stat.S_ISDIR(before.st_mode):
                    try:
                        child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                    except OSError as exc:
                        raise ValueError(f"unsafe {self.label} directory: {relative}") from exc
                    try:
                        if _Identity.from_stat(os.fstat(child_fd)) != identity:
                            raise ValueError(
                                f"{self.label} directory changed while opening: {relative}"
                            )
                        directories[relative] = identity
                        visit(child_fd, relative_path)
                        if _Identity.from_stat(os.fstat(child_fd)) != identity:
                            raise ValueError(
                                f"{self.label} directory changed during enumeration: {relative}"
                            )
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(before.st_mode):
                    entries[relative] = identity
                else:
                    raise ValueError(f"{self.label} contains an unsafe file type: {relative}")

        visit(self._root_fd, PurePosixPath())
        if _Identity.from_stat(os.fstat(self._root_fd)) != self._root_identity:
            raise ValueError(f"{self.label} root changed during enumeration")
        self._verify_root_entry()
        self._entries = entries
        self._directories = directories
        return tuple(sorted(entries))

    def files_under(self, prefix: str, suffix: str | None = None) -> tuple[str, ...]:
        normalized = _relative(prefix)
        start = normalized + "/" if normalized else ""
        return tuple(
            name
            for name in self.scan()
            if name.startswith(start) and (suffix is None or name.endswith(suffix))
        )

    def verify(self) -> None:
        """Prove every scanned entry still names the same object, without following links."""
        for relative in self.scan():
            with self.open_file(relative, max(1, self._entries[relative].size), "tree entry") as opened:
                opened.verify()
        if _Identity.from_stat(os.fstat(self._root_fd)) != self._root_identity:
            raise ValueError(f"{self.label} root changed during validation")
        self._verify_root_entry()

    def _verify_root_entry(self) -> None:
        try:
            current = os.stat(
                self._root_name,
                dir_fd=self._root_parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ValueError(f"{self.label} root changed during validation") from exc
        if _Identity.from_stat(current) != self._root_identity:
            raise ValueError(f"{self.label} root changed during validation")

    def open_file(self, relative: str | Path, max_bytes: int, context: str) -> SecureFile:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise TypeError("max_bytes must be a positive integer")
        name = _relative(relative)
        entries = self._entries
        directories = self._directories
        if entries is None or directories is None:
            self.scan()
            entries = self._entries
            directories = self._directories
        assert entries is not None and directories is not None
        expected = entries.get(name)
        if expected is None:
            raise ValueError(f"missing regular {context}: {name}")
        parts = PurePosixPath(name).parts
        parent_fd = os.dup(self._root_fd)
        prefix: list[str] = []
        try:
            for component in parts[:-1]:
                prefix.append(component)
                expected_directory = directories.get("/".join(prefix))
                if expected_directory is None:
                    raise ValueError(f"unsafe {context} path: {name}")
                before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                if _Identity.from_stat(before) != expected_directory:
                    raise ValueError(f"{context} path changed during validation: {name}")
                child_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
                if _Identity.from_stat(os.fstat(child_fd)) != expected_directory:
                    os.close(child_fd)
                    raise ValueError(f"{context} path changed during validation: {name}")
                os.close(parent_fd)
                parent_fd = child_fd
            current = os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
            if _Identity.from_stat(current) != expected:
                raise ValueError(f"{context} changed during validation: {name}")
            fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent_fd)
            if _Identity.from_stat(os.fstat(fd)) != expected:
                os.close(fd)
                raise ValueError(f"{context} changed while opening: {name}")
            if expected.size > max_bytes:
                os.close(fd)
                raise ValueError(
                    f"{context} exceeds the {max_bytes}-byte validation limit: {name}"
                )
            opened = SecureFile(
                self, name, parent_fd, parts[-1], fd, expected, max_bytes, context
            )
            self._open_files.append(opened)
            return opened
        except (OSError, ValueError) as exc:
            os.close(parent_fd)
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"unable to open {context}: {name}") from exc


def _relative(value: str | Path) -> str:
    text = value.as_posix() if isinstance(value, Path) else value
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"unsafe relative path: {text!r}")
    return path.as_posix()
