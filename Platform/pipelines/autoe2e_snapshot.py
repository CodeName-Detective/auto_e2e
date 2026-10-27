"""Raw snapshots: upstream bytes as one uncompressed tar per unit.

Layout under ``s3://<bucket>/<dataset>/raw-snapshots/<revision>/<snapshot_id>/``::

    units/<unit_id>.tar      uncompressed tar, members "<unit_id>/<relative path>"
    index/<unit_id>.jsonl    one canonical JSON line per member: name, offset, size, sha256
    manifest.json            written last, write-once

Tars are deterministic (sorted members, zeroed owner and mtime), so the same
upstream files always give the same tar digest. The index is canonical JSON
Lines, not Parquet, so its bytes do not depend on a library version and a
write-once rerun reproduces them exactly. A member is read with one ranged GET
of ``[offset, offset + size)``; no extraction is needed.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
from dataclasses import dataclass
from typing import IO, Any, Iterable, Iterator, Mapping, Sequence

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.dataset_publication import canonical_json_bytes, sha256_bytes


SNAPSHOT_SCHEMA = "autoe2e_raw_snapshot_v1"
INDEX_FIELDS = ("name", "offset", "size", "sha256")
_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COPY_CHUNK = 8 * 1024**2


@dataclass(frozen=True)
class MemberRef:
    """One readable member of a unit tar."""

    name: str
    offset: int
    size: int
    sha256: str


def _segment(value: str, what: str) -> str:
    if not _SEGMENT_RE.fullmatch(value):
        raise ValueError(f"invalid {what} {value!r}")
    return value


def snapshot_prefix(dataset: AutoE2EDataset, revision: str, snapshot_id: str) -> str:
    return (
        f"{dataset.value}/raw-snapshots/"
        f"{_segment(revision, 'revision')}/{_segment(snapshot_id, 'snapshot id')}"
    )


def unit_tar_key(prefix: str, unit_id: str) -> str:
    return f"{prefix}/units/{_segment(unit_id, 'unit id')}.tar"


def unit_index_key(prefix: str, unit_id: str) -> str:
    return f"{prefix}/index/{_segment(unit_id, 'unit id')}.jsonl"


def manifest_key(prefix: str) -> str:
    return f"{prefix}/manifest.json"


def _member_name(unit_id: str, relative: str) -> str:
    parts = relative.split("/")
    if not relative or relative.startswith("/") or any(
        part in {"", ".", ".."} for part in parts
    ):
        raise ValueError(f"unsafe member path {relative!r}")
    return f"{unit_id}/{relative}"


def write_unit_tar(
    out: IO[bytes],
    unit_id: str,
    files: Iterable[tuple[str, int, IO[bytes]]],
) -> list[MemberRef]:
    """Write ``files`` as a deterministic uncompressed tar and index it.

    ``files`` yields ``(relative_path, size, stream)``. Members are written in
    sorted path order, so callers must yield them sorted; the function
    rejects an unsorted or duplicated sequence instead of buffering it.
    """
    _segment(unit_id, "unit id")
    members: list[MemberRef] = []
    previous = None
    with tarfile.open(fileobj=out, mode="w|", format=tarfile.PAX_FORMAT) as tar:
        for relative, size, stream in files:
            name = _member_name(unit_id, relative)
            if previous is not None and name <= previous:
                raise ValueError(f"members must be strictly sorted: {name!r}")
            previous = name
            info = tarfile.TarInfo(name)
            info.size = int(size)
            info.mode = 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            digest = hashlib.sha256()
            reader = _HashingReader(stream, digest, info.size)
            tar.addfile(info, reader)
            if reader.read_bytes != info.size:
                raise IOError(f"{name}: read {reader.read_bytes} of {info.size} bytes")
            members.append(MemberRef(name, -1, info.size, digest.hexdigest()))
    return members


class _HashingReader(io.RawIOBase):
    def __init__(self, stream: IO[bytes], digest: Any, limit: int):
        self._stream = stream
        self._digest = digest
        self._remaining = limit
        self.read_bytes = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        """Return exactly ``size`` bytes unless the source truly ends.

        tarfile treats a short read as end of data, but sockets and fsspec
        streams return short reads legitimately, so keep reading.
        """
        if self._remaining <= 0:
            return b""
        size = self._remaining if size < 0 else min(size, self._remaining)
        parts = []
        wanted = size
        while wanted:
            chunk = self._stream.read(wanted)
            if not chunk:
                break
            parts.append(chunk)
            wanted -= len(chunk)
        data = b"".join(parts)
        self._remaining -= len(data)
        self.read_bytes += len(data)
        self._digest.update(data)
        return data


def index_tar(stream: IO[bytes]) -> list[MemberRef]:
    """Stream a tar once and return every regular member's data offset.

    Hashes each member while streaming, so the index can be verified against
    the bytes later without another full read.
    """
    members: list[MemberRef] = []
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for info in tar:
            if info.isdir():
                continue
            if not info.isreg() or info.issparse():
                raise ValueError(
                    f"{info.name}: only regular, non-sparse members can be indexed"
                )
            handle = tar.extractfile(info)
            assert handle is not None
            digest = hashlib.sha256()
            while True:
                chunk = handle.read(_COPY_CHUNK)
                if not chunk:
                    break
                digest.update(chunk)
            members.append(
                MemberRef(info.name, info.offset_data, info.size, digest.hexdigest())
            )
    return members


def index_bytes(members: Sequence[MemberRef]) -> bytes:
    """Serialize an index as canonical JSON Lines sorted by member name."""
    ordered = sorted(members, key=lambda m: m.name)
    names = [member.name for member in ordered]
    if len(set(names)) != len(names):
        raise ValueError("tar lists a member twice")
    if any(member.offset < 0 for member in ordered):
        raise ValueError("index offsets must come from index_tar")
    return b"".join(
        canonical_json_bytes({
            "name": m.name, "offset": m.offset, "size": m.size, "sha256": m.sha256,
        }) + b"\n"
        for m in ordered
    )


def read_index(payload: bytes) -> list[MemberRef]:
    import json

    members = []
    for line in payload.splitlines():
        row = json.loads(line)
        if tuple(sorted(row)) != tuple(sorted(INDEX_FIELDS)):
            raise ValueError(f"index row fields {sorted(row)} != {sorted(INDEX_FIELDS)}")
        members.append(MemberRef(row["name"], int(row["offset"]), int(row["size"]), row["sha256"]))
    if index_bytes(members) != payload:
        raise ValueError("snapshot index is not canonical")
    return members


def read_member(
    s3,
    bucket: str,
    key: str,
    member: MemberRef,
    *,
    verify: bool = True,
    max_bytes: int = 1024**3,
) -> bytes:
    """Fetch one member into memory with a single ranged GET.

    Members above ``max_bytes`` must be streamed by the caller instead, so a
    multi-gigabyte member never silently exhausts pod memory.
    """
    if member.size > max_bytes:
        raise ValueError(f"{member.name}: {member.size} bytes exceeds max_bytes {max_bytes}")
    if member.size == 0:
        payload = b""
    else:
        response = s3.get_object(
            Bucket=bucket,
            Key=key,
            Range=f"bytes={member.offset}-{member.offset + member.size - 1}",
        )
        try:
            payload = response["Body"].read()
        finally:
            response["Body"].close()
    if len(payload) != member.size:
        raise IOError(f"{member.name}: short read {len(payload)} != {member.size}")
    if verify and hashlib.sha256(payload).hexdigest() != member.sha256:
        raise ValueError(f"{member.name}: content differs from the snapshot index")
    return payload


def unit_entry(
    unit_id: str,
    *,
    tar_key: str,
    tar_size: int,
    tar_sha256: str,
    index_key: str,
    index_sha256: str,
    members: Sequence[MemberRef],
) -> dict[str, Any]:
    for digest in (tar_sha256, index_sha256):
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"{unit_id}: invalid digest {digest!r}")
    if not members:
        raise ValueError(f"{unit_id}: unit has no members")
    return {
        "unit_id": _segment(unit_id, "unit id"),
        "tar_key": tar_key,
        "tar_size": int(tar_size),
        "tar_sha256": tar_sha256,
        "index_key": index_key,
        "index_sha256": index_sha256,
        "member_count": len(members),
        "member_bytes": sum(member.size for member in members),
    }


def build_manifest(
    *,
    dataset: AutoE2EDataset,
    upstream: str,
    revision: str,
    snapshot_id: str,
    provenance: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prefix = snapshot_prefix(dataset, revision, snapshot_id)
    ordered = sorted((dict(unit) for unit in units), key=lambda unit: unit["unit_id"])
    ids = [unit["unit_id"] for unit in ordered]
    if not ids:
        raise ValueError("snapshot has no units")
    if len(set(ids)) != len(ids):
        raise ValueError("snapshot lists a unit twice")
    for unit in ordered:
        if unit["tar_key"] != unit_tar_key(prefix, unit["unit_id"]):
            raise ValueError(f"{unit['unit_id']}: tar key is outside the snapshot")
        if unit["index_key"] != unit_index_key(prefix, unit["unit_id"]):
            raise ValueError(f"{unit['unit_id']}: index key is outside the snapshot")
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "dataset": dataset.value,
        "upstream": upstream,
        "revision": revision,
        "snapshot_id": snapshot_id,
        "prefix": prefix,
        "provenance": dict(provenance),
        "unit_count": len(ordered),
        "member_bytes": sum(unit["member_bytes"] for unit in ordered),
        "units": ordered,
    }


def manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(dict(manifest), pretty=True)


def parse_manifest(payload: bytes, *, expected_sha256: str) -> dict[str, Any]:
    """Parse a snapshot manifest only if its bytes match the pinned digest."""
    import json

    if sha256_bytes(payload) != expected_sha256:
        raise ValueError("snapshot manifest bytes differ from the pinned SHA-256")
    manifest = json.loads(payload)
    if manifest.get("schema_version") != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported raw snapshot schema")
    rebuilt = build_manifest(
        dataset=AutoE2EDataset(manifest["dataset"]),
        upstream=manifest["upstream"],
        revision=manifest["revision"],
        snapshot_id=manifest["snapshot_id"],
        provenance=manifest["provenance"],
        units=manifest["units"],
    )
    if rebuilt != manifest:
        raise ValueError("snapshot manifest is not in canonical form")
    return manifest


def iter_units(manifest: Mapping[str, Any], unit_limit: int = 0) -> Iterator[Mapping[str, Any]]:
    units = manifest["units"]
    yield from (units[:unit_limit] if unit_limit else units)
