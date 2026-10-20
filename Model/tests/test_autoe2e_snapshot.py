"""Raw snapshot tar, index, ranged read, and manifest contract tests."""

from __future__ import annotations

import hashlib
import io
import tarfile

import pytest

from Platform.pipelines.autoe2e_dataset import AutoE2EDataset
from Platform.pipelines.autoe2e_snapshot import (
    MemberRef,
    build_manifest,
    index_bytes,
    index_tar,
    iter_units,
    manifest_bytes,
    parse_manifest,
    read_index,
    read_member,
    snapshot_prefix,
    unit_entry,
    unit_index_key,
    unit_tar_key,
    write_unit_tar,
)
from Platform.pipelines.dataset_publication import sha256_bytes

KIT = AutoE2EDataset.KITSCENES
FILES = {
    "camera_ring_front/000001.jpg": b"\xff\xd8front-1",
    "camera_ring_front/000002.jpg": b"\xff\xd8front-2" * 300,
    "lidar_top/000001.parquet": b"PAR1" + bytes(range(256)) * 5,
    "meta/" + "long-name-" * 12 + ".json": b"{}",
}


def _tar(files=FILES, unit="scene-a") -> tuple[bytes, list[MemberRef]]:
    out = io.BytesIO()
    written = write_unit_tar(
        out,
        unit,
        ((path, len(body), io.BytesIO(body)) for path, body in sorted(files.items())),
    )
    return out.getvalue(), written


def test_tar_is_deterministic():
    assert _tar()[0] == _tar()[0]


def test_index_offsets_address_member_bytes_exactly():
    payload, written = _tar()
    members = index_tar(io.BytesIO(payload))
    assert [m.name for m in members] == [w.name for w in written]
    for member, (relative, body) in zip(members, sorted(FILES.items())):
        assert member.name == f"scene-a/{relative}"
        assert payload[member.offset:member.offset + member.size] == body
        assert member.sha256 == hashlib.sha256(body).hexdigest()
    assert [m.sha256 for m in members] == [w.sha256 for w in written]


def test_long_member_names_survive():
    payload, _ = _tar()
    names = [m.name for m in index_tar(io.BytesIO(payload))]
    assert any(len(name) > 100 for name in names)


@pytest.mark.parametrize("bad", ["../x", "/abs", "a//b", "a/./b", ""])
def test_unsafe_member_paths_are_rejected(bad):
    with pytest.raises(ValueError):
        _tar({bad: b"x"})


def test_unsorted_members_are_rejected():
    out = io.BytesIO()
    with pytest.raises(ValueError, match="sorted"):
        write_unit_tar(out, "u", [("b", 1, io.BytesIO(b"b")), ("a", 1, io.BytesIO(b"a"))])


def test_short_stream_is_rejected():
    with pytest.raises((OSError, tarfile.TarError)):
        write_unit_tar(io.BytesIO(), "u", [("a", 10, io.BytesIO(b"abc"))])


def test_index_round_trips_and_is_deterministic():
    payload, _ = _tar()
    members = index_tar(io.BytesIO(payload))
    encoded = index_bytes(members)
    assert encoded == index_bytes(list(reversed(members)))
    assert read_index(encoded) == sorted(members, key=lambda m: m.name)


def test_index_is_canonical_jsonl_without_library_metadata():
    payload, _ = _tar()
    encoded = index_bytes(index_tar(io.BytesIO(payload)))
    assert encoded.endswith(b"\n") and encoded.startswith(b"{\"name\"")
    with pytest.raises(ValueError, match="canonical"):
        read_index(encoded.replace(b'{"name"', b'{ "name"', 1))


def test_short_reads_from_the_source_stream_are_tolerated():
    class Trickle(io.RawIOBase):
        def __init__(self, body):
            self.body = io.BytesIO(body)

        def readable(self):
            return True

        def read(self, size=-1):
            return self.body.read(min(7, size if size >= 0 else 7))

    body = bytes(range(256)) * 40
    out = io.BytesIO()
    write_unit_tar(out, "u", [("a.bin", len(body), Trickle(body))])
    member = index_tar(io.BytesIO(out.getvalue()))[0]
    assert out.getvalue()[member.offset:member.offset + member.size] == body


def test_symlinks_and_sparse_members_are_not_indexed():
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        link = tarfile.TarInfo("u/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "u/target"
        tar.addfile(link)
    with pytest.raises(ValueError, match="regular"):
        index_tar(io.BytesIO(out.getvalue()))


def test_large_members_are_refused_in_memory():
    with pytest.raises(ValueError, match="max_bytes"):
        read_member(_RangeS3(b""), "b", "k", MemberRef("u/a", 512, 10, "a" * 64), max_bytes=5)


def test_index_rejects_duplicate_names():
    member = MemberRef("u/a", 512, 1, "a" * 64)
    with pytest.raises(ValueError, match="twice"):
        index_bytes([member, member])


class _RangeS3:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.ranges: list[tuple[int, int]] = []

    def get_object(self, *, Bucket, Key, Range):
        start, end = (int(x) for x in Range.removeprefix("bytes=").split("-"))
        self.ranges.append((start, end))
        return {"Body": io.BytesIO(self.payload[start:end + 1])}


def test_read_member_uses_one_ranged_get():
    payload, _ = _tar()
    members = index_tar(io.BytesIO(payload))
    s3 = _RangeS3(payload)
    target = members[1]
    assert read_member(s3, "b", "k", target) == FILES["camera_ring_front/000002.jpg"]
    assert s3.ranges == [(target.offset, target.offset + target.size - 1)]


def test_read_member_detects_changed_bytes():
    payload, _ = _tar()
    member = index_tar(io.BytesIO(payload))[0]
    tampered = bytearray(payload)
    tampered[member.offset] ^= 0xFF
    with pytest.raises(ValueError, match="differs"):
        read_member(_RangeS3(bytes(tampered)), "b", "k", member)


def _manifest(units=("scene-b", "scene-a")):
    prefix = snapshot_prefix(KIT, "6fde0034", "snap-1")
    entries = []
    for unit in units:
        payload, _ = _tar(unit=unit)
        members = index_tar(io.BytesIO(payload))
        entries.append(unit_entry(
            unit,
            tar_key=unit_tar_key(prefix, unit),
            tar_size=len(payload),
            tar_sha256=sha256_bytes(payload),
            index_key=unit_index_key(prefix, unit),
            index_sha256=sha256_bytes(index_bytes(members)),
            members=members,
        ))
    return build_manifest(
        dataset=KIT,
        upstream="KIT-MRT/KITScenes-Multimodal",
        revision="6fde0034",
        snapshot_id="snap-1",
        provenance={"method": "s3-copy", "source_execution_id": "a5nk56zsbcg7g6npjfpt"},
        units=entries,
    )


def test_manifest_round_trips_through_its_pinned_digest():
    manifest = _manifest()
    payload = manifest_bytes(manifest)
    parsed = parse_manifest(payload, expected_sha256=sha256_bytes(payload))
    assert [u["unit_id"] for u in parsed["units"]] == ["scene-a", "scene-b"]
    assert [u["unit_id"] for u in iter_units(parsed, 1)] == ["scene-a"]


def test_manifest_rejects_other_bytes():
    payload = manifest_bytes(_manifest())
    with pytest.raises(ValueError, match="pinned"):
        parse_manifest(payload + b"\n", expected_sha256=sha256_bytes(payload))


def test_manifest_rejects_keys_outside_the_snapshot():
    manifest = _manifest()
    manifest["units"][0]["tar_key"] = "elsewhere/units/scene-a.tar"
    payload = manifest_bytes(manifest)
    with pytest.raises(ValueError, match="outside"):
        parse_manifest(payload, expected_sha256=sha256_bytes(payload))


def test_manifest_rejects_duplicate_units():
    with pytest.raises(ValueError, match="twice"):
        _manifest(units=("scene-a", "scene-a"))
