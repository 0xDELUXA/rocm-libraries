#!/usr/bin/env python3
"""
Patch the .hip_fatbin section of an ELF object with a single-arch fatbin.

Replaces the section content with a smaller single-arch fatbin, then
compacts the ELF by shifting subsequent sections to reclaim the freed
space. This avoids the string table duplication overhead that
objcopy --update-section causes on LLVM-produced objects.

Usage: patch_fatbin_inplace.py <input.o> <new_fatbin> <output.o>
"""
import struct
import sys


def _align_up(value, alignment):
    """Round value up to the next multiple of alignment."""
    if alignment <= 1:
        return value
    return (value + alignment - 1) & ~(alignment - 1)


def _compact_elf(elf_data, e_shoff, e_shentsize, e_shnum):
    """Compact an ELF relocatable by removing internal gaps between sections.

    Builds a new bytearray with sections packed tightly (respecting
    alignment), then appends the patched section header table.

    Returns the compacted bytearray unchanged if no space can be reclaimed.
    """
    e_ehsize = struct.unpack_from('<H', elf_data, 52)[0]

    # Read all section headers
    sections = []
    for i in range(e_shnum):
        shdr_off = e_shoff + i * e_shentsize
        sh_offset = struct.unpack_from('<Q', elf_data, shdr_off + 24)[0]
        sh_size = struct.unpack_from('<Q', elf_data, shdr_off + 32)[0]
        sh_addralign = struct.unpack_from('<Q', elf_data, shdr_off + 48)[0]
        sections.append({
            'idx': i,
            'shdr_off': shdr_off,
            'offset': sh_offset,
            'size': sh_size,
            'addralign': max(sh_addralign, 1),
        })

    # Sections with file content, sorted by offset
    content_sections = sorted(
        [s for s in sections if s['size'] > 0 and s['offset'] > 0],
        key=lambda s: s['offset'],
    )

    if not content_sections:
        return elf_data

    # Build compacted output starting with the ELF header
    out = bytearray(elf_data[:e_ehsize])
    cursor = e_ehsize
    offset_map = {}

    for sec in content_sections:
        new_offset = _align_up(cursor, sec['addralign'])
        # Alignment padding
        if new_offset > len(out):
            out.extend(b'\0' * (new_offset - len(out)))
        # Section data
        old_off = sec['offset']
        out.extend(elf_data[old_off:old_off + sec['size']])
        cursor = new_offset + sec['size']
        offset_map[sec['idx']] = new_offset

    # Append section header table (8-byte aligned)
    new_shoff = _align_up(len(out), 8)
    if new_shoff > len(out):
        out.extend(b'\0' * (new_shoff - len(out)))

    for i in range(e_shnum):
        shdr_off = e_shoff + i * e_shentsize
        shdr = bytearray(elf_data[shdr_off:shdr_off + e_shentsize])
        if i in offset_map:
            struct.pack_into('<Q', shdr, 24, offset_map[i])
        out.extend(shdr)

    # Patch e_shoff in the ELF header
    struct.pack_into('<Q', out, 40, new_shoff)

    return out


def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <input.o> <new_fatbin> <output.o>",
              file=sys.stderr)
        return 1

    input_path, fatbin_path, output_path = sys.argv[1:4]

    with open(input_path, 'rb') as f:
        elf_data = bytearray(f.read())
    with open(fatbin_path, 'rb') as f:
        new_fatbin = f.read()

    # Verify 64-bit little-endian ELF
    if elf_data[:4] != b'\x7fELF':
        print("Error: not an ELF file", file=sys.stderr)
        return 1
    if elf_data[4] != 2:
        print("Error: not a 64-bit ELF", file=sys.stderr)
        return 1
    if elf_data[5] != 1:
        print("Error: not little-endian", file=sys.stderr)
        return 1

    e_shoff = struct.unpack_from('<Q', elf_data, 40)[0]
    e_shentsize = struct.unpack_from('<H', elf_data, 58)[0]
    e_shnum = struct.unpack_from('<H', elf_data, 60)[0]
    e_shstrndx = struct.unpack_from('<H', elf_data, 62)[0]

    # Handle extended section count (SHN_UNDEF in e_shnum)
    if e_shnum == 0:
        e_shnum = struct.unpack_from('<Q', elf_data, e_shoff + 32)[0]
    if e_shstrndx == 0xffff:
        e_shstrndx = struct.unpack_from('<I', elf_data, e_shoff + 44)[0]

    # Read section header string table location
    shstrtab_shdr = e_shoff + e_shstrndx * e_shentsize
    shstrtab_offset = struct.unpack_from('<Q', elf_data, shstrtab_shdr + 24)[0]

    # Find .hip_fatbin section
    hip_fatbin_idx = None
    for i in range(e_shnum):
        shdr_off = e_shoff + i * e_shentsize
        sh_name_idx = struct.unpack_from('<I', elf_data, shdr_off)[0]
        name_start = shstrtab_offset + sh_name_idx
        name_end = elf_data.index(0, name_start)
        name = elf_data[name_start:name_end].decode('ascii', errors='replace')
        if name == '.hip_fatbin':
            hip_fatbin_idx = i
            break

    if hip_fatbin_idx is None:
        print("Error: .hip_fatbin section not found", file=sys.stderr)
        return 1

    # Read section offset and size
    shdr_off = e_shoff + hip_fatbin_idx * e_shentsize
    sh_offset = struct.unpack_from('<Q', elf_data, shdr_off + 24)[0]
    sh_size = struct.unpack_from('<Q', elf_data, shdr_off + 32)[0]

    if len(new_fatbin) > sh_size:
        print(f"Error: new fatbin ({len(new_fatbin)} bytes) exceeds "
              f"original section ({sh_size} bytes)", file=sys.stderr)
        return 1

    # Patch section content in-place
    elf_data[sh_offset:sh_offset + len(new_fatbin)] = new_fatbin
    # Zero-fill the remainder so compaction has a clean gap
    remaining = sh_size - len(new_fatbin)
    if remaining > 0:
        elf_data[sh_offset + len(new_fatbin):sh_offset + sh_size] = (
            b'\0' * remaining)

    # Update sh_size in the section header
    struct.pack_into('<Q', elf_data, shdr_off + 32, len(new_fatbin))

    # Compact the ELF to reclaim the freed space
    elf_data = _compact_elf(elf_data, e_shoff, e_shentsize, e_shnum)

    with open(output_path, 'wb') as f:
        f.write(elf_data)

    return 0


if __name__ == '__main__':
    sys.exit(main())
