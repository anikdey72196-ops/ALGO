"""
generate_png_icons.py — Generate high-resolution PNG icons using Python's standard library (zlib/struct).
No external dependencies required!
"""
import zlib
import struct
import math
from pathlib import Path

def create_png(width: int, height: int, pixels: list[tuple[int, int, int, int]]) -> bytes:
    """Encode RGBA pixels to a standard PNG file byte string."""
    raw_data = bytearray()
    for y in range(height):
        raw_data.append(0)  # filter type 0: None
        row_start = y * width
        for x in range(width):
            r, g, b, a = pixels[row_start + x]
            raw_data.extend((r, g, b, a))

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xffffffff
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    png.extend(make_chunk(b"IHDR", ihdr_data))
    # IDAT
    compressed = zlib.compress(bytes(raw_data), level=9)
    png.extend(make_chunk(b"IDAT", compressed))
    # IEND
    png.extend(make_chunk(b"IEND", b""))
    return bytes(png)

def render_terminal_icon(size: int = 192) -> bytes:
    """Renders a sleek high-tech trading terminal icon at the given dimension."""
    pixels = []
    corner_radius = size * 0.22
    cx, cy = size / 2.0, size / 2.0
    half = size / 2.0

    for y in range(size):
        for x in range(size):
            # Compute distance to squircle / rounded rect
            dx = max(abs(x - cx) - (half - corner_radius), 0)
            dy = max(abs(y - cy) - (half - corner_radius), 0)
            dist = math.sqrt(dx * dx + dy * dy)

            # Antialiasing outer boundary
            if dist > corner_radius:
                alpha = max(0.0, min(1.0, 1.0 - (dist - corner_radius)))
            else:
                alpha = 1.0

            if alpha <= 0.001:
                pixels.append((0, 0, 0, 0))
                continue

            # Border detection (thickness ~ 3.5% of size)
            border_thick = max(2.0, size * 0.035)
            is_border = (corner_radius - dist) < border_thick if dist > 0 else (min(x, size - 1 - x, y, size - 1 - y) < border_thick)

            # Gradient background (obsidian deep slate #0f172a to #020617)
            grad_factor = (x + y) / (2.0 * size)
            bg_r = int(15 * (1 - grad_factor) + 2 * grad_factor)
            bg_g = int(23 * (1 - grad_factor) + 6 * grad_factor)
            bg_b = int(42 * (1 - grad_factor) + 23 * grad_factor)

            if is_border:
                # Border gradient (cyan to emerald)
                b_r = int(6 * (1 - grad_factor) + 16 * grad_factor)
                b_g = int(182 * (1 - grad_factor) + 185 * grad_factor)
                b_b = int(212 * (1 - grad_factor) + 129 * grad_factor)
                r, g, b = b_r, b_g, b_b
            else:
                r, g, b = bg_r, bg_g, bg_b

            # Draw Terminal Prompt ">"
            # Scale coordinates into [0, 1] relative to size
            nx = x / size
            ny = y / size

            # Terminal window header bar
            if 0.18 <= ny <= 0.26 and 0.16 <= nx <= 0.84:
                r, g, b = 20, 30, 48
                # dots at left
                if 0.20 <= ny <= 0.24:
                    if abs(nx - 0.22) < 0.015:
                        r, g, b = 239, 68, 68 # red
                    elif abs(nx - 0.27) < 0.015:
                        r, g, b = 245, 158, 11 # amber
                    elif abs(nx - 0.32) < 0.015:
                        r, g, b = 16, 185, 129 # emerald

            # Inside window background
            elif 0.26 < ny <= 0.82 and 0.16 <= nx <= 0.84:
                r, g, b = 3, 7, 18

                # Chevron '>' prompt
                # Top half: from (0.24, 0.36) to (0.36, 0.46)
                # Bottom half: from (0.36, 0.46) to (0.24, 0.56)
                thick = 0.025
                in_chevron = False
                if 0.36 <= ny <= 0.46:
                    expected_x = 0.24 + (ny - 0.36) * 1.2
                    if abs(nx - expected_x) < thick:
                        in_chevron = True
                elif 0.46 < ny <= 0.56:
                    expected_x = 0.36 - (ny - 0.46) * 1.2
                    if abs(nx - expected_x) < thick:
                        in_chevron = True

                # Cursor '_'
                in_cursor = (0.54 <= ny <= 0.565 and 0.42 <= nx <= 0.58)

                if in_chevron or in_cursor:
                    # Glowing emerald
                    r, g, b = 52, 211, 153
                else:
                    # Code lines simulation
                    if 0.63 <= ny <= 0.66 and 0.25 <= nx <= 0.68:
                        r, g, b = (16, 185, 129) if nx < 0.45 else (56, 189, 248) # green / sky
                    elif 0.70 <= ny <= 0.73 and 0.25 <= nx <= 0.75:
                        r, g, b = (244, 63, 94) if nx < 0.50 else (100, 116, 139) # rose / slate

            # Outer subtle glow or blend
            final_a = int(alpha * 255)
            pixels.append((r, g, b, final_a))

    return create_png(size, size, pixels)

def main():
    icons_dir = Path(__file__).resolve().parent.parent / "static" / "icons"
    icons_dir.mkdir(parents=True, exist_ok=True)

    print("Generating icons...")
    for sz in [180, 192, 512]:
        png_data = render_terminal_icon(sz)
        name = "apple-touch-icon.png" if sz == 180 else f"icon-{sz}.png"
        out_path = icons_dir / name
        with open(out_path, "wb") as f:
            f.write(png_data)
        print(f"  [OK] Created {out_path.name} ({len(png_data)} bytes)")

if __name__ == "__main__":
    main()
