
"""
stencil_lib.py — Generic parametric stencil generator.

One engine, many curve families. A Stencil is a print-accurate transparent
canvas (sized in real-world cm at a given DPI); curves are added by name
from a small registry of parametric formulas, or by passing your own
x(t), y(t) functions directly. New stencil types = new formula, not new
script.

Built-in curve families:
  - "ellipse"            the curve traced by a Trammel of Archimedes
  - "archimedean_spiral" r = a + b*theta   (evenly spaced coils)
  - "golden_spiral"      true logarithmic spiral with golden-ratio growth
  - "golden_grid_spiral" the square-subdivision construction (quarter-arcs)
  - "spirograph"         hypotrochoid (R, r, d)
  - "rose"               rose curve r = cos(k*theta)
  - "polygon"            regular n-gon

Usage:
    s = Stencil(width_cm=8.7, height_cm=5.2, dpi=300)
    s.add_curve("ellipse", a_cm=4, b_cm=2.5)
    s.add_curve("archimedean_spiral", turns=4, spacing_cm=0.4)
    s.save("out.png")
"""

import math
import numpy as np
import imageio
import io
# needed for show method 
PHI = (1 + math.sqrt(5)) / 2


def _linspace(a, b, n):
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]


# ---------------------------------------------------------------------------
# Curve registry: each factory takes canvas half-extents (rx_cm, ry_cm) plus
# kwargs, and returns (x_func, y_func, t0, t1, closed) where x_func/y_func
# are in cm, centered on (0, 0).
# ---------------------------------------------------------------------------

def _ellipse(rx_cm, ry_cm, a_cm=None, b_cm=None, rotation_deg=0, **_):
    a = a_cm if a_cm is not None else rx_cm * 0.9
    b = b_cm if b_cm is not None else ry_cm * 0.9
    rot = math.radians(rotation_deg)

    def x(t):
        return a * math.cos(t) * math.cos(rot) - b * math.sin(t) * math.sin(rot)

    def y(t):
        return a * math.cos(t) * math.sin(rot) + b * math.sin(t) * math.cos(rot)

    return x, y, 0, 2 * math.pi, True


def _archimedean_spiral(rx_cm, ry_cm, turns=3, spacing_cm=0.5, start_radius_cm=0.0, **_):
    b = spacing_cm / (2 * math.pi)

    def r(t):
        return start_radius_cm + b * t

    x = lambda t: r(t) * math.cos(t)
    y = lambda t: r(t) * math.sin(t)
    return x, y, 0, 2 * math.pi * turns, False


def _golden_spiral(rx_cm, ry_cm, turns=2, start_radius_cm=0.2, **_):
    growth = math.log(PHI) / (math.pi / 2)

    def r(t):
        return start_radius_cm * math.exp(growth * t)

    x = lambda t: r(t) * math.cos(t)
    y = lambda t: r(t) * math.sin(t)
    return x, y, 0, 2 * math.pi * turns, False


def _spirograph(rx_cm, ry_cm, R_cm=4, r_cm=1.5, d_cm=2, turns=None, **_):
    R, r, d = R_cm, r_cm, d_cm
    from math import gcd
    if turns is None:
        # loop count until the pattern closes (approx via ratio of R,r)
        g = gcd(int(round(R * 100)), int(round(r * 100)))
        turns = max(1, int(round(r * 100 / g)))

    def x(t):
        return (R - r) * math.cos(t) + d * math.cos((R - r) / r * t)

    def y(t):
        return (R - r) * math.sin(t) - d * math.sin((R - r) / r * t)

    return x, y, 0, 2 * math.pi * turns, True


def _rose(rx_cm, ry_cm, k=5, scale_cm=None, **_):
    scale = scale_cm if scale_cm is not None else min(rx_cm, ry_cm) * 0.9

    def r(t):
        return scale * math.cos(k * t)

    x = lambda t: r(t) * math.cos(t)
    y = lambda t: r(t) * math.sin(t)
    period = 2 * math.pi if k % 2 else math.pi
    return x, y, 0, 2 * period, True


def _polygon(rx_cm, ry_cm, sides=6, radius_cm=None, rotation_deg=-90, **_):
    radius = radius_cm if radius_cm is not None else min(rx_cm, ry_cm) * 0.9
    rot = math.radians(rotation_deg)

    def x(t):
        return radius * math.cos(t + rot)

    def y(t):
        return radius * math.sin(t + rot)

    # sampled at exact vertex angles only; add_curve will still subdivide,
    # which is harmless for straight edges (points just lie on the lines)
    return x, y, 0, 2 * math.pi, True, sides  # extra: vertex count hint


_REGISTRY = {
    "ellipse": _ellipse,
    "archimedean_spiral": _archimedean_spiral,
    "golden_spiral": _golden_spiral,
    "spirograph": _spirograph,
    "rose": _rose,
    "polygon": _polygon,
}


class Stencil:
    def __init__(self, width_cm, height_cm=None, dpi=300, margin_cm=0.0):
        self.width_cm = width_cm
        self.height_cm = height_cm if height_cm is not None else width_cm / PHI
        self.dpi = dpi
        self.px_per_cm = dpi / 2.54
        self.margin_cm = margin_cm
        self.W = round(self.width_cm * self.px_per_cm)
        self.H = round(self.height_cm * self.px_per_cm)
        # RGBA array: (H, W, 4) with fully transparent background
        self.img = np.zeros((self.H, self.W, 4), dtype=np.uint8)
        self.cx, self.cy = self.W / 2, self.H / 2

    def _to_px(self, x_cm, y_cm):
        return (self.cx + x_cm * self.px_per_cm, self.cy - y_cm * self.px_per_cm)

    def _draw_line(self, pts, stroke=(255, 255, 255, 255), width_px=3):
        """Draw a line by rasterizing points with anti-aliased thickness."""
        stroke = np.array(stroke, dtype=np.uint8)
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            self._bresenham_thick_line(int(round(x0)), int(round(y0)),
                                        int(round(x1)), int(round(y1)),
                                        width_px, stroke)

    def _bresenham_thick_line(self, x0, y0, x1, y1, width, color):
        """Bresenham line with thickness (simplified thick line rendering)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        half_width = width // 2

        while True:
            # Draw a small square around (x, y) with the given width
            for dx_off in range(-half_width, half_width + 1):
                for dy_off in range(-half_width, half_width + 1):
                    px, py = x + dx_off, y + dy_off
                    if 0 <= px < self.W and 0 <= py < self.H:
                        self.img[py, px] = color

            if x == x1 and y == y1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def add_frame(self, stroke=(255, 255, 255, 255), width_px=3):
        stroke_array = np.array(stroke, dtype=np.uint8)
        # Draw rectangle outline
        for i in range(width_px):
            # Top and bottom edges
            self.img[i, :] = stroke_array
            self.img[self.H - 1 - i, :] = stroke_array
            # Left and right edges
            self.img[:, i] = stroke_array
            self.img[:, self.W - 1 - i] = stroke_array

    def add_parametric(self, x_func, y_func, t0, t1, steps=720, closed=False,
                        stroke=(255, 255, 255, 255), width_px=3):
        ts = _linspace(t0, t1, steps)
        pts = [self._to_px(x_func(t), y_func(t)) for t in ts]
        if closed:
            pts.append(pts[0])
        self._draw_line(pts, stroke=stroke, width_px=width_px)

    def add_curve(self, name, steps=720, stroke=(255, 255, 255, 255), width_px=3, **kwargs):
        if name not in _REGISTRY:
            raise ValueError(f"Unknown curve '{name}'. Options: {list(_REGISTRY)}")
        rx_cm = self.width_cm / 2 - self.margin_cm
        ry_cm = self.height_cm / 2 - self.margin_cm
        result = _REGISTRY[name](rx_cm, ry_cm, **kwargs)
        x_func, y_func, t0, t1, closed = result[:5]
        if name == "polygon":
            sides = result[5]
            steps = sides + 1  # straight edges only need vertices
        self.add_parametric(x_func, y_func, t0, t1, steps=steps, closed=closed,
                             stroke=stroke, width_px=width_px)

    def add_golden_grid_spiral(self, stroke=(255, 255, 255, 255), width_px=3,
                                show_grid=True, min_square_px=8):
        """The square-subdivision construction (as in classic diagrams),
        kept separate since it's grid-based rather than a single formula."""
        W, H = self.W, self.H
        x, y, w, h, direction = 0, 0, W, H, 0
        squares = []
        while min(w, h) > min_square_px and len(squares) < 30:
            side = min(w, h)
            if direction == 0:
                squares.append((x, y, side, direction)); x += side; w -= side
            elif direction == 1:
                squares.append((x, y, side, direction)); y += side; h -= side
            elif direction == 2:
                squares.append((x + w - side, y, side, direction)); w -= side
            else:
                squares.append((x, y + h - side, side, direction)); h -= side
            direction = (direction + 1) % 4

        stroke_array = np.array(stroke, dtype=np.uint8)

        for (sx, sy, side, orientation) in squares:
            if show_grid:
                # Draw rectangle outline
                grid_width = max(1, width_px - 1)
                for i in range(grid_width):
                    self.img[sy + i, sx:sx + side] = stroke_array
                    self.img[sy + side - 1 - i, sx:sx + side] = stroke_array
                    self.img[sy:sy + side, sx + i] = stroke_array
                    self.img[sy:sy + side, sx + side - 1 - i] = stroke_array

            # Draw quarter-arc (simplified as octant approximation)
            if orientation == 0:
                cx, cy = sx, sy + side
                self._draw_quarter_arc(cx, cy, side, 270, 360, stroke_array, width_px)
            elif orientation == 1:
                cx, cy = sx, sy
                self._draw_quarter_arc(cx, cy, side, 0, 90, stroke_array, width_px)
            elif orientation == 2:
                cx, cy = sx + side, sy
                self._draw_quarter_arc(cx, cy, side, 90, 180, stroke_array, width_px)
            else:
                cx, cy = sx + side, sy + side
                self._draw_quarter_arc(cx, cy, side, 180, 270, stroke_array, width_px)

    def _draw_quarter_arc(self, cx, cy, radius, start_deg, end_deg, color, width_px):
        """Draw a quarter-arc using Bresenham circle."""
        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)
        steps = max(int(radius / 2), 20)

        for i in range(steps):
            t = start_rad + (end_rad - start_rad) * i / steps
            x = int(round(cx + radius * math.cos(t)))
            y = int(round(cy - radius * math.sin(t)))
            half_width = width_px // 2
            for dx in range(-half_width, half_width + 1):
                for dy in range(-half_width, half_width + 1):
                    px, py = x + dx, y + dy
                    if 0 <= px < self.W and 0 <= py < self.H:
                        self.img[py, px] = color

    def composite_on(self, rgb=(139, 105, 70)):
        """Composite transparent image onto an RGB background."""
        bg = np.full((self.H, self.W, 3), rgb, dtype=np.uint8)
        # Extract alpha channel and blend
        alpha = self.img[:, :, 3:4] / 255.0
        bg = (bg * (1 - alpha) + self.img[:, :, :3] * alpha).astype(np.uint8)
        return bg

    def save(self, path):
        """Save as RGBA or RGB depending on path extension."""
        if path.lower().endswith(('.png', '.tiff')):
            imageio.imwrite(path, self.img)  # Save with alpha channel
        else:
            rgb_img = self.composite_on()
            imageio.imwrite(path, rgb_img)

    def show(self, composite=True):
        """The 'Nuclear Option' for JupyterLite: Converts array to PNG bytes."""
        img_data = self.composite_on() if composite else self.img
        
        # 1. Standardize the array (Essential for any method)
        if img_data.dtype != np.uint8:
            img_data = (img_data * 255).clip(0, 255).astype(np.uint8) if img_data.max() <= 1.01 else img_data.clip(0, 255).astype(np.uint8)
        
        if img_data.ndim == 3 and img_data.shape[0] in [3, 4]:
            img_data = img_data.transpose(1, 2, 0)

        # 2. Krita Integration (Kept as is)
        try:
            import krita
            if krita.Krita.instance() is not None:
                height, width = img_data.shape[:2]
                doc = krita.Krita.instance().createDocument(width, height, "AI Render", "RGBA", "U8", "")
                layer = doc.createLayer("Result", "paint")
                doc.rootNode().addChildNode(layer, None)
                layer.setPixelData(img_data.tobytes(), 0, 0, width, height)
                krita.Krita.instance().activeWindow().setActiveImage(doc)
                return
        except (ImportError, AttributeError):
            pass

        # 3. The Robust JupyterLite Method: Array -> PNG Bytes -> Browser
        try:
            from IPython.display import display, Image as IPythonImage
            from PIL import Image as PILImage
            
            # Convert NumPy array to PIL Image
            pil_img = PILImage.fromarray(img_data)
            
            # Save the PIL Image to a memory buffer as a PNG
            buf = io.BytesIO()
            pil_img.save(buf, format='PNG')
            byte_im = buf.getvalue()
            
            # Display the raw bytes. This is the most compatible way for web-browsers.
            display(IPythonImage(data=byte_im))
            return
        except ImportError:
            pass

        print("Error: Could not render image in this environment.")


if __name__ == "__main__":
    demos = [
        ("ellipse", dict(a_cm=3.8, b_cm=2.2)),
        ("archimedean_spiral", dict(turns=4, spacing_cm=0.35)),
        ("golden_spiral", dict(turns=2)),
        ("spirograph", dict(R_cm=3.5, r_cm=1.3, d_cm=1.8)),
        ("rose", dict(k=5)),
        ("polygon", dict(sides=6)),
    ]
    
    cols, rows = 3, 2
    cw, ch = 300, 300
    tile = np.full((rows * ch, cols * cw, 3), (30, 30, 30), dtype=np.uint8)
    
    for i, (name, kwargs) in enumerate(demos):
        s = Stencil(width_cm=6, height_cm=6, dpi=150, margin_cm=0.3)
        s.add_frame(width_px=2)
        s.add_curve(name, width_px=3, **kwargs)
        thumb_rgb = s.composite_on((45, 45, 45))
        # Resize using imageio (simplified: downsampling by slicing)
        thumb = thumb_rgb[::2, ::2]  # Simple 2x downsam
