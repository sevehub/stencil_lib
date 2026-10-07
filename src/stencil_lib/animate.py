import io
import zipfile
import imageio
from .stencil_lib import Stencil


def animate(curve_name, param_name, values, width_cm, height_cm=None, dpi=300,
            zip_path="frames.zip", frame_name_fmt="frame_{index:03d}.png",
            stroke=(255, 255, 255, 255), show_frame=True, **fixed_kwargs):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, val in enumerate(values):
            s = Stencil(width_cm=width_cm, height_cm=height_cm, dpi=dpi)
            if show_frame:
                s.add_frame()
            kwargs = dict(fixed_kwargs)
            kwargs[param_name] = val
            s.add_curve(curve_name, stroke=stroke, **kwargs)

            frame_buf = io.BytesIO()
            imageio.imwrite(frame_buf, s.img, format="png")   # <- changed
            zf.writestr(frame_name_fmt.format(index=i), frame_buf.getvalue())
    with open(zip_path, "wb") as f:
        f.write(buffer.getvalue())
    return zip_path
