"""Open Top / Front / Side viewports under the default perspective view.

Display-only helper for teleoperation. It is handed to Omniverse Kit with
``--exec`` (through ``--kit_args``) and never touches the environment, the
recorder or the saved data; the demos come from the unmodified
``scripts/tools/record_demos.py``.

Three extra cameras are created under /World/Views and aimed at the table
workspace (truck spawn area + box), then docked in a row below the main
viewport. Mouse wheel in any view zooms as usual.
"""

import asyncio

import omni.kit.app
import omni.ui as ui
import omni.usd
from omni.kit.viewport.utility import create_viewport_window
from pxr import Gf, UsdGeom

# Workspace centre in world frame: between the truck spawn area and the box.
CENTER = Gf.Vec3d(0.15, -0.10, -0.55)
VIEWS = [
    # name,  eye position (world),                    up vector
    ("Top", Gf.Vec3d(0.15, -0.10, 0.75), Gf.Vec3d(1.0, 0.0, 0.0)),
    ("Front", Gf.Vec3d(1.45, -0.10, -0.25), Gf.Vec3d(0.0, 0.0, 1.0)),
    ("Side", Gf.Vec3d(0.15, -1.40, -0.25), Gf.Vec3d(0.0, 0.0, 1.0)),
]


def _make_camera(stage, path, eye, up):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.GetFocalLengthAttr().Set(18.0)
    xf = UsdGeom.Xformable(cam.GetPrim())
    xf.ClearXformOpOrder()
    m = Gf.Matrix4d()
    m.SetLookAt(eye, CENTER, up)  # world -> camera
    xf.AddTransformOp().Set(m.GetInverse())
    return path


async def _setup():
    app = omni.kit.app.get_app()
    for _ in range(600):
        await app.next_update_async()
        stage = omni.usd.get_context().get_stage()
        if stage is not None and stage.GetPrimAtPath("/World").IsValid():
            break
    # a few more frames so the scene finishes loading before we add prims
    for _ in range(30):
        await app.next_update_async()

    main = ui.Workspace.get_window("Viewport")
    prev = None
    for name, eye, up in VIEWS:
        cam_path = _make_camera(stage, f"/World/Views/{name}Cam", eye, up)
        window = create_viewport_window(name, width=600, height=360)
        for _ in range(3):
            await app.next_update_async()
        window.viewport_api.camera_path = cam_path
        if main is not None:
            if prev is None:
                window.dock_in(main, ui.DockPosition.BOTTOM, 0.38)
            else:
                window.dock_in(prev, ui.DockPosition.RIGHT, 0.5 if name == "Side" else 0.66)
        prev = window
    print("[open_views] opened Top / Front / Side viewports")


asyncio.ensure_future(_setup())
