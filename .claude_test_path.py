"""Trace which trimesh path wins after AppLauncher + sys.path strip."""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Strip pip_prebundle
print("--- BEFORE STRIP ---", flush=True)
print("trimesh in sys.modules:", "trimesh" in sys.modules)
if "trimesh" in sys.modules:
    print("trimesh file:", getattr(sys.modules["trimesh"], "__file__", None))
print("path entries containing pip_prebundle:")
for p in sys.path:
    if "pip_prebundle" in p:
        print(" ", p)

sys.path[:] = [p for p in sys.path if "pip_prebundle" not in p]
for _mod_name in list(sys.modules):
    if _mod_name == "trimesh" or _mod_name.startswith("trimesh.") or \
       _mod_name == "rtree" or _mod_name.startswith("rtree."):
        _mod = sys.modules.get(_mod_name)
        _mod_file = getattr(_mod, "__file__", None) or ""
        if "pip_prebundle" in _mod_file:
            del sys.modules[_mod_name]

# Write debug info to a fresh file
with open("/mnt/storage/lti/UWLab/.claude_logs/path_check.txt", "w") as f:
    f.write("--- AFTER STRIP ---\n")
    f.write(f"trimesh in sys.modules: {'trimesh' in sys.modules}\n")
    f.write("sys.path:\n")
    for p in sys.path:
        f.write(f"  {p}\n")
    f.write("sys.meta_path:\n")
    for finder in sys.meta_path:
        f.write(f"  {finder!r}\n")
    f.write("sys.path_hooks:\n")
    for hook in sys.path_hooks:
        f.write(f"  {hook!r}\n")
    # check importlib find_spec
    import importlib.util
    spec = importlib.util.find_spec("trimesh")
    f.write(f"\nfind_spec('trimesh'): origin={spec.origin if spec else None}, search_locations={spec.submodule_search_locations if spec else None}\n")
    import trimesh
    f.write(f"\ntrimesh after re-import: {trimesh.__file__}\n")
    f.write(f"trimesh.__path__: {trimesh.__path__}\n")
    import trimesh.ray
    f.write(f"trimesh.ray: {trimesh.ray.__file__}\n")

# Now mimic the import chain that record_grasps does
import isaaclab_tasks  # noqa
import uwlab_tasks  # noqa

mesh = trimesh.creation.box()
import inspect
with open("/mnt/storage/lti/UWLab/.claude_logs/path_check.txt", "a") as f:
    f.write("\n--- AFTER uwlab_tasks ---\n")
    f.write(f"trimesh: {trimesh.__file__}\n")
    f.write(f"trimesh.ray: {trimesh.ray.__file__}\n")
    f.write(f"mesh.ray module: {inspect.getfile(type(mesh.ray))}\n")

simulation_app.close()
