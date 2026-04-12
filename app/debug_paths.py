import os, sys

print("\n=== DEBUG: CURRENT FILE ===")
print(__file__)

print("\n=== DEBUG: WORKING DIRECTORY ===")
print(os.getcwd())

print("\n=== DEBUG: CONTENT OF app/ ===")
app_dir = os.path.dirname(__file__)
print(os.listdir(app_dir))

print("\n=== DEBUG: CONTENT OF app/components/ ===")
comp_dir = os.path.join(app_dir, "components")
print("exists:", os.path.exists(comp_dir))
if os.path.exists(comp_dir):
    print(os.listdir(comp_dir))

print("\n=== DEBUG: sys.path ===")
for p in sys.path:
    print(p)
