#!/usr/bin/env python3
"""Basic smoke test to verify the package was built correctly."""

try:
    import mksub
    print("✓ Package import successful")
except ImportError as e:
    print(f"✗ Package import failed: {e}")
    exit(1)

# Test that main function exists
import importlib.util
spec = importlib.util.spec_from_file_location("main", "main.py")
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)

if hasattr(main_module, 'main'):
    print("✓ Main function exists")
else:
    print("✗ Main function not found")
    exit(1)

print("✓ Smoke test passed")