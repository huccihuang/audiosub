#!/usr/bin/env python3
"""Basic smoke test to verify the package was built correctly."""

try:
    import audiosub
    print("✓ Package import successful")
except ImportError as e:
    print(f"✗ Package import failed: {e}")
    exit(1)

# Test that main function exists
import audiosub

if hasattr(audiosub, 'main'):
    print("✓ Main function exists")
else:
    print("✗ Main function not found")
    exit(1)

print("✓ Smoke test passed")