#!/usr/bin/env python3
"""
Test script to verify PET checkpoint migration from v11 to v12.

This tests that old checkpoints with 'non_conservative_force' (singular)
are automatically migrated to 'non_conservative_forces' (plural).
"""

import sys
sys.path.insert(0, '/mnt/shared-storage-gpfs2/lijiahang1/jobs/upet')

import torch
from metatrain.utils.io import load_model as load_metatrain_model

def test_checkpoint_migration(checkpoint_path: str):
    """Test that checkpoint migration works correctly."""
    print(f"Testing checkpoint migration for: {checkpoint_path}")
    print("=" * 80)

    # Load the raw checkpoint to check original version
    print("\n1. Loading raw checkpoint to check version...")
    try:
        raw_checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"   Checkpoint version: {raw_checkpoint.get('model_ckpt_version', 'unknown')}")

        # Check what's in hypers.outputs
        if 'hypers' in raw_checkpoint:
            hypers = raw_checkpoint['hypers']
            if hasattr(hypers, 'outputs'):
                outputs = hypers.outputs
            elif isinstance(hypers, dict) and 'outputs' in hypers:
                outputs = hypers['outputs']
            else:
                outputs = None

            if outputs:
                print(f"   Original outputs keys: {list(outputs.keys())}")
                if 'non_conservative_force' in outputs:
                    print("   ⚠️  Found OLD naming: non_conservative_force")
                if 'non_conservative_forces' in outputs:
                    print("   ✓ Found NEW naming: non_conservative_forces")

    except Exception as e:
        print(f"   ❌ Failed to load raw checkpoint: {e}")
        return False

    # Load through metatrain (this triggers migration)
    print("\n2. Loading through metatrain (triggers migration)...")
    try:
        model = load_metatrain_model(checkpoint_path)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check the model's outputs after migration
    print("\n3. Checking model outputs after migration...")
    if hasattr(model, 'hypers') and hasattr(model.hypers, 'outputs'):
        outputs = model.hypers.outputs
        print(f"   Migrated outputs keys: {list(outputs.keys())}")

        has_old = 'non_conservative_force' in outputs
        has_new = 'non_conservative_forces' in outputs

        if has_new and not has_old:
            print("   ✅ SUCCESS: Migration complete! Now using 'non_conservative_forces'")
            return True
        elif has_old and not has_new:
            print("   ❌ FAILED: Still using old naming 'non_conservative_force'")
            return False
        elif has_old and has_new:
            print("   ⚠️  WARNING: Both naming conventions present (backward compat mode)")
            return True
        else:
            print("   ℹ️  No non-conservative forces output found")
            return True
    else:
        print("   ℹ️  Model doesn't have hypers.outputs (might be exported model)")
        return True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PET checkpoint migration")
    parser.add_argument("checkpoint", help="Path to checkpoint file")
    args = parser.parse_args()

    success = test_checkpoint_migration(args.checkpoint)

    print("\n" + "=" * 80)
    if success:
        print("✅ TEST PASSED")
    else:
        print("❌ TEST FAILED")

    sys.exit(0 if success else 1)
