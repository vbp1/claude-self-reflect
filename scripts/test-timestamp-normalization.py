#!/usr/bin/env python3
"""Test timestamp normalization using Python 3.11+ features."""

from datetime import datetime, timezone


def test_timestamp_normalization():
    """Test various timestamp format normalization scenarios."""
    
    test_cases = [
        # Format: (input, expected_parsed_datetime)
        ("2024-01-15T10:30:00Z", datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)),
        ("2024-01-15T10:30:00+00:00", datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)),
        ("2024-01-15T10:30:00.123456Z", datetime(2024, 1, 15, 10, 30, 0, 123456, tzinfo=timezone.utc)),
        ("2024-01-15T10:30:00.123456+00:00", datetime(2024, 1, 15, 10, 30, 0, 123456, tzinfo=timezone.utc)),
    ]
    
    print("Testing timestamp normalization with Python 3.11+ fromisoformat()...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for timestamp_str, expected_dt in test_cases:
        try:
            # Parse using Python 3.11+ fromisoformat (supports 'Z')
            parsed_dt = datetime.fromisoformat(timestamp_str)
            
            # Ensure UTC timezone
            if parsed_dt.tzinfo is None:
                parsed_dt = parsed_dt.replace(tzinfo=timezone.utc)
            elif parsed_dt.tzinfo != timezone.utc:
                parsed_dt = parsed_dt.astimezone(timezone.utc)
            
            if parsed_dt == expected_dt:
                passed += 1
                print(f"✓ PASS: '{timestamp_str}'")
                print(f"  Parsed: {parsed_dt}")
                # Format back with 'Z' suffix
                formatted = parsed_dt.isoformat().replace("+00:00", "Z")
                print(f"  Formatted back: {formatted}")
            else:
                failed += 1
                print(f"✗ FAIL: '{timestamp_str}'")
                print(f"  Expected: {expected_dt}")
                print(f"  Got: {parsed_dt}")
        except Exception as e:
            failed += 1
            print(f"✗ ERROR parsing '{timestamp_str}': {e}")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    # Test the actual code pattern used in server.py
    print("\nTesting the actual pattern from server.py:")
    print("-" * 40)
    
    test_timestamps = [
        "2024-01-15T10:30:00Z",
        "2024-01-15T10:30:00+00:00",
        "2024-01-15T10:30:00.123456Z",
    ]
    
    for timestamp_str in test_timestamps:
        if timestamp_str:
            try:
                # Python 3.11+ поддерживает 'Z' в fromisoformat
                dt = datetime.fromisoformat(timestamp_str)
                # Убеждаемся, что время в UTC
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                elif dt.tzinfo != timezone.utc:
                    dt = dt.astimezone(timezone.utc)
                # Форматируем с 'Z' суффиксом для UTC
                clean_timestamp = dt.isoformat().replace("+00:00", "Z")
                print(f"✓ '{timestamp_str}' -> '{clean_timestamp}'")
            except (ValueError, TypeError) as e:
                # При ошибке используем текущее время
                clean_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
                print(f"✗ Error with '{timestamp_str}': {e}, using current time: {clean_timestamp}")
        else:
            # Если нет timestamp, используем текущее время
            clean_timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            print(f"Empty timestamp, using current time: {clean_timestamp}")
    
    return passed, failed


def test_python_version():
    """Check Python version for fromisoformat 'Z' support."""
    import sys
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 11):
        print("✅ Python 3.11+ detected - 'Z' suffix is supported in fromisoformat()")
        return True
    elif version >= (3, 7):
        print("⚠️ Python 3.7-3.10 detected - 'Z' suffix support added in 3.11")
        print("  fromisoformat() accepts 'Z' only from Python 3.11+")
        return False
    else:
        print("❌ Python < 3.7 detected - fromisoformat() not available")
        return False


if __name__ == "__main__":
    print("Timestamp Normalization Test\n")
    
    # Check Python version
    has_z_support = test_python_version()
    print()
    
    # Run tests
    passed, failed = test_timestamp_normalization()
    
    if failed == 0:
        print("\n✅ All tests passed! Timestamp normalization is working correctly.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Review the normalization logic.")
    
    if not has_z_support:
        print("\n⚠️ Note: Your Python version doesn't support 'Z' in fromisoformat().")
        print("  Consider upgrading to Python 3.11+ for better ISO 8601 support.")