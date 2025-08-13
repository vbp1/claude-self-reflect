#!/usr/bin/env python3
"""Test text normalization for embeddings."""

import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize text for embedding using Unicode normalization.
    
    This ensures consistent representation of text regardless of:
    - Different Unicode representations (é vs e + ́)
    - Case differences
    - Whitespace variations
    """
    # Unicode normalization to NFC (Canonical Composition)
    # This ensures that é is represented as a single character, not e + combining accent
    normalized = unicodedata.normalize('NFC', text)
    
    # Convert to lowercase for case-insensitive matching
    normalized = normalized.casefold()
    
    # Normalize whitespace - replace multiple spaces/tabs/newlines with single space
    # and strip leading/trailing whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


def test_normalization():
    """Test various normalization scenarios."""
    
    test_cases = [
        # Unicode variations
        ("café", "cafe\u0301"),  # é as single char vs e + combining accent
        ("naïve", "nai\u0308ve"),  # ï as single char vs i + diaeresis
        
        # Case variations
        ("Hello World", "HELLO WORLD"),
        ("Python", "python"),
        ("PyThOn", "PYTHON"),
        
        # Whitespace variations
        ("multiple  spaces", "multiple spaces"),
        ("tabs\t\there", "tabs here"),
        ("newlines\n\nhere", "newlines here"),
        ("  leading and trailing  ", "leading and trailing"),
        ("mixed   \t\n  whitespace", "mixed whitespace"),
        
        # Combined variations
        ("Café  LATTE", "CAFE\u0301 latte"),
        ("  Naïve\tPYTHON  ", "nai\u0308ve python"),
        
        # Special characters
        ("ß", "ss"),  # German eszett should become ss with casefold
        ("İstanbul", "i\u0307stanbul"),  # Turkish capital İ
    ]
    
    print("Testing text normalization for embeddings...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for text1, text2 in test_cases:
        norm1 = normalize_text(text1)
        norm2 = normalize_text(text2)
        
        if norm1 == norm2:
            passed += 1
            print(f"✓ PASS: '{text1}' == '{text2}'")
            print(f"  Normalized: '{norm1}'")
        else:
            failed += 1
            print(f"✗ FAIL: '{text1}' != '{text2}'")
            print(f"  '{text1}' -> '{norm1}'")
            print(f"  '{text2}' -> '{norm2}'")
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    # Test actual Unicode representation
    print("\nUnicode representation test:")
    print("-" * 40)
    
    # Create two versions of café
    cafe1 = "café"  # Single character é
    cafe2 = "cafe\u0301"  # e + combining acute accent
    
    print(f"Original strings look identical: '{cafe1}' vs '{cafe2}'")
    print(f"But have different lengths: {len(cafe1)} vs {len(cafe2)}")
    print(f"Are they equal? {cafe1 == cafe2}")
    
    norm1 = normalize_text(cafe1)
    norm2 = normalize_text(cafe2)
    
    print(f"\nAfter normalization: '{norm1}' vs '{norm2}'")
    print(f"Lengths: {len(norm1)} vs {len(norm2)}")
    print(f"Are they equal? {norm1 == norm2}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = test_normalization()
    
    if failed == 0:
        print("\n✅ All tests passed! Text normalization is working correctly.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Review the normalization logic.")