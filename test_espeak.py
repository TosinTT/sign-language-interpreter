#!/usr/bin/env python3
"""
Simple script to test espeak text-to-speech functionality
For Raspberry Pi sign language interpretation system
"""

import os
import subprocess
import time

def test_espeak_basic():
    """Test basic espeak functionality"""
    print("Testing basic espeak...")
    try:
        # Basic test
        os.system('espeak "Hello, this is a test of espeak"')
        print("✓ Basic espeak test completed")
        time.sleep(1)
    except Exception as e:
        print(f"✗ Basic espeak test failed: {e}")

def test_espeak_parameters():
    """Test espeak with different parameters"""
    print("\nTesting espeak with parameters...")
    
    # Test different voices and speeds
    test_cases = [
        ('espeak -s 120 "Testing slow speech"', "Slow speech test"),
        ('espeak -s 200 "Testing fast speech"', "Fast speech test"), 
        ('espeak -p 30 "Testing low pitch"', "Low pitch test"),
        ('espeak -p 70 "Testing high pitch"', "High pitch test"),
        ('espeak -a 200 "Testing loud volume"', "Loud volume test")
    ]
    
    for command, description in test_cases:
        try:
            print(f"Running: {description}")
            os.system(command)
            time.sleep(1)
            print(f"✓ {description} completed")
        except Exception as e:
            print(f"✗ {description} failed: {e}")

def test_espeak_subprocess():
    """Test espeak using subprocess for better control"""
    print("\nTesting espeak with subprocess...")
    try:
        result = subprocess.run(['espeak', 'Testing subprocess method'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Subprocess espeak test completed")
        else:
            print(f"✗ Subprocess test failed with code: {result.returncode}")
    except Exception as e:
        print(f"✗ Subprocess test failed: {e}")

def test_sign_language_phrases():
    """Test espeak with sign language related phrases"""
    print("\nTesting sign language phrases...")
    
    phrases = [
        "Sign language gesture recognized",
        "Hello in sign language",
        "Thank you for using the system",
        "Gesture A detected",
        "Gesture B detected", 
        "System ready for input"
    ]
    
    for phrase in phrases:
        try:
            print(f"Speaking: {phrase}")
            os.system(f'espeak "{phrase}"')
            time.sleep(0.5)
        except Exception as e:
            print(f"✗ Failed to speak phrase '{phrase}': {e}")

def check_espeak_installation():
    """Check if espeak is properly installed"""
    print("Checking espeak installation...")
    try:
        result = subprocess.run(['which', 'espeak'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ espeak found at: {result.stdout.strip()}")
        else:
            print("✗ espeak not found. Install with: sudo apt-get install espeak")
            return False
        
        # Check version
        version_result = subprocess.run(['espeak', '--version'], 
                                      capture_output=True, text=True)
        if version_result.returncode == 0:
            print(f"✓ espeak version: {version_result.stderr.strip()}")
        
        return True
    except Exception as e:
        print(f"✗ Error checking espeak: {e}")
        return False

def main():
    """Main function to run all tests"""
    print("=" * 50)
    print("ESPEAK TEST SUITE FOR SIGN LANGUAGE SYSTEM")
    print("=" * 50)
    
    # Check installation first
    if not check_espeak_installation():
        print("\nPlease install espeak first: sudo apt-get install espeak")
        return
    
    print("\nStarting espeak tests...")
    print("Make sure your speakers/audio is turned on!")
    
    # Auto-start tests after 2 seconds
    print("Tests will begin in 2 seconds...")
    time.sleep(2)
    
    # Run tests
    test_espeak_basic()
    test_espeak_parameters()
    test_espeak_subprocess()
    test_sign_language_phrases()
    
    print("\n" + "=" * 50)
    print("ESPEAK TESTS COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    main()