"""
Script to convert all Python files from UTF-16 to UTF-8.
This fixes the null bytes issue that occurs when UTF-16 files are read as UTF-8.
"""
import os
from pathlib import Path

def convert_file_to_utf8(file_path):
    """Convert a single file from UTF-16 to UTF-8."""
    try:
        # Try reading as UTF-16
        with open(file_path, 'r', encoding='utf-16') as f:
            content = f.read()
        
        # Write back as UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Converted: {file_path}")
        return True
    except UnicodeError:
        # File might already be UTF-8 or another encoding
        try:
            # Try reading as UTF-8 to verify it's valid
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read()
            print(f"  Already UTF-8: {file_path}")
            return False
        except UnicodeError:
            print(f"✗ Could not convert: {file_path}")
            return False
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return False

def main():
    # Get project root
    project_root = Path(__file__).parent
    
    # Find all Python files
    python_files = list(project_root.rglob("*.py"))
    
    print(f"Found {len(python_files)} Python files\n")
    
    converted_count = 0
    for py_file in python_files:
        if convert_file_to_utf8(py_file):
            converted_count += 1
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Converted: {converted_count} files")
    print(f"Total processed: {len(python_files)} files")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
