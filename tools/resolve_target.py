import sys
import importlib.util
import subprocess
import re
import os

def find_class_line(file_path, class_name):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f, 1):
            if re.match(rf'^\s*class\s+{class_name}\b', line):
                return i
    return None

def find_class_in_package_dir(package_dir, class_name):
    # Search all .py files in the package directory for the class definition
    for root, _, files in os.walk(package_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                line_no = find_class_line(file_path, class_name)
                if line_no:
                    return file_path, line_no
    return None, None

def resolve_target(target: str):
    try:
        target = target.strip()
        parts = target.split('.')
        module_path = '.'.join(parts[:-1])
        class_name = parts[-1]

        # Find the module file
        spec = importlib.util.find_spec(module_path)
        if not spec or not spec.origin:
            print(f"Could not find module {module_path}")
            return

        file_path = spec.origin
        line_no = find_class_line(file_path, class_name)

        # If not found or file is __init__.py, search the package directory
        if (not line_no or file_path.endswith('__init__.py')):
            package_dir = os.path.dirname(file_path)
            file_path2, line_no2 = find_class_in_package_dir(package_dir, class_name)
            if file_path2 and line_no2:
                file_path, line_no = file_path2, line_no2

        if not line_no:
            line_no = 1  # fallback

        subprocess.run(['code', '-g', f'{file_path}:{line_no}'], check=False)

    except Exception as e:
        print(f"Error resolving {target}:\n{e}")

def extract_target_from_line(line: str):
    # This regex captures the target string after _target_
    match = re.match(r'^\s*[^#]*\b_target_:\s*["\']?([a-zA-Z0-9_\.]+)["\']?', line)
    if match:
        return match.group(1)
    return None

def extract_target_from_file(file_path: str):
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # This regex captures the target string after _target_ for lines that are not comments
                match = re.match(r'^\s*[^#]*\b_target_:\s*["\']?([a-zA-Z0-9_\.]+)["\']?', line)
                if match:
                    return match.group(1)
        print(f"No _target_ found in {file_path}")
    except Exception as e:
        print(f"Could not read file {file_path}:\n{e}")

def extract_target_from_file_at_line(file_path: str, line_number: int):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if 0 <= line_number < len(lines):
                return extract_target_from_line(lines[line_number])
        print(f"No _target_ found at line {line_number+1} in {file_path}")
    except Exception as e:
        print(f"Could not read file {file_path}:\n{e}")

if __name__ == "__main__":
    if len(sys.argv) == 2:
        resolve_target(sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[1] == "--top-target":
        target = extract_target_from_file(sys.argv[2])
        if target:
            resolve_target(target)
    elif len(sys.argv) == 4 and sys.argv[1] == "--at-cursor":
        file_path = sys.argv[2]
        line_number = int(sys.argv[3]) - 1  # VS Code lines are 1-based
        target = extract_target_from_file_at_line(file_path, line_number)
        if target:
            resolve_target(target)
        else:
            print("No _target_ found at cursor.")
    else:
        print("Usage:")
        print("  python resolve_target.py <target>")
        print("  python resolve_target.py --top-target <yaml_file>")
        print("  python resolve_target.py --at-cursor <yaml_file> <line_number>")
        sys.exit(1)
