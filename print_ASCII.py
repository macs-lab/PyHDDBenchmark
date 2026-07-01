import sys
from packaging import version
import os

def color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def print_banner(version):
    banner = r"""
     ______          _     _  _____    _____   
    (_____ \        | |   | |(____ \  (____ \  
     _____) ) _   _ | |__ | | _   \ \  _   \ \ 
    |  ____/ | | | ||  __)| || |   | || |   | |
    | |      | |_| || |   | || |__/ / | |__/ / 
    |_|       \__  ||_|   |_||_____/  |_____/  
              (____/                            
    """
    colored_banner = color_text(banner, '36')  # Cyan color
    version_text = f"(v{version})"
    centered_version = version_text.center(len(banner.splitlines()[-1]))
    colored_version = color_text(centered_version, '33')  # Yellow color
    
    return colored_banner + '\n' + colored_version

def check_version(current_version, min_version):
    if version.parse(current_version) < version.parse(min_version):
        return color_text(f"Warning: Your version ({current_version}) is outdated. Please upgrade to at least version {min_version}.", '31')
    return ""

def print_system(current_version='0.1.0', min_version='0.1.0'):
    # Clear the screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print banner and version
    print(print_banner(current_version))
    
    # Check and print version warning if necessary
    version_warning = check_version(current_version, min_version)
    if version_warning:
        print(version_warning)

def suppress_scipy_warning():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

if __name__ == "__main__":
    # suppress_scipy_warning()
    print_system()