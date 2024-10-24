from dataclasses import dataclass
import yaml
import os, sys
from pathlib import Path
from typing import Union, List, Optional
import logging
from datetime import datetime

class FileWriter:
    """Utility class for writing strings to files with various options"""

    def __init__(self, default_encoding: str = 'utf-8'):
        """
        Initialize the FileWriter.

        Args:
            default_encoding (str): Default encoding to use for file operations
        """
        self.default_encoding = default_encoding
        self._setup_logging()

    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def write_string(
            self,
            content: str,
            filepath: Union[str, Path],
            mode: str = 'w',
            encoding: Optional[str] = None,
            create_dirs: bool = True,
            backup_existing: bool = False
    ) -> bool:
        """
        Write a string to a file with error handling.

        Args:
            content (str): The string content to write
            filepath (Union[str, Path]): Path to the file
            mode (str): File open mode ('w' for write, 'a' for append)
            encoding (Optional[str]): File encoding (defaults to class default_encoding)
            create_dirs (bool): Create parent directories if they don't exist
            backup_existing (bool): Create backup of existing file before writing

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert to Path object
            file_path = Path(filepath)

            # Create directories if needed
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if requested
            if backup_existing and file_path.exists():
                self._create_backup(file_path)

            # Write the file
            with open(
                    file_path,
                    mode=mode,
                    encoding=encoding or self.default_encoding
            ) as f:
                f.write(content)

            self.logger.info(f"Successfully wrote to file: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error writing to file {filepath}: {str(e)}")
            return False

    def write_lines(
            self,
            lines: List[str],
            filepath: Union[str, Path],
            mode: str = 'w',
            encoding: Optional[str] = None,
            newline: str = '\n',
            create_dirs: bool = True
    ) -> bool:
        """
        Write a list of strings to a file, one per line.

        Args:
            lines (List[str]): List of strings to write
            filepath (Union[str, Path]): Path to the file
            mode (str): File open mode ('w' for write, 'a' for append)
            encoding (Optional[str]): File encoding (defaults to class default_encoding)
            newline (str): Line ending to use
            create_dirs (bool): Create parent directories if they don't exist

        Returns:
            bool: True if successful, False otherwise
        """
        content = newline.join(lines)
        return self.write_string(
            content=content,
            filepath=filepath,
            mode=mode,
            encoding=encoding,
            create_dirs=create_dirs
        )

    def append_string(
            self,
            content: str,
            filepath: Union[str, Path],
            encoding: Optional[str] = None,
            create_dirs: bool = True
    ) -> bool:
        """
        Append a string to an existing file.

        Args:
            content (str): The string content to append
            filepath (Union[str, Path]): Path to the file
            encoding (Optional[str]): File encoding (defaults to class default_encoding)
            create_dirs (bool): Create parent directories if they don't exist

        Returns:
            bool: True if successful, False otherwise
        """
        return self.write_string(
            content=content,
            filepath=filepath,
            mode='a',
            encoding=encoding,
            create_dirs=create_dirs
        )

    def _create_backup(self, file_path: Path) -> Optional[Path]:
        """Create a backup of the existing file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.parent / f"{file_path.stem}_{timestamp}{file_path.suffix}"
            os.replace(file_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return None

@dataclass
class Asset:
    """Represents an asset configuration"""
    url: str
    file_name: str
    location: str

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        return cls(
            url=data['url'],
            file_name=data['file_name'],
            location=data['location'],
        )

    def get_full_path(self) -> Path:
        return Path(self.location) / self.file_name


@dataclass
class Node:
    """Represents a node configuration"""
    repo: str
    pip_install: Optional[bool]
    custom_script: Optional[str]
    assets: Optional[List[Asset]] = None

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        return cls(
            repo=data['repo'],
            pip_install=data.get('pip_install', False),
            custom_script=data.get('custom_script', ''),
            assets=[Asset.from_dict(assets_data) for assets_data in data.get('assets', [])]
        )

@dataclass
class InstallConfig:
    install_type: str
    platform: str
    nodes: List[Node]
    requirements: List[str]
    custom_script: Optional[str]
    assets: Optional[List[Asset]] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'InstallConfig':
        """Creates an InstallConfig instance from a YAML file"""
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)

            return cls(
                install_type=data['install_type'],
                platform=data['platform'],
                nodes=[Node.from_dict(node_data) for node_data in data.get('nodes', [])],
                requirements=data['requirements'],
                assets=[Asset.from_dict(assets_data) for assets_data in data.get('assets', [])],
                custom_script=data.get('custom_script', ''),
            )
        except Exception as e:
            raise ValueError(f"Error parsing YAML file: {str(e)}")

    def validate(self) -> List[str]:
        """Validates the configuration and returns a list of errors"""
        errors = []

        # Validate install type
        valid_install_types = ['conda', 'pip']
        if self.install_type not in valid_install_types:
            errors.append(f"Invalid install_type: {self.install_type}. Must be one of {valid_install_types}")

        # Validate platform
        valid_platforms = ['linux', 'windows', 'macos']
        if self.platform not in valid_platforms:
            errors.append(f"Invalid platform: {self.platform}. Must be one of {valid_platforms}")

        # Validate nodes
        for i, node in enumerate(self.nodes):
            # Validate repo format
            if not node.repo.startswith('https://'):
                errors.append(f"Node {i}: Invalid repo URL format: {node.repo}")
            if node.assets:
                for asset in node.assets:
                    if not asset.url.startswith('https://'):
                        errors.append(f"Node {i}: Invalid asset URL format: {node.assets.url}")

                    if not asset.file_name:
                        errors.append(f"Node {i}: Missing file_name in assets")

                    if not asset.location:
                        errors.append(f"Node {i}: Missing location in assets")



        return errors


def load_config(yaml_path: str) -> InstallConfig:
    print("Loading file: "+yaml_path)
    """Helper function to load and validate configuration"""
    config = InstallConfig.from_yaml(yaml_path)
    errors = config.validate()

    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))

    return config

def get_downloads(c: InstallConfig):
    items = ""
    for asset in c.assets:
        items += f"""("{asset.url}", "{asset.file_name}", '{asset.location}'),\n"""
    for node in c.nodes:
        for asset in node.assets:
            items += f"""("{asset.url}", "{asset.file_name}", '{asset.location}'),\n"""
    template = f"""
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

downloads = [
    {items}
]

def download_file(url, file_name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with requests.get(url, stream=True, allow_redirects=True) as response:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 4096  # 4KB blocks
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name)
        with open(path+file_name, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

        progress_bar.close()

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('Folder: '+folder_path+', created.')

for url, file_name, path in downloads:
    create_folder_if_not_exists(path)


"""

    post = '''with ThreadPoolExecutor(max_workers=4) as executor:
    future_to_download = {executor.submit(download_file, url, file_name, path): file_name for url, file_name, path in downloads}
    for future in as_completed(future_to_download):
        file_name = future_to_download[future]'''

    return template+post

def get_windows_pip_run_file():
    return '''@echo off

cd ComfyUI

call venv\\Scripts\\activate.bat

python main.py --windows-standalone-build --listen
    '''

def get_windows_conda_run_file():
    return '''@echo off
cd ComfyUI
call conda activate comfyui
python main.py --windows-standalone-build --listen
'''

def get_windows_conda_header(c: InstallConfig):
    base= '''@echo off
call conda remove --name comfyui --all -y
call conda create --name comfyui python=3.11.9 pytorch-cuda=12.4 pytorch cudatoolkit -c pytorch -c nvidia -y
call conda activate comfyui

git clone https://github.com/comfyanonymous/ComfyUI.git
copy download.py ComfyUI/download.py
cd ComfyUI

python -m pip install -r requirements.txt    
'''
    for req in c.requirements:
        base += f'''pip install {req}\n\n'''
    return base

def get_windows_node_script(node: Node):
    name = node.repo.split("/")
    name = name[len(name) - 1]
    additional = ""
    if node.pip_install:
        additional = "pip install -r custom_nodes/" + name + "/requirements.txt\n"
    if node.custom_script:
        additional += node.custom_script+"\n"
    return f'''
git clone {node.repo}.git custom_nodes/{name}\n{additional}'''

def create_windows_conda(c: InstallConfig):
    install_script = get_windows_conda_header(c)

    for node in c.nodes:
        install_script += get_windows_node_script(node)

    install_script += "\npython download.py\n\n"
    install_script += "del download.py\n"
    install_script += c.custom_script+"\n"
    install_script += "python main.py --windows-standalone-build --listen"

    writer = FileWriter()
    writer.write_string(get_windows_conda_run_file(), "install/run.bat",create_dirs=True)
    writer.write_string(install_script, "install/install.bat",create_dirs=True)
    writer.write_string(get_downloads(c), "install/download.py",create_dirs=True)

if __name__ == "__main__":

    # Arguments start from index 1 (index 0 is the script name)
    arguments = sys.argv[1:]

    if len(arguments) != 1:
        print("wrong arguments")
        exit(137)

    config = load_config(arguments[0])

    if config.platform == "windows":
        create_windows_conda(config)
    else:
        raise ValueError("Unsupported platform: " + config.platform)
