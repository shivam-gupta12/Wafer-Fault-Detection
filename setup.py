from setuptools import setup, find_packages
from typing import List

HYPHEH_E_DOT  = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n" , "") for req in requirements]
        if HYPHEH_E_DOT in requirements:
            requirements.remove(HYPHEH_E_DOT)
        
        return requirements


setup (
    name = "src",
    version = '0.0.1',
    author = 'Shivam Gupta',
    author_email = 'shivam.gupta.bpl003@gmail.com',
    packages  = find_packages(),
    install_requires  = get_requirements('requirements.txt')
    
)

