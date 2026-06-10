from setuptools import find_packages, setup
from typing import List
def get_requirements()-> List[str]:
    """
    this function will return list of requirements

    """ 
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    
    except FileNotFoundError:
        print("requirements.txt file is not found")
    
    return requirement_lst

setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Aditya Nanche",
    author_email="adityananche22@gmail.com",
    packages=find_packages(),
    get_requirements=get_requirements()

)

                 