import setuptools


with open('README.md') as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    install_requires = fh.read()

NAME = 'vardautomation'
VERSION = '0.1.2'

setuptools.setup(
    name=NAME,
    version=VERSION,
    author='Vardë',
    author_email='ichunjo.le.terrible@gmail.com',
    description='Automatisation tools',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['vardautomation'],
    package_data={
        'vardautomation': ['py.typed'],
    },
    url='https://github.com/Ichunjo/encode-scripts/tree/master/vardautomation',
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
    install_requires=install_requires,
)
