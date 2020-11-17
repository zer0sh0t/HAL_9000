from setuptools import setup, find_packages

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Education',
    'Operating System :: Microsoft :: Windows :: Windows 10',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
]

setup(
    name='HAL_9000',
    version='2.0.0',
    description='A Neural Network library',
    long_description=open('README.txt').read() + '\n\n' +
    open('CHANGELOG.txt').read(),
    url='',
    author='Harin Kumar',
    author_email='Harinkumar851@gmail.com',
    license='MIT',
    classifiers=classifiers,
    keywords='neuralnetworks',
    packages=find_packages(),
    install_requires=['gym']
)
