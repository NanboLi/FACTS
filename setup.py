from setuptools import setup, find_packages

deps = [
    'einops>=0.8.0',
    'torch>=2.1.0',
]

setup(
    name = 'facts_ssm',
    packages = find_packages(
        exclude=[
            'demos/scripts',
            'demos/scripts/*',
            'assets',
            'assets/*',
        ]        
    ),
    version = '0.1.0',
    license='MIT',
    description = 'FACTS: a FACTored State-space framework for world modelling',
    long_description_content_type = 'text/markdown',
    author = 'Li Nanbo',
    author_email = 'linanbo2008@gmail.com',
    url = 'https://github.com/nanboli/facts',
    keywords = ['state-space models', 'attention', 'world models', 'artificial intelligence'],
    install_requires=deps,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    python_requires=">=3.8",
)
