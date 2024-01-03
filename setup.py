from setuptools import setup, find_packages

setup(
    name='LLMDrift',
    version='0.0.1',
    author='Lingjiao Chen, Matei Zaharia, and James Zou',
    author_email='lingjiao@stanford.edu',
    description='The LLM drift monitoring library',
    packages=find_packages(),
    install_requires=[
       'pandas==2.1.4',
        'langchain==0.0.354',
        'langchain-community==0.0.8',
        'langchain-core==0.1.5',
            'tiktoken==0.5.2',
        'wikipedia==1.4.0',
        'transformers==4.36.2',
    'openai==0.27.0',
    'requests==2.31.0',
    'sqlitedict==2.1.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
