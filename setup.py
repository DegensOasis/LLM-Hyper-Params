from setuptools import setup, find_packages

setup(
    name="llm_param_optimizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'nltk',
        'python-dotenv',
        'openai',
        'transformers',
        'torch',
        'joblib'
    ],
)
