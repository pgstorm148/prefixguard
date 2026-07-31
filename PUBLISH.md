# Publishing prefixguard

Name is free on PyPI (checked: 404 on /pypi/prefixguard/json).

```bash
python -m build
python -m twine check dist/*

# dry run first
python -m twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ prefixguard

# then for real
python -m twine upload dist/*
```

Use a scoped PyPI API token (`__token__` as username). Recommended: set up
Trusted Publishing via GitHub Actions so no token ever lives on your laptop.
