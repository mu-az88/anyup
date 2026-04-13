def pytest_addoption(parser):
    parser.addoption("--manifest_dir",     default="configs/")
    parser.addoption("--imagenet_val_dir", default=None)
