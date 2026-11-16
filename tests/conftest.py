def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: fp16 and fused-kernel tolerance tests that need a CUDA device")
