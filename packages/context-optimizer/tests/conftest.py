import pytest
from context_optimizer.optimizer import ContextOptimizer


@pytest.fixture(autouse=True)
def reset_optimizer_state():
    ContextOptimizer._reset_for_test()
    yield
    ContextOptimizer._reset_for_test()
