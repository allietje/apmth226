# implementation.pus
from data.data import load_mnist
from src.model import create_networks
from src.train import train_bp_dfa
from src.utils import set_seed

seed = 42
set_seed(seed)

(X_train, y_train), (X_test, y_test) = load_mnist()

bp_net, dfa_net, layer_sizes = create_networks(
    width=200,
    depth=3,
    lr_bp=0.001,
    lr_dfa=0.005,
    feedback_scale=0.03,
)

metrics = train_bp_dfa(
    bp_net,
    dfa_net,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=10,
    batch_size=64,
    log_weights=True,
    plot=True,
    plot_path='./results/test_seed2.png'
)
