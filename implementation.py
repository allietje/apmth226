# implementation.pus
from data.data import load_mnist
from src.model import create_networks
from src.train import train_bp_dfa

(X_train, y_train), (X_test, y_test) = load_mnist()

bp_net, dfa_net, layer_sizes = create_networks(
    width=200,
    depth=3,
    lr_bp=0.005,
    lr_dfa=0.01,
    seed=0,
    feedback_scale=0.1,
)

metrics = train_bp_dfa(
    bp_net,
    dfa_net,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=30,
    batch_size=64,
    log_weights=True,
    plot=True,
    plot_path='./results/bp_dfa_implementation.png'
)
