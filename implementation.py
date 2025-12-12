# implementation.py
from src.utils import set_seed
from data.data import load_mnist
from src.model import create_networks
from src.experiment import run_bp_dfa_experiment


def main():
    # -------------------------
    # Hyperparameters
    # -------------------------
    run_name       = "better_run_test"        # name of directory in results that stores this run
    seed           = 42                     # random seed for reproducibility
    width          = 800                    # width of MLP
    depth          = 2                      # depth of MLP
    lr_bp          = 0.001                  # learning rate of BP
    lr_dfa         = 0.003                   # learning rate of DFA
    batch_size     = 128                     # batch size for MNIST/CIFAR
    epochs         = 60                     # epochs
    feedback_scale = 0.03                   # scale of the random matrix B in DFA, lower = more stable but learns slower
    input_dim      = 784                    # 784 for MNIST
    output_dim     = 10                     # 10 for MNIST

    # set global seed for reproducibility
    set_seed(seed)

    # load data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # build networks using the SAME hyperparameters
    bp_net, dfa_net, layer_sizes = create_networks(
        width=width,
        depth=depth,
        lr_bp=lr_bp,
        lr_dfa=lr_dfa,
        seed=None,            # or keep a per-model seed if you want
        feedback_scale=feedback_scale,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    # pack everything into a config dict for logging
    config = {
        "run_name": run_name,
        "seed": seed,
        "layer_sizes": layer_sizes,
        "width": width,
        "depth": depth,
        "lr_bp": lr_bp,
        "lr_dfa": lr_dfa,
        "batch_size": batch_size,
        "epochs": epochs,
        "feedback_scale": feedback_scale,
        "input_dim": input_dim,
        "output_dim": output_dim,
    }

    # run experiment (auto-saves config, metrics, plots, checkpoints)
    run_dir, metrics = run_bp_dfa_experiment( # refers to experiment.py
        config,
        bp_net,
        dfa_net,
        X_train,
        y_train,
        X_test,
        y_test,
        base_dir="results",
    )

    print(f"Finished run in: {run_dir}")
    print(f"Final BP test err:  {metrics['test_err_bp'][-1]:.2f}%")
    print(f"Final DFA test err: {metrics['test_err_dfa'][-1]:.2f}%")


if __name__ == "__main__":
    main()
