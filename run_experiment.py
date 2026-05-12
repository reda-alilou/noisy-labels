import argparse
import numpy as np
from src.utils import load_cifar10, get_dataloaders, NoisyDataset, save_history
from src.noise import inject_symmetric_noise, inject_asymmetric_noise
from src.train import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_type', type=str, required=True,
                        choices=['clean', 'symmetric', 'asymmetric'])
    parser.add_argument('--noise_rate', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    args = parser.parse_args()

    train_dataset, test_dataset = load_cifar10()
    targets = np.array(train_dataset.targets)

    if args.noise_type == 'clean':
        dataset = train_dataset
        actual_rate = 0.0
        name = 'clean'
    elif args.noise_type == 'symmetric':
        noisy_targets, actual_rate = inject_symmetric_noise(targets, args.noise_rate)
        dataset = NoisyDataset(train_dataset, noisy_targets)
        name = f'symmetric_{int(args.noise_rate*100)}'
    elif args.noise_type == 'asymmetric':
        noisy_targets, actual_rate = inject_asymmetric_noise(targets, args.noise_rate)
        dataset = NoisyDataset(train_dataset, noisy_targets)
        name = f'asymmetric_{int(args.noise_rate*100)}'

    if args.label_smoothing > 0.0:
        name = f'{name}_ls{int(args.label_smoothing*100)}'

    print(f"Noise type: {args.noise_type} | Requested: {args.noise_rate:.0%} | "
          f"Actual: {actual_rate:.4f} | Label smoothing: {args.label_smoothing}")

    _, history = run_experiment(
        train_dataset=dataset,
        test_dataset=test_dataset,
        num_epochs=args.epochs,
        experiment_name=name,
        label_smoothing=args.label_smoothing
    )

    save_history(history, f'results/{name}.json')
    print(f"Saved to results/{name}.json")
    print(f"Best test acc: {max(history['test_acc']):.4f}")


if __name__ == '__main__':
    main()