import numpy as np


CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Semantic confusion pairs for asymmetric noise
ASYMMETRIC_CONFUSION = {
    0: 2,   # airplane  → bird
    2: 0,   # bird      → airplane
    3: 5,   # cat       → dog
    5: 3,   # dog       → cat
    4: 7,   # deer      → horse
    7: 4,   # horse     → deer
    1: 9,   # automobile→ truck
    9: 1,   # truck     → automobile
    # frog (6) and ship (8) have no natural pair, left clean
}


def inject_symmetric_noise(targets, noise_rate, num_classes=10, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    noisy_targets = targets.copy()

    num_noisy = int(noise_rate * len(targets))
    noisy_indices = rng.choice(len(targets), num_noisy, replace=False)

    for idx in noisy_indices:
        original = targets[idx]
        other_classes = [c for c in range(num_classes) if c != original]
        noisy_targets[idx] = rng.choice(other_classes)

    actual_rate = np.mean(noisy_targets != targets)
    return noisy_targets, actual_rate


def inject_asymmetric_noise(targets, noise_rate, seed=42):
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    noisy_targets = targets.copy()

    for idx, label in enumerate(targets):
        if label in ASYMMETRIC_CONFUSION:
            if rng.random() < noise_rate:
                noisy_targets[idx] = ASYMMETRIC_CONFUSION[label]

    actual_rate = np.mean(noisy_targets != targets)
    return noisy_targets, actual_rate


def print_noise_stats(original, noisy, num_classes=10):
    original = np.array(original)
    noisy = np.array(noisy)
    total_flipped = np.sum(original != noisy)
    print(f"Total samples     : {len(original)}")
    print(f"Flipped labels    : {total_flipped}")
    print(f"Actual noise rate : {total_flipped / len(original):.4f}")
    print()
    print("Per-class flip count:")
    for c in range(num_classes):
        mask = original == c
        flipped = np.sum(original[mask] != noisy[mask])
        print(f"  {CIFAR10_CLASSES[c]:12s}: {flipped} / {mask.sum()} flipped")