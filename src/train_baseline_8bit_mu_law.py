from train_baseline import parse_args, run_with_args


def main():
    args = parse_args()
    args.variant = "mu_law_8bit"
    if args.save_dir == "./checkpoints_baseline":
        args.save_dir = "./checkpoints_baseline_8bit_mu_law"
    run_with_args(args)


if __name__ == "__main__":
    main()
