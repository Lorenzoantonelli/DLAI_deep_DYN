from train_baseline import parse_args, run_with_args


def main():
    args = parse_args()
    args.variant = "pcm16"
    if args.save_dir == "./checkpoints_baseline":
        args.save_dir = "./checkpoints_baseline_16bit"
    run_with_args(args)


if __name__ == "__main__":
    main()
