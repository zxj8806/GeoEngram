import argparse
import torch

from Geo.reservoir import DifferentiableReservoir
from Geo.training.supervised import (
    make_mnist_poisson_loaders,
    make_mnist_seq_loaders,
    train_epoch,
    eval_epoch,
    ReservoirRateClassifier,
    ReservoirTemporalReadoutClassifier,
    SpikingMLPClassifier,
)
from Geo.training.continual import run_continual_learning
from Geo.models import MixedCurvProtoHead


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="reservoir_seq", choices=["mlp", "reservoir_seq", "reservoir_rate"])
    parser.add_argument("--encoding", type=str, default="seq", choices=["seq", "poisson"])
    parser.add_argument("--in_size", type=int, default=1)
    parser.add_argument("--permute", action="store_true")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cl", action="store_true", default=True)
    parser.add_argument("--cl_benchmark", type=str, default="psmnist", choices=["psmnist", "splitmnist"])
    parser.add_argument("--n_tasks", type=int, default=5)
    parser.add_argument("--digits_per_task", type=int, default=2)
    parser.add_argument("--reset_opt_each_task", action="store_true")
    parser.add_argument("--out_dir", type=str, default="./cl_logs")

    parser.add_argument("--cl_use_replay", action="store_true", default=True)
    parser.add_argument("--cl_no_distill", action="store_true")
    parser.add_argument("--cl_replay_per_class", type=int, default=20)
    parser.add_argument("--cl_replay_batch", type=int, default=64)
    parser.add_argument("--cl_replay_weight", type=float, default=1.0)
    parser.add_argument("--cl_geo_weight", type=float, default=0.5)
    parser.add_argument("--cl_route_weight", type=float, default=0.5)
    parser.add_argument("--cl_temp_weight", type=float, default=0.25)
    parser.add_argument("--cl_memory_warmup_batches", type=int, default=32)

    parser.add_argument("--cl_use_gen_replay", action="store_true")
    parser.add_argument("--cl_gen_replay_batch", type=int, default=64)
    parser.add_argument("--cl_gen_replay_weight", type=float, default=1.0)
    parser.add_argument("--cl_gen_route_weight", type=float, default=0.5)
    parser.add_argument("--cl_gen_noise", type=int, default=32)
    parser.add_argument("--cl_gen_cond", type=int, default=64)
    parser.add_argument("--cl_gen_hidden", type=int, default=256)
    parser.add_argument("--cl_gen_epochs", type=int, default=20)
    parser.add_argument("--cl_gen_batch", type=int, default=128)
    parser.add_argument("--cl_gen_lr", type=float, default=1e-3)

    parser.add_argument("--cl_route_overlap_topk", type=int, default=64)
    parser.add_argument("--cl_route_sep_weight", type=float, default=0.3)
    parser.add_argument("--cl_route_sep_margin", type=float, default=0.5)
    parser.add_argument("--cl_gate_sparse_weight", type=float, default=1e-3)
    parser.add_argument("--cl_gate_overlap_weight", type=float, default=5e-2)
    parser.add_argument("--cl_gate_binary_weight", type=float, default=1e-3)
    parser.add_argument("--cl_gate_smooth_weight", type=float, default=1e-2)

    parser.add_argument("--head", type=str, default="mixedcurv", choices=["linear", "mixedcurv"])
    parser.add_argument("--readout_mode", type=str, default="temporal", choices=["temporal", "rate"])
    parser.add_argument("--mc_de", type=int, default=32)
    parser.add_argument("--mc_dp", type=int, default=16)
    parser.add_argument("--mc_ds", type=int, default=16)
    parser.add_argument("--mc_temp", type=float, default=0.3)
    parser.add_argument("--mc_init_radius", type=float, default=1.0)
    parser.add_argument("--mc_fixed_radius", action="store_true", default=True)
    parser.add_argument("--mc_route_topk", type=int, default=64)

    parser.add_argument("--engram_bins", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)

    parser.add_argument("--mem_thr", type=float, default=0.3)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--lens", type=float, default=0.2)

    parser.add_argument("--cube", type=int, default=8)
    parser.add_argument("--w_scale_latent", type=float, default=0.15)
    parser.add_argument("--w_scale_in", type=float, default=0.5)
    parser.add_argument("--train_w_in", action="store_true", default=True)
    parser.add_argument("--train_w_latent", action="store_true")
    parser.add_argument("--res_max_tasks", type=int, default=8)
    parser.add_argument("--gate_floor", type=float, default=0.02)
    parser.add_argument("--gate_k_frac", type=float, default=0.15)

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "mlp":
        if args.encoding != "seq":
            raise ValueError("")
        if args.head != "linear":
            raise ValueError("")
        model = SpikingMLPClassifier(in_size=args.in_size, hidden=(256, 512, 256), device=device).to(device)
        params = list(model.parameters())
    else:
        if args.model == "reservoir_seq" and args.encoding != "seq":
            raise ValueError("")
        if args.model == "reservoir_rate" and args.encoding != "poisson":
            raise ValueError("")

        neurons = args.cube ** 3
        print(f"Reservoir neurons: {neurons}")
        inputs = args.in_size if args.encoding == "seq" else 784

        res = DifferentiableReservoir(
            cube_shape=(args.cube, args.cube, args.cube),
            inputs=inputs,
            w_scale_latent=args.w_scale_latent,
            w_scale_in=args.w_scale_in,
            train_w_in=args.train_w_in,
            train_w_latent=args.train_w_latent,
            max_tasks=max(int(args.res_max_tasks), int(args.n_tasks)),
            task_gate_floor=args.gate_floor,
            gate_k_frac=args.gate_k_frac,
            device=device,
        )
        res.summary()

        readout = None
        if args.head == "mixedcurv":
            readout = MixedCurvProtoHead(
                in_dim=res.n_neurons * int(args.engram_bins),
                n_classes=10,
                d_e=args.mc_de,
                d_p=args.mc_dp,
                d_s=args.mc_ds,
                init_radius=args.mc_init_radius,
                learn_radius=not args.mc_fixed_radius,
                temp=args.mc_temp,
                route_topk=args.mc_route_topk,
            ).to(device)
            if args.model == "reservoir_seq" and args.readout_mode == "temporal":
                args.readout_mode = "rate"

        if args.model == "reservoir_seq":
            model = ReservoirTemporalReadoutClassifier(
                reservoir=res,
                n_classes=10,
                readout=readout,
                readout_mode=args.readout_mode,
                engram_bins=args.engram_bins,
            ).to(device)
        else:
            model = ReservoirRateClassifier(
                reservoir=res,
                n_classes=10,
                readout=readout,
                engram_bins=args.engram_bins,
            ).to(device)

        params = list(model.readout.parameters())
        params += [res.task_gate_logits]
        if args.train_w_in:
            params += [res.w_in]
        if args.train_w_latent:
            params += [res.w_latent]

    def make_optimizer():
        return torch.optim.Adam(params, lr=args.lr)

    forward_kwargs = dict(mem_thr=args.mem_thr, decay=args.decay, lens=args.lens)

    if args.cl:
        if args.encoding != "seq":
            raise ValueError("")
        if args.model == "reservoir_rate":
            raise ValueError("")

        use_replay = bool(args.cl_use_replay)
        use_distill = not bool(args.cl_no_distill)
        use_gen_replay = bool(args.cl_use_gen_replay)

        run_continual_learning(
            model=model,
            device=device,
            make_optimizer_fn=make_optimizer,
            forward_kwargs=forward_kwargs,
            n_tasks=args.n_tasks,
            benchmark=args.cl_benchmark,
            base_seed=args.seed,
            digits_per_task=args.digits_per_task,
            epochs_per_task=args.epochs,
            batch_size=args.batch,
            in_size=args.in_size,
            num_workers=args.num_workers,
            limit_train=args.limit_train,
            limit_test=args.limit_test,
            reset_optimizer_each_task=args.reset_opt_each_task,
            out_dir=args.out_dir,
            verbose=True,
            use_replay=use_replay,
            replay_per_class=args.cl_replay_per_class,
            replay_batch_size=args.cl_replay_batch,
            replay_weight=args.cl_replay_weight,
            use_distill=use_distill,
            geo_weight=args.cl_geo_weight,
            route_weight=args.cl_route_weight,
            temp_weight=args.cl_temp_weight,
            memory_warmup_batches=args.cl_memory_warmup_batches,
            use_gen_replay=use_gen_replay,
            gen_replay_batch_size=args.cl_gen_replay_batch,
            gen_replay_weight=args.cl_gen_replay_weight,
            gen_route_weight=args.cl_gen_route_weight,
            gen_noise_dim=args.cl_gen_noise,
            gen_cond_dim=args.cl_gen_cond,
            gen_hidden_dim=args.cl_gen_hidden,
            gen_train_epochs=args.cl_gen_epochs,
            gen_train_batch_size=args.cl_gen_batch,
            gen_lr=args.cl_gen_lr,
            route_overlap_topk=args.cl_route_overlap_topk,
            route_sep_weight=args.cl_route_sep_weight,
            route_sep_margin=args.cl_route_sep_margin,
            gate_sparse_weight=args.cl_gate_sparse_weight,
            gate_overlap_weight=args.cl_gate_overlap_weight,
            gate_binary_weight=args.cl_gate_binary_weight,
            gate_smooth_weight=args.cl_gate_smooth_weight,
        )
        return

    if args.encoding == "seq":
        train_loader, test_loader = make_mnist_seq_loaders(
            batch_size=args.batch,
            in_size=args.in_size,
            permute=args.permute,
            seed=args.seed,
            limit_train=args.limit_train,
            limit_test=args.limit_test,
            num_workers=args.num_workers,
        )
    else:
        train_loader, test_loader = make_mnist_poisson_loaders(
            batch_size=args.batch,
            time_steps=20,
            p_scale=0.25,
            limit_train=args.limit_train,
            limit_test=args.limit_test,
            num_workers=args.num_workers,
        )

    optim = make_optimizer()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_diag = train_epoch(
            model,
            train_loader,
            optim,
            device,
            mem_thr=args.mem_thr,
            decay=args.decay,
            lens=args.lens,
            log_every=200,
            return_diagnostics=True,
        )
        te_loss, te_acc, te_diag = eval_epoch(
            model,
            test_loader,
            device,
            mem_thr=args.mem_thr,
            decay=args.decay,
            lens=args.lens,
            return_diagnostics=True,
        )


if __name__ == "__main__":
    main()