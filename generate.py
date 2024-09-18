import torch
import torch.nn as nn
import os
import random

from utilities.argument_funcs import parse_generate_args, print_generate_args
from dataset import create_epiano_datasets, process_midi

from model import MusicMamba
from utilities.constants import *
from utilities.device import get_device, use_cuda, cpu_device

from midi_tokenize.remi.vocab import Vocab

myvocab = Vocab()

def generate(model, primer=None, mode_mask=None, target_seq_length=1024, beam=0, beam_chance=1.0):
        print("Generating sequence of max length:", target_seq_length)

        gen_seq = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())
        gen_mode_mask = torch.full((1,target_seq_length), TOKEN_PAD, dtype=TORCH_LABEL_TYPE, device=get_device())

        num_primer = len(primer)
        gen_seq[..., :num_primer] = primer.type(TORCH_LABEL_TYPE).to(get_device())
        gen_mode_mask[..., :num_primer] = mode_mask.type(TORCH_LABEL_TYPE).to(get_device())

        softmax    = nn.Softmax(dim=-1)
        
        cur_i = num_primer
        while(cur_i < target_seq_length):
            y = softmax(model.forward(gen_seq[..., :cur_i], gen_mode_mask[..., :cur_i]))[..., :TOKEN_END]
            token_probs = y[:, cur_i-1, :]

            if(beam == 0):
                beam_ran = 2.0
            else:
                beam_ran = random.uniform(0,1)

            if(beam_ran <= beam_chance):
                token_probs = token_probs.flatten()
                top_res, top_i = torch.topk(token_probs, beam)

                beam_rows = top_i // VOCAB_SIZE
                beam_cols = top_i % VOCAB_SIZE

                gen_seq = gen_seq[beam_rows, :]
                gen_seq[..., cur_i] = beam_cols

            else:
                distrib = torch.distributions.categorical.Categorical(probs=token_probs)
                next_token = distrib.sample()
                gen_seq[:, cur_i] = next_token

                if(next_token == TOKEN_END):
                    print("Model called end of sequence at:", cur_i, "/", target_seq_length)
                    break

            cur_i += 1
            if(cur_i % 500 == 0):
                print(cur_i, "/", target_seq_length)

        return gen_seq[:, :cur_i]


def main():
    args = parse_generate_args()
    print_generate_args(args)

    if(args.force_cpu):
        use_cuda(False)
        print("WARNING: Forced CPU usage, expect model to perform slower")
        print("")

    os.makedirs(args.output_dir, exist_ok=True)

    _, _, dataset = create_epiano_datasets(args.midi_root, args.num_prime, random_seq=False)

    f = str(random.randrange(len(dataset)))

    if(f.isdigit()):
        idx = int(f)
        primer, _, mode_mask  = dataset[idx]
        primer = primer.to(get_device())
        mode_mask = mode_mask.to(get_device())

        print("Using primer index:", idx, "(", dataset.data_files[idx], ")")

    else:
        raw_mid = myvocab.midi2REMI(f, Mode_label=False)
        if(len(raw_mid) == 0):
            print("Error: No midi messages in primer file:", f)
            return
        remi_tokens = torch.tensor(raw_mid["remi_tokens"], dtype=TORCH_LABEL_TYPE, device=cpu_device())
        mode_mask = torch.tensor(raw_mid["mode_mask"], dtype=TORCH_LABEL_TYPE, device=cpu_device())

        primer, _, mode_mask, downbeats_ids  = process_midi(remi_tokens, mode_mask, args.num_prime, random_seq=False)
        primer = torch.tensor(primer, dtype=TORCH_LABEL_TYPE, device=get_device())
        mode_mask = torch.tensor(mode_mask, dtype=TORCH_LABEL_TYPE, device=get_device())

        print("Using primer file:", f)

    model = MusicMamba(d_model=args.d_model, d_ffn=args.d_ffn, n_heads=args.n_heads, d_intermediate=args.d_intermediate, n_layer=args.n_layer, vocab_size=args.vocab_size, dtype=torch.float32, device=get_device())
    model.to(get_device())

    model.load_state_dict(torch.load(args.model_weights))

    f_path = os.path.join(args.output_dir, "primer.mid")

    model.eval()
    with torch.set_grad_enabled(False):
        rand_seq = generate(model, primer[:args.num_prime], mode_mask, args.target_seq_length, beam=0)
        f_path = os.path.join(args.output_dir, "rand.mid")
        myvocab.REMI2MIDI(rand_seq[0].cpu().numpy(), f_path)


if __name__ == "__main__":
    main()
