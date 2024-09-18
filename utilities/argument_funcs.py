import argparse

from .constants import SEPERATOR

def parse_generate_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-midi_root", type=str, default="./data/FolkDB", help="Midi file to prime the generator with")
    parser.add_argument("-output_dir", type=str, default="./generate", help="Folder to write generated midi to")
    parser.add_argument("--force_cpu", action="store_true", help="Forces model to run on a cpu even when gpu is available")

    parser.add_argument("-target_seq_length", type=int, default=1024, help="Target length you'd like the midi to be")
    parser.add_argument("-num_prime", type=int, default=256, help="Amount of messages to prime the generator with")
    parser.add_argument("-model_weights", type=str, default="./checkpoints/MusicMamba.pickle", help="Pickled model weights file saved with torch.save and model.state_dict()")
    parser.add_argument("-beam", type=int, default=0, help="Beam search k. 0 for random probability sample and 1 for greedy")
    
    parser.add_argument("-max_sequence", type=int, default=4096, help="Maximum midi sequence to consider")
    parser.add_argument("-n_layer", type=int, default=2, help="Number of MambaBlock layers to use")
    parser.add_argument("-d_intermediate", type=int, default=0)
    parser.add_argument("-vocab_size", type=int, default=509)
    parser.add_argument("-d_ffn", type=int, default=512, help="Dimension of the feed forward (TransformerBlock's output)")
    parser.add_argument("-n_heads", type=int, default=4, help="Number of heads to use for multi-head attention")
    parser.add_argument("-d_model", type=int, default=1024, help="Dimension of the model (output dim of embedding layers, etc.)")

    return parser.parse_args()


def print_generate_args(args):

    print(SEPERATOR)
    print("midi_root:", args.midi_root)
    print("output_dir:", args.output_dir)
    print("force_cpu:", args.force_cpu)
    print("")
    print("target_seq_length:", args.target_seq_length)
    print("num_prime:", args.num_prime)
    print("model_weights:", args.model_weights)
    print("beam:", args.beam)
    print("")
    print("max_sequence:", args.max_sequence)
    print("n_layer:", args.n_layer)
    print("d_model:", args.d_model)
    print("")
    print(SEPERATOR)
    print("")


def write_model_params(args, output_file):

    o_stream = open(output_file, "w")

    o_stream.write("lr: " + str(args.lr) + "\n")
    o_stream.write("ce_smoothing: " + str(args.ce_smoothing) + "\n")
    o_stream.write("batch_size: " + str(args.batch_size) + "\n")
    o_stream.write("max_sequence: " + str(args.max_sequence) + "\n")
    o_stream.write("n_layer: " + str(args.n_layer) + "\n")
    o_stream.write("d_model: " + str(args.d_model) + "\n")
    
    o_stream.close()
