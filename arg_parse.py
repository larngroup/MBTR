import os
import argparse


def arg_parser():
    """
    Argument Parser Function

    Outputs:
    - FLAGS: arguments object

    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--inference_option',
        type=str,
        help='Train/Validation or Evaluation'
    )

    parser.add_argument(
        '--prot_len',
        type=int,
        default=575,
        help='Protein Sequence Length'
    )

    parser.add_argument(
        '--smiles_len',
        type=int,
        default=5,
        help='SMILES String length'
    )

    parser.add_argument(
        '--prot_dict_len',
        type=int,
        default=16693,
        help='Protein Dictionary Size'
    )

    parser.add_argument(
        '--smiles_dict_len',
        type=int,
        default=45,
        help='Protein Dictionary Size'
    )

    parser.add_argument(
        '--prot_atv_fun',
        type=str,
        nargs='+',
        help='Activation Function'
    )

    parser.add_argument(
        '--prot_emb_size',
        type=int,
        nargs='+',
        help='Protein Embedding Size'
    )

    parser.add_argument(
        '--prot_enc_layers',
        type=int,
        nargs='+',
        help='Number of Protein Encoder Layers'
    )

    parser.add_argument(
        '--prot_enc_heads',
        type=int,
        nargs='+',
        help='Number of Protein Encoder Heads'
    )

    parser.add_argument(
        '--prot_enc_dff',
        type=int,
        nargs='+',
        help='Hidden Size Protein Encoder'
    )

    parser.add_argument(
        '--drop_rate',
        type=float,
        nargs='+',
        help='Dropout Rate Value'
    )

    parser.add_argument(
        '--prot_dim_k',
        type=int,
        default=0,
        help='Linear Attention Dimension Reduction Value'
    )

    parser.add_argument(
        '--prot_param_share',
        type=str,
        default="",
        help='Linear Attention Parameter Sharing Option'
    )

    parser.add_argument(
        '--prot_full_attention',
        type=int,
        default=1,
        help='Attention Mode Option: Full (True) or Linear (False)'
    )

    parser.add_argument(
        '--prot_return_intermediate',
        type=int,
        default=0,
        help='Return Encoder Intermediate Layers Output + Attention weights'
    )

    parser.add_argument(
        '--smiles_atv_fun',
        type=str,
        nargs='+',
        help='Activation Function'
    )

    parser.add_argument(
        '--smiles_emb_size',
        type=int,
        nargs='+',
        help='SMILES Embedding Size'
    )

    parser.add_argument(
        '--smiles_enc_layers',
        type=int,
        nargs='+',
        help='Number of SMILES Encoder Layers'
    )

    parser.add_argument(
        '--smiles_enc_heads',
        type=int,
        nargs='+',
        help='Number of SMILES Encoder Heads'
    )

    parser.add_argument(
        '--smiles_enc_dff',
        type=int,
        nargs='+',
        help='Hidden Size SMILES Encoder'
    )

    parser.add_argument(
        '--smiles_dim_k',
        type=int,
        default=0,
        help='Linear Attention Dimension Reduction Value'
    )

    parser.add_argument(
        '--smiles_param_share',
        type=str,
        default="",
        help='Linear Attention Parameter Sharing Option'
    )

    parser.add_argument(
        '--smiles_full_attention',
        type=int,
        default=1,
        help='Attention Mode Option: Full (True) or Linear (False)'
    )

    parser.add_argument(
        '--smiles_return_intermediate',
        type=int,
        default=0,
        help='Return Encoder Intermediate Layers Output + Attention weights'
    )

    parser.add_argument(
        '--optimizer_fn',
        type=str,
        nargs='+',
        action='append',
        help='Optimizer Function Parameters'
    )
    
    parser.add_argument(
        '--loss_opt',
        type=str,
        nargs='+',
        action='append',
        help='Loss Function Option'
    )

    parser.add_argument(
        '--num_fc_layer',
        type=int,
        nargs='+',
        help='Number of FC Dense Layers'
    )

    parser.add_argument(
        '--num_units_fc_layers',
        type=int,
        nargs='+',
        action='append',
        help='FC Dense Layers Units'

    )

    parser.add_argument(
        '--fc_act_fun',
        type=str,
        nargs='+',
        help='FC Activation Function'
    )

    parser.add_argument(
        '--output_act_fun',
        type=str,
        default='linear',
        help='Output Dense Layer Activation Function'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        nargs='+',
        help='Batch Size'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        nargs='+',
        help='Number of epochs'
    )

    parser.add_argument(
        '--log_dir',
        type=str,
        default="",
        help='Logging File Directory'
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default="",
        help='Checkpoint Files Directory'
    )

    parser.add_argument(
        '--prot_seq_list_bind_vector',
        type=str,
        nargs='+',
        help='Protein Sequence list to get bind vector'
    )

    parser.add_argument(
        '--prot_id_list_bind_vector',
        type=str,
        nargs='+',
        help='Protein ID list to get bind vector'
    )

    parser.add_argument(
        '--smiles_list_bind_vector',
        type=str,
        nargs='+',
        help='SMILES list to get bind vector'
    )

    parser.add_argument(
        '--bind_results_path',
        type=str,
        default='./results',
        help='Binding Vector Results Directory'
    )

    FLAGS, _ = parser.parse_known_args()

    return FLAGS


def logging(FLAGS, msg):
    """
    Logging function to update the log file

    Args:
    - FLAGS: arguments object
    - msg [str]: info to add to the log file

    """

    fpath = os.path.join(FLAGS.log_dir, "log.txt")
    with open(fpath, 'a') as fw:
        fw.write("%s\n" % msg)

    print("----------//----------")
    print(msg)
    print("----------//----------")
