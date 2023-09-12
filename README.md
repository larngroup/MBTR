# MBTR
Metal-Binding Transformer - Zinc Application

## Usage
- Need to unpack data files and model ckpt
- The architecture supports the use of the Linear Multi-Head Attention arXiv:2006.04768

### Training
```
python main.py --inference_option Train --prot_emb_size 256 --prot_atv_fun gelu --prot_enc_layers 2 --prot_enc_heads 8 --prot_enc_dff 1024 --drop_rate 0.1 --smiles_emb_size 256 --smiles_atv_fun gelu --smiles_enc_layers 2 --smiles_enc_heads 8 --smiles_enc_dff 1024 --num_fc_layer 3 --num_units_fc_layers 128 64 32 --fc_act_fun gelu --batch_size 32 --epochs 500 --loss_opt focal 1 0.40 0.60 --optimizer_fn radam 5e-04 0.9 0.999 1e-08 1e-05 136100 0.02 1e-05
```

### Validation
```
python main.py --inference_option Validation --prot_emb_size 128 256 --prot_atv_fun gelu --prot_enc_layers 2 3 4 --prot_enc_heads 4 8 --prot_enc_dff 512 1024 2048 --drop_rate 0.1 --smiles_emb_size 128 256 --smiles_atv_fun gelu --smiles_enc_layers 2 3 4 --smiles_enc_heads 4 8 --smiles_enc_dff 512 1024 2048 --num_fc_layer 2 3 4 --num_units_fc_layers 1024 512 256 128 64 32 16 --fc_act_fun gelu --batch_size 32 64 --epochs 500 --loss_opt focal 1 0.40 0.60 --loss_opt standard 0.40 0.60 --optimizer_fn radam 5e-04 0.9 0.999 1e-08 1e-05 136100 0.02 1e-05
```

### Evaluation
```
python main.py --inference_opt Evaluation
```

### Binding Vector Prediction
```
python binding_vectors.py --prot_id_list_bind_vector P31460 A0A4Y3TY52 Q19AK4 --prot_seq_list_bind_vector MRQVDAATHGGRAVIELREKILSGELPGGMRLFEVSTAELLDISRTPVREALSRLTEEGLLNRLPGGGFVVRRFGFADVVDAIEVRGVMEGTAARLAAERGVSKVALEEIDATVQQLDLCFGDRVDDVDFDGYAALNRIFHHQLAALCGSEMIRREVERASSLPFASPSAFLPDKANIGAFRRSLRGAQEQHKAIVAAIVAREGARAEAVAREHSRTARTNLEYMIREAPELIAQVPGLALISD --smiles_list_bind_vector [Zn+2] [Zn+2] [Zn+2] [Zn+2]
```
