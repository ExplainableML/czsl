import torch
import torch.optim as optim

from models.image_extractor import get_image_extractor
from models.visual_product import VisualProductNN
from models.manifold_methods import RedWine, LabelEmbedPlus, AttributeOperator
from models.modular_methods import GatedGeneralNN
from models.graph_method import GraphFull
from models.symnet import Symnet
from models.compcos import CompCos

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def configure_model(args, dataset):
    image_extractor = None
    is_open = False

    if args.model == 'visprodNN':
        model = VisualProductNN(dataset, args)
    elif args.model == 'redwine':
        model = RedWine(dataset, args)
    elif args.model == 'labelembed+':
        model = LabelEmbedPlus(dataset, args)
    elif args.model == 'attributeop':
        model = AttributeOperator(dataset, args)
    elif args.model == 'tmn':
        model = GatedGeneralNN(dataset, args, num_layers=args.nlayers, num_modules_per_layer=args.nmods)
    elif args.model == 'symnet':
        model = Symnet(dataset, args)
    elif args.model == 'graphfull':
        model = GraphFull(dataset, args)
    elif args.model == 'compcos':
        model = CompCos(dataset, args)
        if dataset.open_world and not args.train_only:
            is_open = True
    else:
        raise NotImplementedError

    model = model.to(device)

    if args.update_features:
        print('Learnable image_embeddings')
        image_extractor = get_image_extractor(arch = args.image_extractor, pretrained = True)
        image_extractor = image_extractor.to(device)

    # configuring optimizer
    if args.model=='redwine':
        optim_params = filter(lambda p: p.requires_grad, model.parameters())
    elif args.model=='attributeop':
        attr_params = [param for name, param in model.named_parameters() if 'attr_op' in name and param.requires_grad]
        other_params = [param for name, param in model.named_parameters() if 'attr_op' not in name and param.requires_grad]
        optim_params = [{'params':attr_params, 'lr':0.1*args.lr}, {'params':other_params}]
    elif args.model=='tmn':
        gating_params = [
            param for name, param in model.named_parameters()
            if 'gating_network' in name and param.requires_grad
        ]
        network_params = [
            param for name, param in model.named_parameters()
            if 'gating_network' not in name and param.requires_grad
        ]
        optim_params = [
            {
                'params': network_params,
            },
            {
                'params': gating_params,
                'lr': args.lrg
            },
        ]
    else:
        model_params = [param for name, param in model.named_parameters() if param.requires_grad]
        optim_params = [{'params':model_params}]
    if args.update_features:
        ie_parameters = [param for name, param in image_extractor.named_parameters()]
        optim_params.append({'params': ie_parameters,
                            'lr': args.lrg})
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    model.is_open = is_open

    return image_extractor, model, optimizer