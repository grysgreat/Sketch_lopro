@torch.no_grad()
def scale_quant_only(layer, WR, input_feat, w_bit=4, group_size = 128, index = 1):
    def get_module(root_module, module_path):
        current = root_module
        for part in module_path.split("."):
            current = getattr(current, part)
        return current    
    quant_list = ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj"]
    for name in quant_list:
        if name in input_feat:    
            module = get_module(layer, name)
            qres = scale_quant(layer.self_attn.q_proj ,WR[name], input_feat[name] , w_bit, group_size ,max_clip=0.5)
            module.register_parameter('weight', nn.Parameter(WR[name].cpu()+qres.cpu(), requires_grad=False))
    return layer

