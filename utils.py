def update_params(model, lr_inner, parameters, source_params):
    for tgt, src in zip(parameters, source_params):
        name_t, param_t = tgt
        grad = src
        tmp = param_t - lr_inner * grad
        set_param(model, name_t, tmp)
)

def set_param(curr_mod, name, param):
    
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        setattr(curr_mod, name, param)
        
def detach_params(self):
    for name, param in self.named_params(self):
        self.set_param(self, name, param.detach())   